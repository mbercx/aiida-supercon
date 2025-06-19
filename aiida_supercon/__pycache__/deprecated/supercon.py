# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from .epw import EpwWorkChain
from aiida.engine import calcfunction

from scipy.interpolate import interp1d
import numpy

load_profile()

@calcfunction
def stash_to_remote(stash_data: orm.RemoteStashFolderData) -> orm.RemoteData:
    """Convert a ``RemoteStashFolderData`` into a ``RemoteData``."""

    if stash_data.get_attribute("stash_mode") != "copy":
        raise NotImplementedError("Only the `copy` stash mode is supported.")

    remote_data = orm.RemoteData()
    remote_data.set_attribute(
        "remote_path", stash_data.get_attribute("target_basepath")
    )
    remote_data.computer = stash_data.computer

    return remote_data


@calcfunction
def split_list(list_node: orm.List) -> dict:
    return {f'el_{no}': orm.Float(el) for no, el in enumerate(list_node.get_list())}

@calcfunction
def calculate_tc(max_eigenvalue: orm.XyData) -> orm.Float:
    me_array = max_eigenvalue.get_array('max_eigenvalue')
    if me_array[:, 1].max() < 1.0:
        return orm.Float(0.0)
    else:
        return orm.Float(float(interp1d(me_array[:, 1], me_array[:, 0])(1.0)))

@calcfunction
def calculate_Allen_Dynes_tc(a2f: orm.ArrayData, mustar: orm.Float) -> orm.Float:
    w        = a2f.get_array('frequency')
    # Here we preassume that there are 10 smearing values for a2f calculation
    spectral = a2f.get_array('a2f')[:, 9]   
    mev2K    = 11.604525006157

    _lambda  = 2*numpy.trapz(numpy.divide(spectral, w), x=w)

    # wlog =  np.exp(np.average(np.divide(alpha, w), weights=np.log(w)))
    wlog     =  numpy.exp(2/_lambda*numpy.trapz(numpy.multiply(numpy.divide(spectral, w), numpy.log(w)), x=w))

    Tc = wlog/1.2*numpy.exp(-1.04*(1+_lambda)/(_lambda-mustar*(1+0.62*_lambda))) * mev2K


    return orm.Float(Tc)

class SuperConWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the electron-phonon coupling."""

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('epw_folder', valid_type=(orm.RemoteData, orm.RemoteStashFolderData))
        spec.input('interpolation_distance', valid_type=(orm.Float, orm.List))
        spec.input('convergence_threshold', valid_type=orm.Float, required=False)
        spec.input('converged_workchain_pk', valid_type=orm.Int, required=False, 
            help='This parameter is for test purposes only. It should not be abused.'
            )
        spec.input('always_run_final', valid_type=orm.Bool, default=lambda: orm.Bool(False))

        spec.expose_inputs(
            EpwWorkChain, namespace='epw', exclude=(
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the interpolation `EpwCalculation`s.'
            }
        )

        spec.expose_inputs(
            EpwCalculation, namespace='epw_interp', exclude=(
                'parent_folder_ph', 'parent_folder_nscf', 'kfpoints', 'qfpoints'
            ),
            namespace_options={
                'help': 'Inputs for the interpolation `EpwCalculation`s.'
            }
        )
        spec.expose_inputs(
            EpwCalculation, namespace='epw_final', exclude=(
                'parent_folder_ph', 'parent_folder_nscf', 'kfpoints', 'qfpoints'
            ),
            namespace_options={
                'help': 'Inputs for the final `EpwCalculation`.'
            }
        )
        spec.expose_inputs(
            EpwCalculation, namespace='epw_aniso', exclude=(
                'parent_folder_ph', 'parent_folder_nscf', 'kfpoints', 'qfpoints'
            ),
            namespace_options={
                'help': 'Inputs for the aniso `EpwCalculation`.'
            }
        )
        spec.outline(
            cls.setup,
            if_(cls.should_run_epw)(
                cls.run_epw,
                cls.inspect_epw,
            ),
            while_(cls.should_run_conv)(
                cls.generate_reciprocal_points,
                cls.run_interp_epw,
                cls.inspect_interp_epw,
            ),
            if_(cls.should_run_final)(
                cls.final_epw,
            ),
            if_(cls.should_run_aniso)(
                cls.aniso_epw,
            ),
            cls.results
        )
        spec.output('parameters', valid_type=orm.Dict,
                    help='The `output_parameters` output node of the final EPW calculation.')
        spec.output('max_eigenvalue', valid_type=orm.XyData,
                    help='The temperature dependence of the max eigenvalue for the final EPW.')
        spec.output('a2f', valid_type=orm.XyData,
                    help='The contents of the `.a2f` file for the final EPW.')
        spec.output('Tc', valid_type=orm.Float,
                    help='The isotropic linearised Eliashberg Tc interpolated from the max eigenvalue curve.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_EPW',
            message='The `epw.x` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_EPW_INTERP',
            message='The interpolation `epw.x` sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_EPW_FINAL',
            message='The final `epw.x` sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_EPW_ANISO',
            message='The aniso `epw.x` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'supercon.yaml'

    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            parent_epw,
            parent_final,
            protocol=None, 
            overrides=None, 
            epw_folder=None, 
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()

        if not parent_epw:
            builder.epw = EpwWorkChain.get_builder_from_protocol(
                codes=codes,
                structure=structure,
                protocol=protocol,
                overrides=overrides,
                **kwargs
            )
        else:
            # TODO: Add check to make sure epw_folder is on same computer as epw_code
            builder.parent_epw = parent_epw

        if not parent_final:
            epw_builder = EpwCalculation.get_builder()
            epw_builder.code = codes['epw']
        else:
            builder.parent_final = parent_final

        for epw_namespace in ('epw_interp', 'epw_final', 'epw_aniso'):
            epw_inputs = inputs.get(epw_namespace, None) 

            epw_builder = EpwCalculation.get_builder()

            if epw_namespace == 'epw_interp' and scon_epw_code is not None:
                epw_builder.code = codes['epw']
            else:
                epw_builder.code = codes['epw']

            
            epw_builder.metadata = epw_inputs['metadata']
            if 'settings' in epw_inputs:
                epw_builder.settings = orm.Dict(epw_inputs['settings'])

            builder[epw_namespace]= epw_builder

        if isinstance(inputs['interpolation_distance'], float):
            builder.interpolation_distance = orm.Float(inputs['interpolation_distance'])
        if isinstance(inputs['interpolation_distance'], list):
            qpoints_distance = parent_epw.inputs.qpoints_distance
            interpolation_distance = [v for v in inputs['interpolation_distance'] if v < qpoints_distance / 2]
            builder.interpolation_distance = orm.List(interpolation_distance)

        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])

        builder.always_run_final = orm.Bool(inputs.get('always_run_final', False))
        builder.structure = parent_epw.inputs.structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        intp = self.inputs.get('interpolation_distance')
        if isinstance(intp, orm.List):
            self.ctx.interpolation_list = list(split_list(intp).values())
        else:
            self.ctx.interpolation_list = [intp]

        self.ctx.interpolation_list.sort()
        self.ctx.final_interp = None
        self.ctx.allen_dynes_values = []
        self.ctx.is_converged = False
        self.ctx.degaussq = None

    def should_run_conv(self):
        """Check if the convergence loop should continue or not."""
        if self.inputs.converged_workchain_pk:
            self.report(
                f'An EpwWorkChain/EpwCalculation<{self.inputs.converged_workchain_pk.value}> was provided as `converged_workchain`.'
                f'Convergence is not guaranteed.'
                f'The calculation will continue, but you should check the results.'
                )
            return False
        
        if 'convergence_threshold' in self.inputs:
            try:
                self.ctx.epw_interp[-3].outputs.output_parameters['allen_dynes']  # This is to check that we have at least 3 allen-dynes
                prev_allen_dynes = self.ctx.epw_interp[-2].outputs.output_parameters['allen_dynes']
                new_allen_dynes = self.ctx.epw_interp[-1].outputs.output_parameters['allen_dynes']
                self.ctx.is_converged = (
                    abs(prev_allen_dynes - new_allen_dynes)
                    < self.inputs.convergence_threshold
                )
                self.report(f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {self.ctx.is_converged.value}')
            except (AttributeError, IndexError, KeyError):
                self.report('Not enough data to check convergence.')

        else:
            self.report('No `convergence_threshold` input was provided, convergence automatically achieved.')
            self.ctx.is_converged = True

        return len(self.ctx.interpolation_list) > 0 and not self.ctx.is_converged
    
    def generate_reciprocal_points(self):
        """Generate the qpoints and kpoints meshes for the interpolation."""

        inputs = {
            'structure': self.inputs.structure,
            'distance': self.ctx.interpolation_list.pop(),
            'force_parity': orm.Bool(False),
            'metadata': {
                'call_link_label': 'create_kpoints_from_distance'
            }
        }
        inter_points = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        self.ctx.inter_points = inter_points

    def run_epw(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        inputs = AttributeDict(self.exposed_inputs(EpwWorkChain, namespace='epw'))

        inputs.parent_folder_ph = self.inputs.epw_folder
        inputs.parent_folder_nscf = self.inputs.epw_folder
        inputs.kfpoints = self.ctx.inter_points
        inputs.qfpoints = self.ctx.inter_points

        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw_interp'))

        inputs.parent_folder_epw = self.inputs.epw_folder
        inputs.kfpoints = self.ctx.inter_points
        inputs.qfpoints = self.ctx.inter_points

        try:
            settings = inputs.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = ['aiida.a2f']
        inputs.settings = orm.Dict(settings)

        if self.ctx.degaussq:
            parameters = inputs.parameters.get_dict()
            parameters['INPUTEPW']['degaussq'] = self.ctx.degaussq
            inputs.parameters = orm.Dict(parameters)

        inputs.metadata.call_link_label = 'epw_interp'
        calcjob_node = self.submit(EpwCalculation, **inputs)
        mesh = 'x'.join(str(i) for i in self.ctx.inter_points.get_kpoints_mesh()[0])
        self.report(f'launching interpolation `epw` with PK {calcjob_node.pk} and interpolation mesh {mesh}')

        return ToContext(epw=calcjob_node)

    def inspect_epw(self):
        """Verify that the epw.x workflow finished successfully."""
        epw_calculation = self.ctx.epw

        if not epw_calculation.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {epw_calculation.exit_status}')
            # return self.exit_codes.ERROR_SUB_PROCESS_EPW_INTERP
            self.exit_codes.ERROR_SUB_PROCESS_EPW
        else:
            parameters = epw_calculation.inputs['parameters']
            parameters["INPUTEPW"]["use_ws"] = epw_calculation.inputs.parameters["INPUTEPW"].get("use_ws", False)
            parameters['INPUTEPW']['nbndsub'] = epw_calculation.inputs.parameters['INPUTEPW']['nbndsub']
            if 'bands_skipped' in epw_calculation.inputs.parameters['INPUTEPW']:
                parameters['INPUTEPW']['bands_skipped'] = epw_calculation.inputs.parameters['INPUTEPW'].get('bands_skipped')
            
            epw_builder.kpoints = epw_source.inputs.kpoints
            epw_builder.qpoints = epw_source.inputs.qpoints

            epw_builder.parameters = orm.Dict(parameters)
            
    def run_interp_epw(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw_interp'))

        inputs.parent_folder_epw = self.inputs.epw_folder
        inputs.kfpoints = self.ctx.inter_points
        inputs.qfpoints = self.ctx.inter_points

        try:
            settings = inputs.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = ['aiida.a2f']
        inputs.settings = orm.Dict(settings)

        if self.ctx.degaussq:
            parameters = inputs.parameters.get_dict()
            parameters['INPUTEPW']['degaussq'] = self.ctx.degaussq
            inputs.parameters = orm.Dict(parameters)

        inputs.metadata.call_link_label = 'epw_interp'
        calcjob_node = self.submit(EpwCalculation, **inputs)
        mesh = 'x'.join(str(i) for i in self.ctx.inter_points.get_kpoints_mesh()[0])
        self.report(f'launching interpolation `epw` with PK {calcjob_node.pk} and interpolation mesh {mesh}')

        return ToContext(epw_interp=append_(calcjob_node))

    def inspect_interp_epw(self):
        """Verify that the epw.x workflow finished successfully."""
        epw_calculation = self.ctx.epw_interp[-1]

        if not epw_calculation.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {epw_calculation.exit_status}')
            self.ctx.epw_interp.pop()
            # return self.exit_codes.ERROR_SUB_PROCESS_EPW_INTERP
        else:
            self.ctx.final_interp = self.ctx.inter_points
            try:
                self.report(f"Allen-Dynes: {epw_calculation.outputs.output_parameters['allen_dynes']}")
            except KeyError:
                self.report(f"Could not find Allen-Dynes temperature in parsed output parameters!")

            if self.ctx.degaussq is None:
                frequency = epw_calculation.outputs.a2f.get_array('frequency')
                self.ctx.degaussq = frequency[-1] / 100

    def should_run_final(self):
        """Check if the final ``epw.x`` calculation should be run."""
        # if not self.inputs.always_run_final and 'convergence_threshold' in self.inputs:
        #     return self.ctx.is_converged
        if self.inputs.converged_workchain_pk:
            return True
        
        Tc_allen_dynes = self.ctx.epw_interp[-1].outputs.output_parameters['allen_dynes']

        if self.ctx.final_interp is None and Tc_allen_dynes < 2.0:
            return False

        return True

    def final_epw(self):
        """Run the final ``epw.x`` calculation."""
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw_final'))
        
        if self.inputs.converged_workchain_pk:
            converged_workchain = orm.load_node(self.inputs.converged_workchain_pk.value)
            if converged_workchain.process_class == EpwCalculation:
                parent_folder_epw = converged_workchain.outputs.remote_folder
                if parent_folder_epw.is_cleaned:
                    raise ValueError(
                        'The remote folder of the converged workchain is cleaned.',
                    )
                else:
                    inputs.parent_folder_epw = parent_folder_epw
                    inputs.kfpoints = converged_workchain.inputs.kfpoints
                    inputs.qfpoints = converged_workchain.inputs.qfpoints
                    
                    parameters = inputs.parameters.get_dict()

                    Tc_allen_dynes = converged_workchain.outputs.output_parameters['allen_dynes']

                    if Tc_allen_dynes < 2.0:
                        raise ValueError(
                            f'The provided EpwCalculation<{converged_workchain.pk}> has a Tc < 1.0 K.'
                            f'It should be dropped.'
                        )
                    else:
                        self.report(
                            f'The provided EpwCalculation<{converged_workchain.pk}> has Allen-Dynes Tc = {Tc_allen_dynes} K.'
                        )
                    #We preassume that the isotropic Tc should not exceed 4*Tc of Allen DYnes Tc
                    
                    # parameters['INPUTEPW']['temps'] = f'1 {Tc_allen_dynes*3}'
                    parameters['INPUTEPW']['temps'] = '1 40'
                    parameters['INPUTEPW']['nstemp'] = 8

                    inputs.parameters = orm.Dict(parameters)
                
            elif converged_workchain.process_class == SuperConWorkChain:

                raise NotImplementedError(
                    'A SuperConWorkChain was provided as `converged_workchain`.'
                    'This is not yet implemented.'
                )
            else:
                raise ValueError(
                    'The `converged_workchain` must be an EpwCalculation or SuperConWorkChain.'
                )
            
            
        else:
            
            inputs.parent_folder_epw = self.ctx.epw_interp[-1].outputs.remote_folder

            inputs.kfpoints = self.ctx.final_interp
            inputs.qfpoints = self.ctx.final_interp

            Tc_allen_dynes = self.ctx.epw_interp[-1].outputs.output_parameters['allen_dynes']

            parameters = inputs.parameters.get_dict()

            if self.ctx.degaussq:
                parameters['INPUTEPW']['degaussq'] = self.ctx.degaussq
            
            parameters['INPUTEPW']['temps'] = f'1 {Tc_allen_dynes}'
            inputs.parameters = orm.Dict(parameters)
        

        try:
            settings = inputs.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = ['aiida.dos', 'aiida.a2f*', 'aiida.phdos*']
        inputs.settings = orm.Dict(settings)
            
        inputs.metadata.call_link_label = 'epw_final'

        calcjob_node = self.submit(EpwCalculation, **inputs)
        self.report(f'launching final `epw` {calcjob_node.pk}')

        return ToContext(final_epw=calcjob_node)
    
    def inspect_final_epw(self):
        """Verify that the final epw.x workflow finished successfully."""
        epw_calculation = self.ctx.final_epw

        if not epw_calculation.is_finished_ok:
            self.report(f'Final `epw.x` failed with exit status {epw_calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW_FINAL

    def should_run_aniso(self):
        """Check if the aniso loop should continue or not."""
        Tc_lE = calculate_tc(self.ctx.final_epw.outputs.max_eigenvalue)

        if Tc_lE < 5.0:
            self.report(
                f'Isotropic Tc from {Tc_lE.value} K is less than 5.0 K. Aniso calculation is skipped.'
                )
            return False
        
        self.ctx.Tc_lE = Tc_lE
        
        return self.inputs.run_aniso

    def aniso_epw(self):
        """Run the aniso ``epw.x`` calculation."""
        
        Tc_lE = self.ctx.Tc_lE

        """Run the aniso ``epw.x`` calculation."""
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw_aniso'))

        inputs.parent_folder_epw = self.ctx.final_epw.outputs.remote_folder
        inputs.kfpoints = self.ctx.final_epw.inputs.kfpoints
        inputs.qfpoints = self.ctx.final_epw.inputs.qfpoints

        parameters = inputs.parameters.get_dict()
        
        if self.ctx.degaussq:
            parameters['INPUTEPW']['degaussq'] = self.ctx.degaussq
        
        parameters['INPUTEPW']['temps'] = f'3.5 {Tc_lE.value}'
        inputs.parameters = orm.Dict(parameters)

        try:
            settings = inputs.settings.get_dict()
        except AttributeError:
            settings = {}
            
        settings['ADDITIONAL_RETRIEVE_LIST'] = ['aiida.imag_aniso_*', 'aiida.lambda*']
        
        if parameters['INPUTEPW'].get('lpade', False):
            settings['ADDITIONAL_RETRIEVE_LIST'].extend(('aiida.pade_aniso_gap0*', ))
        
        inputs.settings = orm.Dict(settings)

        inputs.metadata.call_link_label = 'epw_aniso'

        calcjob_node = self.submit(EpwCalculation, **inputs)
        self.report(f'launching aniso `epw` {calcjob_node.pk}')

        return ToContext(aniso_epw=calcjob_node)
    
    def inspect_aniso_epw(self):
        """Verify that the aniso epw.x workflow finished successfully."""
        epw_calculation = self.ctx.aniso_epw

        if not epw_calculation.is_finished_ok:
            self.report(f'Anisotropic `epw.x` failed with exit status {epw_calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_EPW_ANISO
        
    def results(self):
        """TODO"""
        self.out('Tc', self.ctx.Tc_lE)
        self.out('parameters', self.ctx.final_epw.outputs.output_parameters)
        self.out('max_eigenvalue', self.ctx.final_epw.outputs.max_eigenvalue)
        self.out('a2f', self.ctx.final_epw.outputs.a2f)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

