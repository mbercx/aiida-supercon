# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, while_, append_, calcfunction

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

# from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_wannier90_workflows.common.types import WannierProjectionType

from .b2w import EpwB2WWorkChain
from .a2f import EpwA2fWorkChain
from .iso import EpwIsoWorkChain
from .aniso import EpwAnisoWorkChain
from .base import EpwBaseWorkChain

@calcfunction
def split_list(list_node: orm.List) -> dict:
    return {f'el_{no}': orm.Float(el) for no, el in enumerate(list_node.get_list())}

class EpwSuperConWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the electron-phonon coupling."""
    

    _NAMESPACE = 'supercon'
    
    _CONV_NAMESPACE = 'a2f_conv'
    
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'muc'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    _restart_from_ephmat = {
        'INPUTEPW': (
            ('elph',        False),
            ('ep_coupling', False),
            ('ephwrite',    False),
            ('restart',     True),
        )
    }
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.input('interpolation_distances', required=False, valid_type=orm.List)
        spec.input('convergence_threshold', required=False, valid_type=orm.Float)
        spec.input('always_run_final', required=False, valid_type=orm.Bool, default=lambda: orm.Bool(True))    
        
        spec.expose_inputs(
            EpwB2WWorkChain, 
            namespace=EpwA2fWorkChain._B2W_NAMESPACE, 
            exclude=('clean_workdir'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwB2WWorkChain`.'
            }
        )
        
        # Here A2f and A2f_conv are exclusive, they share the same namespace
        spec.expose_inputs(
            EpwA2fWorkChain, 
            namespace=EpwA2fWorkChain._INTP_NAMESPACE, 
            exclude=(
                'clean_workdir', 
                f'{EpwA2fWorkChain._INTP_NAMESPACE}.parent_folder_nscf',
                f'{EpwA2fWorkChain._INTP_NAMESPACE}.parent_folder_chk',
                f'{EpwA2fWorkChain._INTP_NAMESPACE}.parent_folder_ph',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwA2fWorkChain`.'
            }
        )
        spec.expose_inputs(
            EpwIsoWorkChain, 
            namespace=EpwIsoWorkChain._INTP_NAMESPACE, 
            exclude=(
                'clean_workdir', 
                f"{EpwIsoWorkChain._INTP_NAMESPACE}.parent_folder_nscf",
                f"{EpwIsoWorkChain._INTP_NAMESPACE}.parent_folder_chk",
                f"{EpwIsoWorkChain._INTP_NAMESPACE}.parent_folder_ph",
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwIsoWorkChain`.'
            }
        )
        spec.expose_inputs(
            EpwAnisoWorkChain, 
            namespace=EpwAnisoWorkChain._INTP_NAMESPACE, 
            exclude=(
                'clean_workdir', 
                f"{EpwAnisoWorkChain._INTP_NAMESPACE}.parent_folder_nscf",
                f"{EpwAnisoWorkChain._INTP_NAMESPACE}.parent_folder_chk",
                f"{EpwAnisoWorkChain._INTP_NAMESPACE}.parent_folder_ph",
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwAnisoWorkChain`.'
            }
        )
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.prepare_intp,
            while_(cls.should_run_conv)(
                cls.run_conv,
                cls.inspect_conv,
            ),
            if_(cls.should_run_a2f)(
                cls.run_a2f,
                cls.inspect_a2f,
            ),
            if_(cls.should_run_iso)(
                cls.run_iso,
                cls.inspect_iso,
            ),
            cls.run_aniso,
            cls.inspect_aniso,
            cls.results
        )
        # spec.output('parameters', valid_type=orm.Dict,
        #             help='The `output_parameters` output node of the final EPW calculation.')
        # spec.output('Tc_allen_dynes', valid_type=orm.Float, required=False,
        #             help='The Allen-Dynes Tc interpolated from the a2f file.')
        # spec.output('Tc_iso', valid_type=orm.Float, required=False,
        #             help='The isotropic linearised Eliashberg Tc interpolated from the max eigenvalue curve.')
        # spec.output('Tc_aniso', valid_type=orm.Float, required=False,
        #             help='The anisotropic Eliashberg Tc interpolated from the max eigenvalue curve.')

        spec.expose_outputs(
            EpwB2WWorkChain,
            namespace=EpwA2fWorkChain._B2W_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the anisotropic `EpwCalculation`.'
            }
        )
        
        spec.expose_outputs(
            EpwA2fWorkChain,
            namespace=EpwA2fWorkChain._INTP_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwA2fWorkChain`.'
            }
        )
        
        spec.expose_outputs(
            EpwIsoWorkChain,
            namespace=EpwIsoWorkChain._INTP_NAMESPACE,
            exclude=('dos', 'phdos', 'phdos_proj', 'a2f', 'a2f_proj', 'lambda_FS', 'lambda_k_pairs'),
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwIsoWorkChain`.'
            }
        )
        
        spec.expose_outputs(
            EpwAnisoWorkChain,
            namespace=EpwAnisoWorkChain._INTP_NAMESPACE,
            exclude=('dos', 'phdos', 'phdos_proj', 'a2f', 'a2f_proj', 'lambda_FS', 'lambda_k_pairs'),
            namespace_options={
                'required': False,
                'help': 'Outputs from the `EpwAnisoWorkChain`.'
            }
        )

        spec.exit_code(401, 'ERROR_SUB_PROCESS_B2W',
            message='The `b2w` sub process failed')
        spec.exit_code(402, 'ERROR_CONVERGENCE_NOT_REACHED',
            message='The convergence is not reached in current interpolation list.')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_A2F',
            message='The `a2f` sub process failed')
        spec.exit_code(406, 'ERROR_Allen_Dynes_Tc_TOO_LOW',
            message='The Allen-Dynes Tc is too low.')
        spec.exit_code(407, 'ERROR_SUB_PROCESS_ISO',
            message='The `iso` sub process failed')
        spec.exit_code(408, 'ERROR_ISOTROPIC_TC_TOO_LOW',
            message='The isotropic Tc is too low.')
        spec.exit_code(409, 'ERROR_SUB_PROCESS_ANISO',
            message='The `aniso` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._NAMESPACE}.yaml'
    
    @staticmethod
    def get_descendant(
        supercon: orm.WorkChainNode,
        link_label_filter: str
        ) -> orm.WorkChainNode:
        """Get the descendant workchains of the intp workchain."""
        try:
            return supercon.base.links.get_outgoing(
                link_label_filter=link_label_filter
                ).first().node
        except AttributeError:
            return None

    @staticmethod
    def get_descendants(
        supercon: orm.WorkChainNode,
        link_label_filter: str
        ) -> orm.WorkChainNode:
        """Get the descendant workchains of the intp workchain."""
        try:
            return supercon.base.links.get_outgoing(
                link_label_filter=link_label_filter
                ).all()
        except AttributeError:
            return None
    @classmethod
    def get_builder_restart_from_ph(
        cls,
        from_ph_workchain: orm.WorkChainNode,
        protocol=None,
        overrides=None,
        **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        pass
    
    def get_builder_restart_from_b2w(
        cls,
        from_b2w_workchain: orm.WorkChainNode,
        protocol=None,
        overrides=None,
        **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        inputs = cls.get_protocol_inputs(protocol, overrides)
        
        builder = cls.get_builder()
        
        if not from_b2w_workchain or not from_b2w_workchain.process_class == EpwB2WWorkChain:
            raise ValueError('Currently we only accept `EpwB2WWorkChain`')
    
        # b2w_parameters = from_b2w_workchain.inputs.epw.parameters.get_dict()
        
        # parameters = builder.epw.parameters.get_dict()
        
        # for namespace, keyword in cls._blocked_keywords:
        #     if keyword in b2w_parameters[namespace]:
        #         parameters[namespace][keyword] = b2w_parameters[namespace][keyword]
        if from_b2w_workchain.is_finished_ok:
            builder.pop(EpwA2fWorkChain._B2W_NAMESPACE)
            parent_folder_epw = from_b2w_workchain.outputs.epw.remote_folder
        else:
            b2w_builder = EpwB2WWorkChain.get_builder_restart(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(EpwA2fWorkChain._B2W_NAMESPACE, None),
                **kwargs
                )
            
            # Actually there is no exclusion of EpwB2WWorkChain namespace
            # So we need to set the _data manually
            builder[EpwA2fWorkChain._B2W_NAMESPACE]._data = b2w_builder._data
        

        for (epw_namespace, epw_workchain_class) in (
            (EpwA2fWorkChain._INTP_NAMESPACE, EpwA2fWorkChain),
            (EpwIsoWorkChain._INTP_NAMESPACE, EpwIsoWorkChain),
            (EpwAnisoWorkChain._INTP_NAMESPACE, EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_restart_from_b2w(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(epw_namespace, None),
                **kwargs
                )
            
            if epw_workchain_class._B2W_NAMESPACE in epw_builder:
                epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)
                
            builder[epw_namespace]._data = epw_builder._data
        
        if parent_folder_epw:
            builder[EpwA2fWorkChain._INTP_NAMESPACE].parent_folder_epw = parent_folder_epw
        
        builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        
        return builder
    
    @classmethod
    def get_builder_from_a2f(
        cls,
        from_a2f_workchain: orm.WorkChainNode,
        protocol=None,
        overrides=None,
        **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        
        builder = cls.get_builder()
        
        if not from_a2f_workchain or not from_a2f_workchain.process_class == EpwA2fWorkChain:
            raise ValueError('Currently we only accept `EpwA2fWorkChain`')
        
        structure = from_a2f_workchain.inputs.structure
        code = from_a2f_workchain.inputs[EpwA2fWorkChain._INTP_NAMESPACE].epw.code
        
        parent_folder_epw = from_a2f_workchain.outputs.epw.remote_folder
        
        for (epw_namespace, epw_workchain_class) in (
            (EpwIsoWorkChain._INTP_NAMESPACE, EpwIsoWorkChain),
            (EpwAnisoWorkChain._INTP_NAMESPACE, EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder()
            
            intp_builder = EpwBaseWorkChain.get_builder_from_protocol(
                code,
                structure,
                protocol,
                overrides=overrides.get(epw_namespace, None),
                **kwargs)
            
            epw_builder[epw_workchain_class._INTP_NAMESPACE]._data = intp_builder._data
            
            epw_builder[epw_workchain_class._INTP_NAMESPACE].parent_folder_epw = parent_folder_epw
            
            builder[epw_namespace]._data = epw_builder._data
            
        builder.pop('interpolation_distances')
        builder.pop('convergence_threshold')
        builder.pop('always_run_final')
        
        return builder
        
    @classmethod
    def get_builder_restart(
        cls,
        from_supercon_workchain: orm.WorkChainNode,
    ):
        builder = from_supercon_workchain.get_builder_restart()
                
        if not from_supercon_workchain or not from_supercon_workchain.process_class == EpwSuperConWorkChain:
            raise ValueError('Currently we only accept `EpwSuperConWorkChain`')
        
        if from_supercon_workchain.is_finished_ok:
            raise Warning('The `EpwSuperConWorkChain` is already finished.')
        
        try:
            for sub_workchain in (
                EpwA2fWorkChain,
                EpwIsoWorkChain,
                EpwAnisoWorkChain,
                ):
                builder[sub_workchain._INTP_NAMESPACE].pop(sub_workchain._B2W_NAMESPACE)
        except KeyError:
            pass
        
        # Firstly we should check whether we should restart from b2w workchain.
        if EpwA2fWorkChain._B2W_NAMESPACE in from_supercon_workchain.inputs:
            b2w_workchain = EpwSuperConWorkChain.get_descendant(
                from_supercon_workchain,
                EpwA2fWorkChain._B2W_NAMESPACE
                )
                # If the b2w workchain is finished, simply pop it.
            if b2w_workchain.is_finished_ok:
                builder.pop(EpwA2fWorkChain._B2W_NAMESPACE)
                builder[EpwA2fWorkChain._INTP_NAMESPACE][EpwA2fWorkChain._INTP_NAMESPACE].parent_folder_epw = b2w_workchain.outputs.epw.remote_folder
            # If the b2w workchain is not finished, we only need to restart from b2w workchain leaving the following inputs unchanged.
            else:
                b2w_builder = EpwB2WWorkChain.get_builder_restart(
                    from_b2w_workchain=b2w_workchain,
                    )
                
                builder[EpwA2fWorkChain._B2W_NAMESPACE]._data = b2w_builder._data

                return builder
        else:
            builder.pop(EpwA2fWorkChain._B2W_NAMESPACE)
            builder[EpwA2fWorkChain._INTP_NAMESPACE][EpwA2fWorkChain._INTP_NAMESPACE].parent_folder_epw = b2w_workchain.outputs.epw.remote_folder
            
        # If interpolation list is not an input port, it must be that the previous
        # workchain has finished convergence test. Or it start from a given grid without convergence test.
        
        if 'interpolation_distances' in from_supercon_workchain.inputs:
            initial_interpolation_distances = from_supercon_workchain.inputs.interpolation_distances.get_list()
            # Then we need to know whether convergence is finished.
            a2f_conv_workchains = EpwSuperConWorkChain.get_descendants(
                from_supercon_workchain,
                cls._CONV_NAMESPACE
                )
            
            for a2f_conv_workchain in a2f_conv_workchains:
                if a2f_conv_workchain.node.is_finished_ok:
                    initial_interpolation_distances.remove(a2f_conv_workchain.node.inputs.a2f.qfpoints_distance.value)
            
            if len(initial_interpolation_distances) > 0:
                builder.interpolation_distances = orm.List(initial_interpolation_distances)
                return builder
            
            else:
                builder.pop(EpwA2fWorkChain._INTP_NAMESPACE)
                builder.pop('interpolation_distances')
                builder.pop('convergence_threshold')
                builder.pop('always_run_final')
        
        # If there is no interpolation distance, either it's finished or we don't do it at all.
        
        else:
            if EpwA2fWorkChain._INTP_NAMESPACE in from_supercon_workchain.inputs:
                a2f_workchain = EpwSuperConWorkChain.get_descendant(
                    from_supercon_workchain,
                    EpwA2fWorkChain._INTP_NAMESPACE
                    )
                if a2f_workchain and a2f_workchain.is_finished_ok:
                    builder.pop(EpwA2fWorkChain._INTP_NAMESPACE)
                else:
                    a2f_builder = EpwA2fWorkChain.get_builder_restart(
                        from_a2f_workchain=a2f_workchain,
                        )
                    a2f_builder.parent_folder_epw = a2f_workchain.inputs[EpwA2fWorkChain._INTP_NAMESPACE].parent_folder_epw
                    builder[EpwA2fWorkChain._INTP_NAMESPACE]._data = a2f_builder._data
                    return builder
            else:
                builder.pop(EpwA2fWorkChain._INTP_NAMESPACE)
                
        if EpwIsoWorkChain._INTP_NAMESPACE in from_supercon_workchain.inputs:
            iso_workchain = EpwSuperConWorkChain.get_descendant(
                from_supercon_workchain,
                EpwIsoWorkChain._INTP_NAMESPACE
                )
            if iso_workchain and iso_workchain.is_finished_ok:
                builder.pop(EpwIsoWorkChain._INTP_NAMESPACE)
            else:
                iso_builder = EpwIsoWorkChain.get_builder_restart(
                    from_iso_workchain=iso_workchain,
                )
                builder[EpwIsoWorkChain._INTP_NAMESPACE]._data = iso_builder._data
                return builder
        else:
            builder.pop(EpwIsoWorkChain._INTP_NAMESPACE)
            
        if EpwAnisoWorkChain._INTP_NAMESPACE in from_supercon_workchain.inputs:
            aniso_workchain = EpwSuperConWorkChain.get_descendant(
                from_supercon_workchain,
                EpwAnisoWorkChain._INTP_NAMESPACE
                )
            if aniso_workchain and aniso_workchain.is_finished_ok:
                raise Warning('The `EpwSuperConWorkChain` has already finished.')
            else:
                aniso_builder = EpwAnisoWorkChain.get_builder_restart(
                    from_aniso_workchain=aniso_workchain,
                    )
                builder[EpwAnisoWorkChain._INTP_NAMESPACE]._data = aniso_builder._data
                return builder
        else:
            raise Warning('The `EpwSuperConWorkChain` has already finished.')
                
    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            protocol='moderate',
            overrides=None, 
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :TODO:
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)
        args = (codes, structure, protocol)
        
        builder = cls.get_builder()
        
        b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
            *args,
            overrides=inputs.get(EpwA2fWorkChain._B2W_NAMESPACE, None),
            **kwargs
        )

        b2w_builder.w90_intp.pop('open_grid')
        b2w_builder.w90_intp.pop('projwfc')

        builder[EpwA2fWorkChain._B2W_NAMESPACE]._data = b2w_builder._data
            
        for (epw_namespace, epw_workchain_class) in (
            (EpwA2fWorkChain._INTP_NAMESPACE, EpwA2fWorkChain),
            (EpwIsoWorkChain._INTP_NAMESPACE, EpwIsoWorkChain),
            (EpwAnisoWorkChain._INTP_NAMESPACE, EpwAnisoWorkChain),
        ):
            epw_builder = epw_workchain_class.get_builder_from_protocol(
                *args,
                overrides=inputs.get(epw_namespace, None),
                **kwargs
            )
            
            epw_builder.pop(epw_workchain_class._B2W_NAMESPACE)
            
            builder[epw_namespace]._data = epw_builder._data
            
            # builder.structure = structure
            # builder.qfpoints_distance = orm.Float(inputs.get('qfpoints_distance', None))
            # builder.kfpoints_factor = orm.Float(inputs.get('kfpoints_factor', None))
            
        builder.interpolation_distances = orm.List(inputs.get('interpolation_distances', None))
        builder.convergence_threshold = orm.Float(inputs['convergence_threshold'])
        builder.always_run_final = orm.Bool(inputs.get('always_run_final', True))
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        
        return builder

    def validate_inputs(self):
        """Validate the inputs."""
        if 'interpolation_distances' in self.inputs:
            if isinstance(self.inputs.interpolation_distances, orm.List) and len(self.inputs.interpolation_distances) > 1:
                if not self.inputs.convergence_threshold:
                    return self.exit_codes.ERROR_CONVERGENCE_THRESHOLD_NOT_SPECIFIED
            else:
                raise ValueError('The `interpolation_distances` must be a list with at least one element.')
            
            if EpwA2fWorkChain._INTP_NAMESPACE not in self.inputs:
                raise ValueError('The `a2f` must be provided as an input port.')
        else:
            if 'convergence_threshold' in self.inputs:
                raise ValueError('The `convergence_threshold` must not be provided if the `interpolation_distances` is not provided.')

    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        
        if EpwA2fWorkChain._INTP_NAMESPACE in self.inputs:
            self.ctx.inputs_a2f = AttributeDict(self.exposed_inputs(EpwA2fWorkChain, namespace=EpwA2fWorkChain._INTP_NAMESPACE))
        if EpwIsoWorkChain._INTP_NAMESPACE in self.inputs:
            self.ctx.inputs_iso = AttributeDict(self.exposed_inputs(EpwIsoWorkChain, namespace=EpwIsoWorkChain._INTP_NAMESPACE))
        
        self.ctx.inputs_aniso = AttributeDict(self.exposed_inputs(EpwAnisoWorkChain, namespace=EpwAnisoWorkChain._INTP_NAMESPACE))
        
        if hasattr(self.inputs, 'interpolation_distances'):
            self.report("Will check convergence")
            self.ctx.interpolation_distances = self.inputs.get('interpolation_distances').get_list()
            self.ctx.interpolation_distances.sort()
            self.ctx.do_conv = True
            self.ctx.final_a2f = None
            self.ctx.allen_dynes_values = []
            self.ctx.is_converged = False
        else:
            self.ctx.do_conv = False

    def should_run_b2w(self):
        """Check if the b2w workflow should continue or not."""

        return EpwA2fWorkChain._B2W_NAMESPACE in self.inputs
    
    def run_b2w(self):
        """Run the b2w workflow."""
        inputs = AttributeDict(
            self.exposed_inputs(
                EpwB2WWorkChain, 
                namespace=EpwA2fWorkChain._B2W_NAMESPACE
            )
        )
        inputs.metadata.call_link_label = EpwA2fWorkChain._B2W_NAMESPACE
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching `b2w` with PK {workchain_node.pk}')

        return ToContext(workchain_b2w=workchain_node)    

    def inspect_b2w(self):
        """Inspect the b2w workflow."""
        b2w_workchain = self.ctx.workchain_b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W
    
    def prepare_intp(self):
        """Prepare the inputs for the interpolation workflow."""
        
        if 'interpolation_distances' not in self.inputs:
            self.report('No interpolation distances provided, will not do interpolation.')
            return
        
        parameters = self.ctx.inputs_a2f[EpwA2fWorkChain._INTP_NAMESPACE].epw.parameters.get_dict()
        
        if self.should_run_b2w():
            b2w_workchain = self.ctx.workchain_b2w

            b2w_parameters = b2w_workchain.inputs[EpwB2WWorkChain._EPW_NAMESPACE].epw.parameters.get_dict()
            
            parent_folder_epw = b2w_workchain.outputs.epw.remote_folder

            for namespace, keyword in self._blocked_keywords:
                if keyword in b2w_parameters[namespace]:
                    parameters[namespace][keyword] = b2w_parameters[namespace][keyword]
            
            self.ctx.inputs_a2f[EpwA2fWorkChain._INTP_NAMESPACE].parent_folder_epw = parent_folder_epw
            self.ctx.inputs_a2f[EpwA2fWorkChain._INTP_NAMESPACE].epw.parameters = orm.Dict(parameters)
            
    def should_run_conv(self):
        """Check if the conv loop should continue or not."""
        if not self.ctx.do_conv:
            return False

        try:
            prev_allen_dynes = self.ctx.a2f_conv[-2].outputs.output_parameters['Allen_Dynes_Tc']
            new_allen_dynes = self.ctx.a2f_conv[-1].outputs.output_parameters['Allen_Dynes_Tc']
            self.ctx.is_converged = (
                abs(prev_allen_dynes - new_allen_dynes) / new_allen_dynes
                < self.inputs.convergence_threshold.value
            )
            self.report(f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {self.ctx.is_converged.value}')
            
        except (AttributeError, IndexError, KeyError):
            self.report('Not enough data to check convergence.')

        ## TODO: If Allen-Dynes Tc is converged, clean the remote folder:
        ## rm out/aiida.ephmat/*, out/aiida.epwmatwp, 
        ## Leaving only the final a2f calculation for later use.
        
        if self.ctx.is_converged:
            self.ctx.inputs_iso[EpwIsoWorkChain._INTP_NAMESPACE].parent_folder_epw = self.ctx.a2f_conv[-1].outputs.remote_folder
            return False
        
        else:
            if len(self.ctx.interpolation_distances) == 0 and not self.ctx.is_converged:
                if self.inputs.always_run_final.value:
                    self.ctx.inputs_iso[EpwIsoWorkChain._INTP_NAMESPACE].parent_folder_epw = self.ctx.a2f_conv[-1].outputs.remote_folder
                    self.report('Allen-Dynes Tc is not converged, but will run the following workchains!.')
                    return False
                else:   
                    return self.exit_codes.ERROR_CONVERGENCE_NOT_REACHED
            else:
                return True
    
    def run_conv(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""
        
        inputs = self.ctx.inputs_a2f
        
        inputs.a2f.qfpoints_distance = self.ctx.interpolation_distances.pop()

        inputs.metadata.call_link_label = self._CONV_NAMESPACE
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)

        self.report(f'launching interpolation `epw` with PK {workchain_node.pk} [qfpoints_distance = {inputs.a2f.qfpoints_distance}]')

        return ToContext(a2f_conv=append_(workchain_node))

    def inspect_conv(self):
        """Verify that the conv workflow finished successfully."""
        a2f_workchain = self.ctx.a2f_conv[-1]
        
        if not a2f_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f_workchain.exit_status}')
            self.ctx.a2f_conv.pop()
        else:
            ## TODO: Use better way to get the mesh
            a2f_calculation = a2f_workchain.called_descendants[-1]
            mesh = 'x'.join(str(i) for i in a2f_calculation.inputs.qfpoints.get_kpoints_mesh()[0])

            try:
                self.report(f"Allen-Dynes: {a2f_workchain.outputs.output_parameters['Allen_Dynes_Tc']} at {mesh}")
            except KeyError:
                self.report(f"Could not find Allen-Dynes temperature in parsed output parameters!")

    def should_run_a2f(self):
        """Check if the a2f workflow should continue or not."""
        return EpwA2fWorkChain._INTP_NAMESPACE in self.inputs
    
    def run_a2f(self):
        """Run the a2f workflow."""
        inputs = self.ctx.inputs_a2f
        
        inputs.metadata.call_link_label = EpwA2fWorkChain._INTP_NAMESPACE
        workchain_node = self.submit(EpwA2fWorkChain, **inputs)
        return ToContext(workchain_a2f=workchain_node)

    def inspect_a2f(self):
        """Inspect the a2f workflow."""
        a2f_workchain = self.ctx.workchain_a2f
        if not a2f_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {a2f_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_A2F
                
        self.ctx.inputs_iso[EpwIsoWorkChain._INTP_NAMESPACE].parent_folder_epw = a2f_workchain.outputs.remote_folder

    def should_run_iso(self):
        """Check if the iso workflow should continue or not."""
        
        Allen_Dynes_Tc = self.ctx.inputs_iso.inputs[EpwIsoWorkChain._INTP_NAMESPACE].parent_folder_epw.creator.outputs.output_parameters['Allen_Dynes_Tc']
        
        if Allen_Dynes_Tc < EpwIsoWorkChain._MIN_TEMP:
            return self.exit_codes.ERROR_ISOTROPIC_TC_TOO_LOW
        
        self.ctx.Allen_Dynes_Tc = Allen_Dynes_Tc
        
        return EpwIsoWorkChain._INTP_NAMESPACE in self.inputs
    
    def run_iso(self):
        """Run the iso workflow."""
        inputs = self.ctx.inputs_iso
        
        parent_folder_epw = inputs[EpwIsoWorkChain._INTP_NAMESPACE].parent_folder_epw
        a2f_parameters = parent_folder_epw.creator.inputs.parameters.get_dict()
        
        inputs.estimated_Tc_iso = orm.Float(self.ctx.Allen_Dynes_Tc * 1.5)
        
        parameters = inputs[EpwIsoWorkChain._INTP_NAMESPACE].epw.parameters.get_dict()
        
        for namespace, keyword in self._blocked_keywords:
            if keyword in a2f_parameters[namespace]:
                parameters[namespace][keyword] = a2f_parameters[namespace][keyword]
        
        for namespace, _params in self._restart_from_ephmat.items():
            for key, value in _params:
                parameters[namespace][key] = value
        
        inputs[EpwIsoWorkChain._INTP_NAMESPACE].epw.parameters = orm.Dict(parameters)
        
        inputs[EpwIsoWorkChain._INTP_NAMESPACE].qfpoints_distance = parent_folder_epw.creator.caller.inputs.qfpoints_distance
        inputs[EpwIsoWorkChain._INTP_NAMESPACE].kfpoints_factor = parent_folder_epw.creator.caller.inputs.kfpoints_factor
        
        inputs.metadata.call_link_label = EpwIsoWorkChain._INTP_NAMESPACE
        workchain_node = self.submit(EpwIsoWorkChain, **inputs)
        return ToContext(workchain_iso=workchain_node)

    def inspect_iso(self):
        """Inspect the iso workflow."""
        iso_workchain = self.ctx.workchain_iso
        if not iso_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {iso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ISO

        
        self.ctx.inputs_aniso[EpwAnisoWorkChain._INTP_NAMESPACE].parent_folder_epw = iso_workchain.outputs.remote_folder
        
    def should_run_aniso(self):
        """Check if the aniso workflow should continue or not."""
        
        Tc_iso = self.ctx.inputs_iso.inputs[EpwIsoWorkChain._INTP_NAMESPACE].parent_folder_epw.creator.caller.caller.outputs.Tc_iso.value
        
        if Tc_iso < EpwAnisoWorkChain._MIN_TEMP:
            return self.exit_codes.ERROR_ANISOTROPIC_TC_TOO_LOW
        
        self.ctx.Tc_iso = Tc_iso
        
        return EpwAnisoWorkChain._INTP_NAMESPACE in self.inputs
    
    def run_aniso(self):
        """Run the aniso workflow."""
        inputs = self.ctx.inputs_aniso

        inputs.estimated_Tc_aniso = orm.Float(self.ctx.Tc_iso * 2.0)
        
        parent_folder_epw = inputs[EpwAnisoWorkChain._INTP_NAMESPACE].parent_folder_epw
        iso_parameters = parent_folder_epw.creator.inputs.parameters.get_dict()
        parameters = inputs.aniso.epw.parameters.get_dict()
        
        for namespace, keyword in self._blocked_keywords:
            if keyword in iso_parameters[namespace]:
                parameters[namespace][keyword] = iso_parameters[namespace][keyword] 
        
        for namespace, _params in self._restart_from_ephmat.items():
            for key, value in _params:
                parameters[namespace][key] = value
        
        inputs[EpwAnisoWorkChain._INTP_NAMESPACE].epw.parameters = orm.Dict(parameters)
        
        inputs[EpwAnisoWorkChain._INTP_NAMESPACE].qfpoints_distance = parent_folder_epw.creator.caller.inputs.qfpoints_distance
        inputs[EpwAnisoWorkChain._INTP_NAMESPACE].kfpoints_factor = parent_folder_epw.creator.caller.inputs.kfpoints_factor
        
        inputs.metadata.call_link_label = EpwAnisoWorkChain._INTP_NAMESPACE
        workchain_node = self.submit(EpwAnisoWorkChain, **inputs)
        return ToContext(workchain_aniso=workchain_node)

    def inspect_aniso(self):
        """Inspect the aniso workflow."""
        aniso_workchain = self.ctx.workchain_aniso
        if not aniso_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {aniso_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ANISO
        
    def results(self):
        """TODO"""
        
        if 'workchain_b2w' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_b2w, 
                    EpwB2WWorkChain,
                    namespace=EpwA2fWorkChain._B2W_NAMESPACE
                )
            )
        
        # Either expose the last convergence
        
        if 'a2f_conv' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.a2f_conv[-1], 
                    EpwA2fWorkChain,
                    namespace=self._CONV_NAMESPACE
                )
            )
        
        # Or expose the A2F workchain since they are exclusive
        if 'workchain_a2f' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_a2f, 
                    EpwA2fWorkChain,
                    namespace=EpwA2fWorkChain._INTP_NAMESPACE
                )
            )
            
        if 'workchain_iso' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_iso, 
                    EpwIsoWorkChain,
                    namespace=EpwIsoWorkChain._INTP_NAMESPACE
                )
            )
            
        if 'workchain_aniso' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_aniso, 
                    EpwAnisoWorkChain,
                    namespace=EpwAnisoWorkChain._INTP_NAMESPACE
                )
            )

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

