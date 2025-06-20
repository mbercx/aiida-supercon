# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, process_handler, calcfunction

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from .intp import EpwBaseIntpWorkChain

from ..common.restart import RestartType

from scipy.interpolate import interp1d
import numpy


@calcfunction
def calculate_iso_tc(max_eigenvalue: orm.XyData) -> orm.Float:
    me_array = max_eigenvalue.get_array('max_eigenvalue')
    if me_array[:, 1].max() < 1.0:
        return orm.Float(0.0)
    else:
        return orm.Float(float(interp1d(me_array[:, 1], me_array[:, 0])(1.0)))

@calcfunction
def calculate_Allen_Dynes_tc(a2f: orm.ArrayData, mustar = 0.13) -> orm.Float:
    w        = a2f.get_array('frequency')
    # Here we preassume that there are 10 smearing values for a2f calculation
    spectral = a2f.get_array('a2f')[:, 9]   
    mev2K    = 11.604525006157

    _lambda  = 2*numpy.trapz(numpy.divide(spectral, w), x=w)

    # wlog =  np.exp(np.average(np.divide(alpha, w), weights=np.log(w)))
    wlog     =  numpy.exp(2/_lambda*numpy.trapz(numpy.multiply(numpy.divide(spectral, w), numpy.log(w)), x=w))

    Tc = wlog/1.2*numpy.exp(-1.04*(1+_lambda)/(_lambda-mustar*(1+0.62*_lambda))) * mev2K


    return orm.Float(Tc)

class EpwAnisoWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the anisotropic critical temperature."""
    
    _INTP_NAMESPACE = 'aniso'
    _RESTART_INTP = RestartType.RESTART_ANISO
    _frozen_restart_parameters = {
        'INPUTEPW': {
            'elph': False,
            'ep_coupling': False,
            'epwread': True,
            'epwwrite': False,
            'ephwrite': False,
            'restart': True,
        },
    }
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    
    _frozen_plot_gap_function_parameters = {
        'INPUTEPW': {
            'iverbosity': 2,
        }
    }
    _min_temp = 3.5
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('plot_gap_function', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='Whether to plot the gap function.')
        spec.input('use_ir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='Whether to use the intermediate representation.')

        spec.outline(
            cls.setup,
            if_(cls.should_run_b2w)(
                cls.run_b2w,
                cls.inspect_b2w,
            ),
            cls.prepare_process,
            cls.run_process,
            cls.inspect_process,
            cls.results
        )
        spec.output('a2f', valid_type=orm.XyData,
            help='The contents of the `.a2f` file.')
        # spec.output('Tc_aniso', valid_type=orm.Float,
        #   help='The anisotropic Tc interpolated from the a2f file.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_EPW',
            message='The `epw` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_ANISO',
            message='The `aniso` sub process failed')
        spec.exit_code(403, 'ERROR_TEMPERATURE_OUT_OF_RANGE',
            message='The `aniso` calculation have less than two temperatures within aniso Tc ')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'
    
    @classmethod
    def validate_inputs(cls, value, port_namespace):  # pylint: disable=unused-argument
        """Validate the top level namespace."""

        # if not ('qfpoints_distance' in port_namespace or 'qfpoints' in port_namespace):
        #     return "Neither `qfpoints` nor `qfpoints_distance` were specified."

        if not ('parent_epw_folder' in port_namespace or 'epw' in port_namespace):
            return "Only one of `parent_epw_folder` or `epw` can be accepted."

    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            protocol=None, 
            from_workchain=None,
            overrides=None, 
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        builder = super().get_builder_from_protocol(
            codes, 
            structure, 
            protocol, 
            from_workchain,
            overrides,
            restart_intp=cls._RESTART_INTP,
            **kwargs
        )
        
        return builder

    def prepare_process(self):
        """Prepare the process for the current interpolation distance."""
        
        super().prepare_process()
                
        parameters = self.ctx.inputs.epw.parameters.get_dict()
        
        try:
            settings = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = [
            'out/aiida.dos', 'aiida.a2f*', 'aiida.phdos*', 
            'aiida.pade_aniso_gap0_*', 'aiida.imag_aniso_gap0*',
            'aiida.lambda_k_pairs']
                
        if self.inputs.plot_gap_function.value:
            for namespace, _parameters in self._frozen_plot_gap_function_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
            settings['ADDITIONAL_RETRIEVE_LIST'].extend([
                'aiida.imag_aniso_gap0_*.frmsf',
                'aiida.lambda.frmsf',
                ])
            
        self.ctx.inputs.epw.settings = orm.Dict(settings)
        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp = self.ctx.intp

        if not intp.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ANISO
        
        if False:
            return self.handle_temperature_out_of_range(aniso)


    def results(self):
        """TODO"""
        
        super().results()
        
        self.out('a2f', self.ctx.intp.outputs.a2f)

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

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report(f'Action taken: {action}')
        
    @process_handler(priority=403,)
    def handle_temperature_out_of_range(self, calculation):
        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
        if calculation.exit_status == self.exit_codes.ERROR_TEMPERATURE_OUT_OF_RANGE:
            self.report_error_handled(calculation, 'unrecoverable error, aborting...')
            return ProcessHandlerReport(True, self.exit_codes.ERROR_TEMPERATURE_OUT_OF_RANGE)
