# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, process_handler, calcfunction

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from .intp import EpwBaseIntpWorkChain


from importlib.resources import files

class EpwAnisoWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the anisotropic critical temperature."""
    
    _INTP_NAMESPACE = 'aniso'
    
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
    
    _DEFAULT_FILIROBJ = "ir_nlambda6_ndigit8.dat"
    _frozen_plot_gap_function_parameters = {
        'INPUTEPW': {
            'iverbosity': 2,
        }
    }
    
    _frozen_ir_parameters = {
        'INPUTEPW': {
            'fbw': True,
            'muchem': True,
            'gridsamp': 2,
            'broyden_beta': -0.7,
            # 'filirobj': './' + _DEFAULT_FILIROBJ,
        }
    }
    
    _min_temp = 3.5
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('plot_gap_function', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='Whether to plot the gap function.')
        spec.input('estimated_Tc_aniso', valid_type=orm.Float, default=lambda: orm.Float(40.0),
            help='The estimated Tc for the aniso calculation.')
        spec.input('use_ir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='Whether to use the intermediate representation.')
        # spec.input(
        #     'filirobj', 
        #     valid_type=orm.SinglefileData, 
        #     help='The file containing the intermediate representation.',
        #     required=False,
        # )
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
        # spec.output('a2f', valid_type=orm.XyData,
        #     help='The contents of the `.a2f` file.')
        # spec.output('Tc_aniso', valid_type=orm.Float,
        #   help='The anisotropic Tc interpolated from the a2f file.')

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

        if not ('parent_epw_folder' in port_namespace or 'epw' in port_namespace):
            return "Only one of `parent_epw_folder` or `epw` can be accepted."
        
        return None

    @classmethod
    def get_builder_restart(
        cls,
        from_aniso_workchain
        ):
        
        return super()._get_builder_restart(
            from_intp_workchain=from_aniso_workchain,
            )
        
    @classmethod
    def get_builder_from_protocol(
            cls, 
            codes, 
            structure, 
            protocol=None, 
            overrides=None, 
            **kwargs
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        builder = super().get_builder_from_protocol(
            codes, 
            structure, 
            protocol, 
            overrides,
            **kwargs
        )
        
        return builder

    def prepare_process(self):
        """Prepare the process for the current interpolation distance."""
        
        super().prepare_process()
                
        parameters = self.ctx.inputs.epw.parameters.get_dict()
        
        temps = f'{self._MIN_TEMP} {self.inputs.estimated_Tc_aniso}'
        parameters['INPUTEPW']['temps'] = temps
        
        try:
            settings = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = [
            'out/aiida.dos', 'aiida.a2f*', 'aiida.phdos*', 
            'aiida.pade_aniso_gap0_*', 'aiida.imag_aniso_gap0*',
            'aiida.lambda_k_pairs', 'aiida.lambda_FS'
            ]
                
        if self.inputs.plot_gap_function.value:
            for namespace, _parameters in self._frozen_plot_gap_function_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
            settings['ADDITIONAL_RETRIEVE_LIST'].extend([
                'aiida.imag_aniso_gap0_*.frmsf',
                'aiida.lambda.frmsf',
                ])
            
        from importlib.resources import files
        if self.inputs.use_ir.value:
            for namespace, _parameters in self._frozen_ir_parameters.items():
                for keyword, value in _parameters.items():
                    parameters[namespace][keyword] = value
            
            filirobj = self.ctx.inputs.epw.code.filepath_executable.parent.parent / 'EPW' / 'irobjs' / self._DEFAULT_FILIROBJ

            parameters['INPUTEPW']['filirobj'] = str(filirobj)
            
        self.ctx.inputs.epw.settings = orm.Dict(settings)
        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ANISO
        
        if False:
            return self.handle_temperature_out_of_range(aniso)

    def results(self):
        """TODO"""
        
        super().results()
        
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
