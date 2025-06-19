# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida.engine import calcfunction

from scipy.interpolate import interp1d
import numpy
import warnings

from .intp import EpwBaseIntpWorkChain
from .b2w import EpwB2WWorkChain

from .utils.calculators import calculate_iso_tc


class EpwIsoWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""
    
    _INTP_NAMESPACE = 'iso'
    
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    _min_temp = 1.0
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('estimated_Tc_iso', valid_type=orm.Float, default=lambda: orm.Float(40.0),
            help='The estimated Tc for the iso calculation.')
        spec.input('linearized_Eliashberg', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='Whether to use the linearized Eliashberg function.')
        
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
        # spec.output('parameters', valid_type=orm.Dict,
        #             help='The `output_parameters` output node of the final EPW calculation.')
        # spec.output('a2f', valid_type=orm.XyData,
        #             help='The contents of the `.a2f` file.')
        # spec.output('max_eigenvalue', valid_type=orm.XyData,
        #     help='The temperature dependence of the max eigenvalue for the final EPW.')

        spec.output('Tc_iso', valid_type=orm.Float,
                    help='The isotropic Tc interpolated from the a2f file.')

        spec.exit_code(402, 'ERROR_SUB_PROCESS_ISO',
            message='The `iso` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'
    
    @classmethod
    def get_builder_restart(
        cls, from_iso_workchain
        ):

        return super()._get_builder_restart(
            from_intp_workchain=from_iso_workchain,
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
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
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
        temps = f'{self._MIN_TEMP} {self.inputs.estimated_Tc_iso}'
        parameters['INPUTEPW']['temps'] = temps
        
        self.ctx.inputs.epw.parameters = orm.Dict(parameters)

        try:
            settings = self.ctx.inputs.epw.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['ADDITIONAL_RETRIEVE_LIST'] = [
            'aiida.a2f',
            'aiida.a2f_proj',
            'out/aiida.dos',
            'aiida.phdos',
            'aiida.phdos_proj',
            'aiida.lambda_FS',
            'aiida.lambda_k_pairs'
            ]
        
        self.ctx.inputs.epw.settings = orm.Dict(settings)
                
    def inspect_process(self):
        """Verify that the epw.x workflow finished successfully."""
        intp_workchain = self.ctx.workchain_intp

        if not intp_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {intp_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_ISO

        inputs = {
                'max_eigenvalue':intp_workchain.outputs.max_eigenvalue,
                'metadata': {
                    'call_link_label': 'calculate_iso_tc'
                }
            }
        Tc_iso = calculate_iso_tc(**inputs)
        self.ctx.Tc_iso = Tc_iso

    def results(self):
        """TODO"""
        
        super().results()
        
        # self.out('a2f', self.ctx.intp.outputs.a2f)
        self.out('max_eigenvalue', self.ctx.intp.outputs.max_eigenvalue)
        self.out('Tc_iso', self.ctx.Tc_iso)
