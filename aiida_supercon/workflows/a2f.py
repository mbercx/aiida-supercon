# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm
from aiida.engine import calcfunction, ToContext, if_


from .base import EpwBaseWorkChain
from .b2w import EpwB2WWorkChain
from .intp import EpwBaseIntpWorkChain

from scipy.interpolate import interp1d
import numpy
from ..common.restart import RestartType
import warnings
from ..common.restart import RestartState
@calcfunction
def calculate_tc(max_eigenvalue: orm.XyData) -> orm.Float:
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

class EpwA2fWorkChain(EpwBaseIntpWorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""
    
    _INTP_NAMESPACE = 'a2f'
    _ALL_NAMESPACES = [EpwBaseIntpWorkChain._B2W_NAMESPACE, _INTP_NAMESPACE]
    _RESTART_STATE = RestartState(_ALL_NAMESPACES)
    
    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    @classmethod
    def validate_inputs(cls, inputs, ctx=None):
        """Validate the inputs."""
        return None
        
        
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

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
        
        spec.inputs[cls._INTP_NAMESPACE].validator = cls.validate_inputs
        spec.inputs.validator = cls.validate_inputs
        
        # spec.output('a2f', valid_type=orm.XyData,
        #             help='The contents of the `.a2f` file.')
        # spec.output('Tc_allen_dynes', valid_type=orm.Float,
        #             help='The Allen-Dynes Tc interpolated from the a2f file.')

        spec.exit_code(
            402, 'ERROR_SUB_PROCESS_A2F',
            message='The `epw.x` workflow failed.'
            )
    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'{cls._INTP_NAMESPACE}.yaml'

    @classmethod
    def get_builder_restart(
        cls, 
        from_a2f_workchain
        ):
        
        return super()._get_builder_restart(
            from_intp_workchain=from_a2f_workchain,
            )
        
    # @classmethod
    # def get_builder_restart(
    #     cls, 
    #     from_a2f_workchain=None,
    #     **kwargs
    #     ):
    #     """Return a builder prepopulated with inputs selected according to the chosen protocol."""
    #     builder = cls.get_builder()
    #     parent_builder = from_a2f_workchain.get_builder_restart()
        
    #     b2w = from_a2f_workchain.base.links.get_outgoing(
    #         link_label_filter=cls._B2W_NAMESPACE
    #         ).first()
        
    #     if cls._B2W_NAMESPACE not in from_a2f_workchain.inputs or b2w.node.is_finished_ok:
    #         builder.pop(cls._B2W_NAMESPACE)
    #     else:
    #         b2w_builder = EpwB2WWorkChain.get_builder_restart(
    #             from_b2w_workchain=b2w.node
    #             )
    #         builder[cls._B2W_NAMESPACE]._data = b2w_builder._data
        
    #     a2f = from_a2f_workchain.base.links.get_outgoing(
    #         link_label_filter=cls._INTP_NAMESPACE
    #         ).first()

        
    #     if a2f and a2f.node.is_finished_ok:
    #         warnings.warn(
    #             f"The A2F workchain <{from_a2f_workchain.pk}> is already finished.",
    #             stacklevel=2
    #             )
    #         return
    #     else:
    #         builder[cls._INTP_NAMESPACE]._data = parent_builder[cls._INTP_NAMESPACE]._data
        
    #     return builder
    
    # @classmethod
    # def get_builder_from_b2w(
    #     cls,
    #     from_b2w_workchain: orm.WorkChainNode,
    #     protocol=None,
    #     overrides=None,
    #     **kwargs
    #     ):
    #     """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        
    #     inputs = cls.get_protocol_inputs(protocol, overrides)
        
    #     builder = cls.get_builder()


    #     if not from_b2w_workchain or not from_b2w_workchain.process_class == EpwB2WWorkChain:
    #         raise ValueError('Currently we only accept `EpwB2WWorkChain`')
        
    #     structure = from_b2w_workchain.inputs.structure
    #     code = from_b2w_workchain.inputs[EpwB2WWorkChain._EPW_NAMESPACE]['epw'].code
        
    #     builder.structure = structure
        
    #     if from_b2w_workchain.is_finished_ok:
    #         builder.pop(cls._B2W_NAMESPACE)
    #     else:
    #         b2w_builder = EpwB2WWorkChain.get_builder_restart(
    #             from_b2w_workchain=from_b2w_workchain,
    #             protocol=protocol,
    #             overrides=overrides.get(cls._B2W_NAMESPACE, None),
    #             **kwargs
    #             )
            
    #         # Actually there is no exclusion of EpwB2WWorkChain namespace
    #         # So we need to set the _data manually
    #         builder[cls._B2W_NAMESPACE]._data = b2w_builder._data
            
    #     intp_builder = EpwBaseWorkChain.get_builder_from_protocol(
    #         code=code,
    #         structure=structure,
    #         protocol=protocol,
    #         overrides=inputs.get(cls._INTP_NAMESPACE, None),
    #         **kwargs
    #         )
        
    #     intp_builder.parent_folder_epw = from_b2w_workchain.outputs.epw.remote_folder
        
    #     builder[cls._INTP_NAMESPACE]._data = intp_builder._data
        
    #     builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        
    #     return builder
        
        
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
            return self.exit_codes.ERROR_SUB_PROCESS_A2F

    def results(self):
        """TODO"""

        super().results()
        
        # self.out('a2f', self.ctx.workchain_intp.outputs.a2f)
