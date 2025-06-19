# -*- coding: utf-8 -*-
"""Work chain for computing the critical temperature based off an `EpwWorkChain`."""
from aiida import orm, load_profile
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_, ExitCode

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida.engine import calcfunction

from ..common.restart import RestartType

from .base import EpwBaseWorkChain
from .b2w import EpwB2WWorkChain
import warnings
class EpwBaseIntpWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the Allen-Dynes critical temperature."""
    
    # --- Child classes should override these placeholders ---
    _B2W_NAMESPACE = 'b2w'  # e.g., 'intp' for A2fWorkChain
    _INTP_NAMESPACE = 'intp'  # e.g., 'a2f' for A2fWorkChain
    # ---------------------------------------------------------

    _blocked_keywords = [
        ('INPUTEPW', 'use_ws'),
        ('INPUTEPW', 'nbndsub'),
        ('INPUTEPW', 'bands_skipped'),
        ('INPUTEPW', 'vme'),
    ]
    
    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        # TODO: Seems we don't need structure input port here
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        
        spec.expose_inputs(
            EpwB2WWorkChain, namespace=cls._B2W_NAMESPACE, exclude=(
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the interpolation `EpwCalculation`s.'
            }
        )
        
        # spec.inputs[cls._B2W_NAMESPACE].validator = None
        spec.expose_inputs(
            EpwBaseWorkChain, namespace=cls._INTP_NAMESPACE, exclude=(
                'clean_workdir',
                'parent_folder_nscf',
                'parent_folder_ph',
                'parent_folder_chk',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the a2f `EpwBaseWorkChain`.'
            }
        )
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
        spec.expose_outputs(
            EpwB2WWorkChain, 
            namespace=cls._B2W_NAMESPACE, 
            namespace_options={
                'required': False,
                'help': 'Outputs for the `EpwB2WWorkChain`.'
            }
        )
        spec.expose_outputs(
            EpwBaseWorkChain, 
            namespace=cls._INTP_NAMESPACE,
            exclude=(
                'remote_folder',
                'output_parameters',
            )
        )
        spec.output('output_parameters', valid_type=orm.Dict,
                    help='The `output_parameters` output node of the final EPW calculation.')
        spec.output('remote_folder', valid_type=orm.RemoteData,
                    help='The remote folder of the final EPW calculation.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_B2W',
            message='The `B2W` sub process failed')

    @staticmethod
    def get_descendant(
        intp: orm.WorkChainNode,
        link_label_filter: str
        ) -> orm.WorkChainNode:
        """Get the descendant workchains of the intp workchain."""
        try:
            return intp.base.links.get_outgoing(
                link_label_filter=link_label_filter
                ).first().node
        except AttributeError:
            return None
        

    @classmethod
    def _get_builder_restart(
        cls, 
        from_intp_workchain=None,
        ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        builder = from_intp_workchain.get_builder_restart()
        # parent_builder = from_intp_workchain.get_builder_restart()
        
        b2w = EpwBaseIntpWorkChain.get_descendant(
            from_intp_workchain,
            cls._B2W_NAMESPACE
            )
        
        if (
            cls._B2W_NAMESPACE not in from_intp_workchain.inputs 
            or 
            b2w.is_finished_ok
            ):
            
            builder.pop(cls._B2W_NAMESPACE)
        else:
            b2w_builder = EpwB2WWorkChain.get_builder_restart(
                from_b2w_workchain=b2w
                )
            builder[cls._B2W_NAMESPACE]._data = b2w_builder._data
        
        intp = EpwBaseIntpWorkChain.get_descendant(
            from_intp_workchain,
            cls._INTP_NAMESPACE
            )
        
        if intp and intp.is_finished_ok:
            warnings.warn(
                f"The Workchain <{from_intp_workchain.pk}> is already finished.",
                stacklevel=2
                )
            return
        else:
            builder[cls._INTP_NAMESPACE].parent_folder_epw = intp.inputs.parent_folder_epw
        
            return builder
    
    @classmethod
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
        
        structure = from_b2w_workchain.inputs.structure
        code = from_b2w_workchain.inputs[EpwB2WWorkChain._EPW_NAMESPACE]['epw'].code
        
        builder.structure = structure
        
        if from_b2w_workchain.is_finished_ok:
            builder.pop(cls._B2W_NAMESPACE)
        else:
            b2w_builder = EpwB2WWorkChain.get_builder_restart(
                from_b2w_workchain=from_b2w_workchain,
                protocol=protocol,
                overrides=overrides.get(cls._B2W_NAMESPACE, None),
                **kwargs
                )
            
            # Actually there is no exclusion of EpwB2WWorkChain namespace
            # So we need to set the _data manually
            builder[cls._B2W_NAMESPACE]._data = b2w_builder._data
            
        intp_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._INTP_NAMESPACE, None),
            **kwargs
            )
        
        intp_builder.parent_folder_epw = from_b2w_workchain.outputs.epw.remote_folder
        
        builder[cls._INTP_NAMESPACE]._data = intp_builder._data
        
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        
        return builder
        
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
        inputs = cls.get_protocol_inputs(protocol, overrides)
        builder = cls.get_builder()
        builder.structure = structure

        b2w_builder = EpwB2WWorkChain.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._B2W_NAMESPACE, {}),
            **kwargs
        )
        
        b2w_builder.w90_intp.pop('open_grid')
        b2w_builder.w90_intp.pop('projwfc')
        
        builder[cls._B2W_NAMESPACE]._data = b2w_builder._data
        
        intp_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._INTP_NAMESPACE, None),
        )
        
        builder[cls._INTP_NAMESPACE]._data = intp_builder._data

        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder
    
    def setup(self):
        """Setup steps, i.e. initialise context variables."""
        self.ctx.degaussq = None
        inputs = self.exposed_inputs(EpwBaseWorkChain, namespace=self._INTP_NAMESPACE)
                
        self.ctx.inputs = inputs
        
    def should_run_b2w(self):
        """Check if the epw loop should continue or not."""
        return self._B2W_NAMESPACE in self.inputs
    
    def run_b2w(self):
        """Run the ``restart`` EPW calculation for the current interpolation distance."""

        self.report(f'Running B2W...')
        inputs = self.exposed_inputs(EpwB2WWorkChain, namespace=self._B2W_NAMESPACE)
        
        
        inputs.metadata.call_link_label = self._B2W_NAMESPACE
        workchain_node = self.submit(EpwB2WWorkChain, **inputs)

        self.report(f'launching `{self._B2W_NAMESPACE}` with PK {workchain_node.pk}')

        return ToContext(workchain_b2w=workchain_node)

    def inspect_b2w(self):
        """Verify that the epw.x workflow finished successfully."""
        b2w_workchain = self.ctx.workchain_b2w

        if not b2w_workchain.is_finished_ok:
            self.report(f'`epw.x` failed with exit status {b2w_workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_B2W

        # We set the parent folder here to keep the logic of restart from EpwB2WWorkChain
        # Since the only connection between the EpwBaseIntpWorkChain and EpwB2WWorkChain is the parent_folder_epw
        # And the parameter of EpwB2WWorkChain is deduced from the parent_folder_epw
        self.ctx.inputs.parent_folder_epw = b2w_workchain.outputs.epw.remote_folder

        
    def prepare_process(self):
        """Prepare the process for the current interpolation distance."""
        

        parameters = self.ctx.inputs.epw.parameters.get_dict()
        
        b2w_parameters = self.ctx.inputs.parent_folder_epw.creator.inputs.parameters.get_dict()
                        
        for namespace, keyword in self._blocked_keywords:
            if keyword in b2w_parameters[namespace]:
                parameters[namespace][keyword] = b2w_parameters[namespace][keyword]
        
        self.ctx.inputs.epw.parameters = orm.Dict(parameters)
        
        
    def run_process(self):
        """Prepare the process for the current interpolation distance."""
        inputs = self.ctx.inputs
        
        inputs.metadata.call_link_label = self._INTP_NAMESPACE
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)

        self.report(f'launching `{self._INTP_NAMESPACE}` with PK {workchain_node.pk}')

        return ToContext(workchain_intp=workchain_node)

    def results(self):
        """TODO"""
        
        if 'workchain_b2w' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_b2w, 
                    EpwB2WWorkChain, 
                    namespace=self._B2W_NAMESPACE
                    )
            )
        
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_intp, 
                EpwBaseWorkChain, 
                namespace=self._INTP_NAMESPACE
                )
        )
        
        self.out('output_parameters', self.ctx.workchain_intp.outputs.output_parameters)
        self.out('remote_folder', self.ctx.workchain_intp.outputs.remote_folder)
        # self.out('retrieved', self.ctx.intp.outputs.retrieved)

    def on_terminated(self):
        """Clean up the work chain."""
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
