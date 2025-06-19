# -*- coding: utf-8 -*-
"""Work chain to compute the electron-phonon coupling with brute force QE."""

from aiida import orm
from aiida.common import AttributeDict

from aiida.engine import WorkChain, ToContext
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.q2r.base import Q2rBaseWorkChain
from aiida_quantumespresso.workflows.matdyn.base import MatdynBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

class ElectronPhononWorkChain(ProtocolMixin, WorkChain):
    """Work chain to compute the electron-phonon coupling."""

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('qpoints_distance', valid_type=orm.Float, default=lambda: orm.Float(0.5))
        spec.input('kpoints_factor', valid_type=orm.Int, default=lambda: orm.Int(2))
        spec.input('interpolation_factor', valid_type=orm.Int, default=lambda: orm.Int(2))

        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf', exclude=(
                'clean_workdir', 'pw.structure', 'kpoints', 'kpoints_distance'
            ),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` that does the initial `scf` calculation.'
            }
        )
        spec.expose_inputs(
            PhBaseWorkChain, namespace='ph_base', exclude=(
                'clean_workdir', 'ph.parent_folder', 'ph.qpoints'
            ),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` that does the `ph.x` calculation.'
            }
        )
        spec.expose_inputs(
            Q2rBaseWorkChain, namespace='q2r_base', exclude=('clean_workdir', 'q2r.parent_folder'),
            namespace_options={
                'help': 'Inputs for the `Q2rBaseWorkChain` that does the `q2r.x` calculation.'
            }
        )
        spec.expose_inputs(
            MatdynBaseWorkChain, namespace='matdyn_base', exclude=(
                'clean_workdir', 'matdyn.force_constants', 'matdyn.remote_folder'
            ),
            namespace_options={
                'help': 'Inputs for the `MatDynBaseWorkChain` that does the `matdyn.x` calculation.'
            }
        )
        spec.output('retrieved', valid_type=orm.FolderData)

        spec.outline(
            cls.generate_reciprocal_points,
            cls.run_scf_int,
            cls.inspecf_scf_int,
            cls.run_scf,
            cls.inspecf_scf,
            cls.run_ph,
            cls.inspect_ph,
            cls.run_q2r,
            cls.inspect_q2r,
            cls.run_matdyn,
            cls.inspect_matdyn,
            cls.results,
        )
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_SCF_FIT',
            message='The dense scf `PwBaseWorkChain` sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The scf `PwBaseWorkChain` sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_PHONON',
            message='The electron-phonon `PhBaseWorkChain` sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_Q2R',
            message='The `Q2rBaseWorkChain` sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_FAILED_MATDYN',
            message='The `MatdynBaseWorkChain` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'electronphonon.yaml'

    @classmethod
    def get_builder_from_protocol(cls, pw_code, ph_code, q2r_code, matdyn_code, structure, protocol=None, overrides=None, **kwargs):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param ph_code: the ``Code`` instance configured for the ``quantumespresso.ph`` plugin.
        :param q2r_code: the ``Code`` instance configured for the ``quantumespresso.q2r`` plugin.
        :param matdyn_code: the ``Code`` instance configured for the ``quantumespresso.matdyn`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (pw_code, structure, protocol)
        scf = PwBaseWorkChain.get_builder_from_protocol(*args, overrides=inputs.get('scf', None), **kwargs)
        scf['pw'].pop('structure', None)
        scf.pop('clean_workdir', None)
        scf.pop('kpoints_distance', None)

        args = (ph_code, None, protocol)
        ph_base = PhBaseWorkChain.get_builder_from_protocol(*args, overrides=inputs.get('ph_base', None), **kwargs)
        ph_base.pop('clean_workdir', None)
        ph_base.ph.pop('qpoints')

        q2r_base = Q2rBaseWorkChain.get_builder()
        q2r_inputs = inputs.get('q2r_base', None)
        q2r_base.q2r['parameters'] = orm.Dict(q2r_inputs['q2r']['parameters'])
        q2r_base.q2r['metadata'] = q2r_inputs['q2r']['metadata']
        q2r_base.q2r['code'] = q2r_code

        matdyn_base = MatdynBaseWorkChain.get_builder()
        matdyn_inputs = inputs.get('matdyn_base', None)
        matdyn_kpoints = orm.KpointsData()
        matdyn_kpoints.set_kpoints_mesh(matdyn_inputs['matdyn']['kpoints'])
        matdyn_base.matdyn['kpoints'] = matdyn_kpoints
        matdyn_base.matdyn['parameters'] = orm.Dict(matdyn_inputs['matdyn']['parameters'])
        matdyn_base.matdyn['metadata'] = matdyn_inputs['matdyn']['metadata']
        matdyn_base.matdyn['code'] = matdyn_code

        builder = cls.get_builder()
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        builder.kpoints_factor = orm.Int(inputs['kpoints_factor'])
        builder.interpolation_factor = orm.Int(inputs['interpolation_factor'])
        builder.structure = structure
        builder.scf = scf
        builder.ph_base = ph_base
        builder.q2r_base = q2r_base
        builder.matdyn_base = matdyn_base
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def generate_reciprocal_points(self):
        """Generate the qpoints and kpoints meshes for the `ph.x` and `pw.x` calculations."""

        inputs = {
            'structure': self.inputs.structure,
            'distance': self.inputs.qpoints_distance,
            'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
            'metadata': {
                'call_link_label': 'create_qpoints_from_distance'
            }
        }
        qpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        qpoints_mesh = qpoints.get_kpoints_mesh()[0]
        kpoints = orm.KpointsData()
        kpoints_mesh = [v * self.inputs.kpoints_factor.value for v in qpoints_mesh]
        kpoints.set_kpoints_mesh(kpoints_mesh)

        interpolation_kpoints = orm.KpointsData()
        interpolation_kpoints.set_kpoints_mesh([v * self.inputs.interpolation_factor.value for v in kpoints_mesh])
        self.ctx.qpoints = qpoints
        self.ctx.kpoints = kpoints
        self.ctx.interpolation_kpoints = interpolation_kpoints

    def run_scf_int(self):
        """Run the dense grid `scf` calculation with `pw.x`, used to generate the data for the interpolation."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.metadata.call_link_label = 'scf_int'
        inputs.pw.structure = self.inputs.structure
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('SYSTEM', {})['la2F'] = True
        inputs.kpoints = self.ctx.interpolation_kpoints

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

        workchain_node = self.submit(PwBaseWorkChain, **inputs)

        return ToContext(workchain_scf_fit=workchain_node)

    def inspecf_scf_int(self):
        """Verify that the `scf` calculation for the interpolation finished successfully."""
        workchain = self.ctx.workchain_scf_fit

        if not workchain.is_finished_ok:
            self.report(f'`scf_fit` PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF_FIT

        self.ctx.current_folder = workchain.outputs.remote_folder

    def run_scf(self):
        """Run the `scf` calculation with `pw.x` to prepare for the `ph.x`."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.metadata.call_link_label = 'scf'
        inputs.pw.structure = self.inputs.structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.kpoints = self.ctx.kpoints

        workchain_node = self.submit(PwBaseWorkChain, **inputs)

        return ToContext(workchain_scf=workchain_node)

    def inspecf_scf(self):
        """Verify that the initial SCF calculation finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(f'scf PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder

    def run_ph(self):
        """Run the `PhBaseWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(PhBaseWorkChain, namespace='ph_base'))
        inputs.metadata.call_link_label = 'ph_base'
        inputs.ph.parent_folder = self.ctx.current_folder
        inputs.ph.qpoints = self.ctx.qpoints

        workchain_node = self.submit(PhBaseWorkChain, **inputs)

        return ToContext(workchain_ph=workchain_node)

    def inspect_ph(self):
        """Verify that the `PhBaseWorkChain` finished successfully."""
        workchain = self.ctx.workchain_ph

        if not workchain.is_finished_ok:
            self.report(f'Electron-phonon PhBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONON

        self.ctx.current_folder = workchain.outputs.remote_folder

    def run_q2r(self):
        """Run the ``Q2rBaseWorkChain``."""
        inputs = AttributeDict(self.exposed_inputs(Q2rBaseWorkChain, namespace='q2r_base'))
        inputs.metadata.call_link_label = 'q2r_base'
        inputs.q2r.parent_folder = self.ctx.current_folder

        workchain_node = self.submit(Q2rBaseWorkChain, **inputs)

        return ToContext(workchain_q2r=workchain_node)

    def inspect_q2r(self):
        """Verify that the ``Q2rBaseWorkChain`` finished successfully."""
        workchain = self.ctx.workchain_q2r

        if not workchain.is_finished_ok:
            self.report(f'The `Q2rBaseWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_Q2R

        self.ctx.force_constants = workchain.outputs.force_constants
        self.ctx.current_folder = workchain.outputs.remote_folder

    def run_matdyn(self):
        """Run the ``MatdynBaseWorkChain``."""
        inputs = AttributeDict(self.exposed_inputs(MatdynBaseWorkChain, namespace='matdyn_base'))
        inputs.metadata.call_link_label = 'matdyn_base'
        inputs.matdyn.force_constants = self.ctx.force_constants
        inputs.matdyn.parent_folder = self.ctx.current_folder

        workchain_node = self.submit(MatdynBaseWorkChain, **inputs)

        return ToContext(workchain_matdyn=workchain_node)

    def inspect_matdyn(self):
        """Verify that the ``MatdynBaseWorkChain`` finished successfully."""
        workchain = self.ctx.workchain_matdyn

        if not workchain.is_finished_ok:
            self.report(f'The `MatdynBaseWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MATDYN

    def results(self):
        """Add the most important results to the outputs of the work chain."""
        self.out('retrieved', self.ctx.workchain_matdyn.outputs.retrieved)

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
