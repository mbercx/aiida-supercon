# -*- coding: utf-8 -*-
"""Work chain to compute the electron-phonon coupling.

The first step is to compute the electron-phonon (dvscf) on a coarse grid. I think 0.3A^-1 q-point grid should be good but you can use the same k-point grid (as opposed to twice) so it should be cheaper. This step should be done only once

For this step you need to run 

pw.x < scf.in
ph.x < ph.in

Once the calculation is done, you need to rename and gather the results in a "save" folder
This is done with QE/EPW/bin/pp.py script
You just run it as python3 QE/EPW/bin/pp.py
you need to provide the prefix name to the script
What is very important is that the "save" folder is saved in the AiiDA framework
This will be the biggest thing to save and is typically ~ 500 mb to 1 Gb

We can discuss this if we want to keep it or not but if possible, I would say yes
Ok then the EPW step starts

2. Find initial wannier projection

pw.x < scf.in
pw.x < nscf.in
projwfc.x
+ wannier steps from Junfeng workflow?
-> What we need here is the block of inputs provided to epw.in linked to the wannierization 
(wdata(1) = "<wannier inputs>" input variable; is basically a vector of input variables that are directly passed to wannier.x)

e.g.

 wdata(1) = 'bands_plot = .true.'
 wdata(2) = 'begin kpoint_path'
 wdata(3) = 'G 0.00 0.00 0.00 M 0.50 0.00 0.00'
 wdata(4) = 'M 0.50 0.00 0.00 K 0.333333333333 0.333333333333 0.00'
 wdata(5) = 'K 0.333333333333 0.333333333333 0.00 G 0.0 0.0 0.00'
 wdata(6) = 'end kpoint_path'
 wdata(7) = 'bands_plot_format = gnuplot'
 wdata(8) = 'dis_num_iter      = 5000'
 wdata(9) = 'num_print_cycles  = 10'
 wdata(10) = 'dis_mix_ratio     = 1.0'
 wdata(11) = 'conv_tol = 1E-12'
 wdata(12) = 'conv_window = 4'


3. EPW step (in a different folder). You just need to do a soft link to the "save" folder from the above step

You need to do
pw.x <scf.in   (Could potentially be the same as in step 2)
pw.x <nscf.in -> JQ: we can simply use these from the wannier workflow

epw.x < epw1.in 

At the end of this step, you have the electron-phonon matrix element in real space
This needs to be stored for sure
The most important and only big file is "PREFIX..epmatwp"
For TiO, this is 872 Mb ...
From this file you can interpolate to any fine grid density

I mean use for next calculation but you may want to keep it in order to do more convergence later
for example if you find that a 40x40x40 grid is not enough
you dont want to redo step 1 and 2 to get 60x60x60

Files to save:

ln -s ../epw8-conv1/crystal.fmt
ln -s ../epw8-conv1/epwdata.fmt
ln -s ../epw8-conv1/<prefix>.bvec
ln -s ../epw8-conv1/<prefix>.chk
ln -s ../epw8-conv1/<prefix>.kgmap
ln -s ../epw8-conv1/<prefix>.kmap
ln -s ../epw8-conv1/<prefix>.mmn
ln -s ../epw8-conv1/<prefix>.nnkp
ln -s ../epw8-conv1/<prefix>.ukk
ln -s ../epw8-conv1/<prefix>.epmatwp (Note: quite big file!)
ln -s ../epw8-conv1/vmedata.fmt
ln -s ../epw8-conv1/dmedata.fmt
ln -s ../epw8-conv1/save (Is basically the save folder from step 1)

4. EPW interpolation to get Eliashberg Tc

epw.x < epw2.in
epw2.in
 
and basically here you can change the fine grid in epw2.in to converge things
This run can be done in a different folder but you need to soft link a number of files from the previous calculation 2.

"""
from pathlib import Path

from aiida import orm
from aiida.common import AttributeDict

from aiida.engine import WorkChain, ToContext
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.bands import validate_inputs as validate_inputs_bands
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.common.types import WannierProjectionType

class EpwWorkChain(ProtocolMixin, WorkChain):
    """Main work chain to start calculating properties using EPW.

    Has support for both the selected columns of the density matrix (SCDM) and
    (projectability-disentangled Wannier function) PDWF projection types.
    """

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('qpoints_distance', valid_type=orm.Float, default=lambda: orm.Float(0.5))
        spec.input('kpoints_distance_scf', valid_type=orm.Float, default=lambda: orm.Float(0.15))
        spec.input('kpoints_factor_nscf', valid_type=orm.Int, default=lambda: orm.Int(2))
        spec.input('w90_chk_to_ukk_script', valid_type=(orm.RemoteData, orm.SinglefileData))

        spec.expose_inputs(
            Wannier90OptimizeWorkChain, namespace='w90_bands', exclude=(
                'structure', 'clean_workdir',
            ),
            namespace_options={
                'help': 'Inputs for the `Wannier90OptimizeWorkChain/Wannier90BandsWorkChain`.'
            }
        )
        spec.inputs['w90_bands'].validator = validate_inputs_bands
        spec.expose_inputs(
            PhBaseWorkChain, namespace='ph_base', exclude=(
                'clean_workdir', 'ph.parent_folder', 'ph.qpoints'
            ),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` that does the `ph.x` calculation.'
            }
        )
        spec.expose_inputs(
            EpwCalculation, namespace='epw', exclude=(
                'parent_folder_ph', 'parent_folder_nscf', 'kpoints', 'qpoints', 'kfpoints', 'qfpoints'
            ),
            namespace_options={
                'help': 'Inputs for the `EpwCalculation`.'
            }
        )
        spec.output('retrieved', valid_type=orm.FolderData)
        spec.output('epw_folder', valid_type=orm.RemoteStashFolderData)

        spec.outline(
            cls.generate_reciprocal_points,
            cls.run_wannier90,
            cls.inspect_wannier90,
            cls.run_ph,
            cls.inspect_ph,
            cls.run_epw,
            cls.inspect_epw,
            cls.results,
        )
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_PHONON',
            message='The electron-phonon `PhBaseWorkChain` sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_WANNIER90',
            message='The `Wannier90BandsWorkChain` sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_FAILED_EPW',
            message='The `EpwWorkChain` sub process failed')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'epw.yaml'

    @classmethod
    def get_builder_from_protocol(cls, codes, structure, protocol=None, overrides=None,
                                  wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
                                  reference_bands=None, bands_kpoints=None,
                                  **kwargs):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        w90_bands_inputs = inputs.get('w90_bands', {})
        pseudo_family = w90_bands_inputs.pop('pseudo_family', None)

        if wannier_projection_type == WannierProjectionType.ATOMIC_PROJECTORS_QE:
            if reference_bands is None:
                raise ValueError(f'reference_bands must be specified for {wannier_projection_type}')
            w90_bands = Wannier90OptimizeWorkChain.get_builder_from_protocol(
                structure=structure,
                codes=codes,
                pseudo_family=pseudo_family,
                overrides=w90_bands_inputs,
                reference_bands=reference_bands,
                bands_kpoints=bands_kpoints,
            )
            w90_bands.separate_plotting = False
            # pop useless inputs, otherwise the builder validation will fail
            # at validating empty inputs
            w90_bands.pop("projwfc", None)
        elif wannier_projection_type == WannierProjectionType.SCDM:
            w90_bands = Wannier90BandsWorkChain.get_builder_from_protocol(
                structure=structure,
                codes=codes,
                pseudo_family=pseudo_family,
                overrides=w90_bands_inputs,
            )
        else:
            raise ValueError(f'Unsupported wannier_projection_type: {wannier_projection_type}')

        w90_bands.pop('structure', None)
        w90_bands.pop('open_grid', None)

        args = (codes['ph'], None, protocol)
        ph_base = PhBaseWorkChain.get_builder_from_protocol(*args, overrides=inputs.get('ph_base', None), **kwargs)
        ph_base.pop('clean_workdir', None)
        ph_base.ph.pop('qpoints')

        epw_builder = EpwCalculation.get_builder()

        epw_builder.code = codes['epw']
        epw_inputs = inputs.get('epw', None)

        epw_builder.parameters = orm.Dict(epw_inputs['parameters'])

        if 'target_base' not in epw_builder.metadata['options']['stash']:
            epw_computer = codes['epw'].computer
            if epw_computer.transport_type == 'core.local':
                target_basepath = Path(epw_computer.get_workdir(), 'stash').as_posix()
            elif epw_computer.transport_type == 'core.ssh':
                target_basepath = Path(
                    epw_computer.get_workdir().format(username=epw_computer.get_configuration()['username']), 'stash'
                ).as_posix()
            epw_inputs['metadata']['options']['stash']['target_base'] = target_basepath

        epw_builder.metadata = epw_inputs['metadata']
        epw_builder.settings = orm.Dict(epw_inputs['settings'])

        builder = cls.get_builder()
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        builder.kpoints_distance_scf = orm.Float(inputs['kpoints_distance_scf'])
        builder.kpoints_factor_nscf = orm.Int(inputs['kpoints_factor_nscf'])
        builder.structure = structure
        builder.w90_bands = w90_bands
        builder.ph_base = ph_base
        builder.epw = epw_builder
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
        inputs = {
            'structure': self.inputs.structure,
            'distance': self.inputs.kpoints_distance_scf,
            'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
            'metadata': {
                'call_link_label': 'create_kpoints_scf_from_distance'
            }
        }
        kpoints_scf = create_kpoints_from_distance(**inputs)

        qpoints_mesh = qpoints.get_kpoints_mesh()[0]
        kpoints_nscf = orm.KpointsData()
        kpoints_nscf.set_kpoints_mesh([v * self.inputs.kpoints_factor_nscf.value for v in qpoints_mesh])

        self.ctx.qpoints = qpoints
        self.ctx.kpoints_scf = kpoints_scf
        self.ctx.kpoints_nscf = kpoints_nscf

    def run_wannier90(self):
        """Run the wannier90 workflow."""
        if 'projwfc' in self.inputs.w90_bands:
            self.report('Running a Wannier90BandsWorkChain.')
            w90_class = Wannier90BandsWorkChain
        else:
            self.report('Running a Wannier90OptimizeWorkChain.')
            w90_class = Wannier90OptimizeWorkChain

        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace='w90_bands')
        )
        inputs.metadata.call_link_label = 'w90_bands'
        inputs.structure = self.inputs.structure

        set_kpoints(inputs, self.ctx.kpoints_nscf, w90_class)
        inputs['scf']['kpoints'] = self.ctx.kpoints_scf

        workchain_node = self.submit(w90_class, **inputs)
        self.report(f'launching wannier90 work chain {workchain_node.pk}')

        return ToContext(workchain_w90_bands=workchain_node)

    def inspect_wannier90(self):
        """Verify that the wannier90 workflow finished successfully."""
        workchain = self.ctx.workchain_w90_bands

        if not workchain.is_finished_ok:
            self.report(f'`Wannier90BandsWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

    def run_ph(self):
        """Run the `PhBaseWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(PhBaseWorkChain, namespace='ph_base'))

        scf_base_wc = self.ctx.workchain_w90_bands.base.links.get_outgoing(link_label_filter='scf').first().node
        inputs.ph.parent_folder = scf_base_wc.outputs.remote_folder

        inputs.ph.qpoints = self.ctx.qpoints

        inputs.metadata.call_link_label = 'ph_base'
        workchain_node = self.submit(PhBaseWorkChain, **inputs)
        self.report(f'launching `ph` {workchain_node.pk}')

        return ToContext(workchain_ph=workchain_node)

    def inspect_ph(self):
        """Verify that the `PhBaseWorkChain` finished successfully."""
        workchain = self.ctx.workchain_ph

        if not workchain.is_finished_ok:
            self.report(f'Electron-phonon PhBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONON

    def run_epw(self):
        """Run the `epw.x` calculation."""        
        inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw'))

        inputs.parent_folder_ph = self.ctx.workchain_ph.outputs.remote_folder

        nscf_base_wc =  self.ctx.workchain_w90_bands.base.links.get_outgoing(link_label_filter='nscf').first().node
        inputs.parent_folder_nscf = nscf_base_wc.outputs.remote_folder

        fine_points = orm.KpointsData()
        fine_points.set_kpoints_mesh([1, 1, 1])

        inputs.kpoints = self.ctx.kpoints_nscf
        inputs.kfpoints = fine_points
        inputs.qpoints = self.ctx.qpoints
        inputs.qfpoints = fine_points      

        parameters = inputs.parameters.get_dict()

        wannier_params = self.ctx.workchain_w90_bands.inputs.wannier90.wannier90.parameters.get_dict()
        exclude_bands = wannier_params.get('exclude_bands') #TODO check this!
        if exclude_bands:
            parameters['INPUTEPW']['bands_skipped'] = f'exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}'

        parameters['INPUTEPW']['nbndsub'] = wannier_params['num_wann']
        inputs.parameters = orm.Dict(parameters)

        if 'projwfc' in self.inputs.w90_bands:
            w90_remote_data = self.ctx.workchain_w90_bands.outputs.wannier90__remote_folder
        else:
            w90_remote_data = self.ctx.workchain_w90_bands.outputs.wannier90_optimal__remote_folder

        wannier_chk_path = Path(w90_remote_data.get_remote_path(), 'aiida.chk')
        nscf_xml_path = Path(self.ctx.workchain_w90_bands.outputs.nscf.remote_folder.get_remote_path(), 'out/aiida.xml')

        prepend_text = inputs.metadata.options.get('prepend_text', '')
        prepend_text += f'\n{self.inputs.w90_chk_to_ukk_script.get_remote_path()} {wannier_chk_path} {nscf_xml_path} aiida.ukk'
        inputs.metadata.options.prepend_text = prepend_text

        inputs.metadata.call_link_label = 'epw'

        calcjob_node = self.submit(EpwCalculation, **inputs)
        self.report(f'launching `epw` {calcjob_node.pk}')

        return ToContext(calcjob_epw=calcjob_node)

    def inspect_epw(self):
        """Verify that the `epw.x` calculation finished successfully."""
        calcjob = self.ctx.calcjob_epw

        if not calcjob.is_finished_ok:
            self.report(f'`EpwCalculation` failed with exit status {calcjob.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPW

    def results(self):
        """Add the most important results to the outputs of the work chain."""
        self.out('retrieved', self.ctx.calcjob_epw.outputs.retrieved)
        self.out('epw_folder', self.ctx.calcjob_epw.outputs.remote_stash)

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
