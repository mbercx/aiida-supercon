# -*- coding: utf-8 -*-
"""Work chain to compute the electron-phonon coupling.

This work chain is refined to accept the parent folders of the Wannier90 and Phonon work chains.

Firstly it will validate the parent folders.

- If the phonon parent folders are valid, it will use the qpoints from the phonon parent folders.

Then it will check if the wannier90 parent folders are valid. 

- If so, it will check if the kpoints of the wannier90 work chain are compatible with the qpoints of the phonon work chain. 

- If they are not compatible, it will re-generate the kpoints of the wannier90 work chain based on the qpoints of the phonon work chain. Then it will re-run the wannier90 work chain.

- If the wannier90 parent folders are not valid, it will run the wannier90 work chain from scratch.

If none of the parent folders are provided or are not valid, it will run the Wannier90 and Phonon work chains from scratch.

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

import logging
import warnings

from aiida.engine import PortNamespace, ProcessBuilder, WorkChain, ToContext, if_, while_
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_wannier90_workflows.workflows import Wannier90BaseWorkChain, Wannier90BandsWorkChain, Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.optimize import validate_inputs as validate_inputs_w90_intp
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.common.types import WannierProjectionType

from .utils.kpoints import is_compatible

from .base import EpwBaseWorkChain

# def validate_inputs_restart(inputs, ctx=None):
#     """
#     Validate the inputs based on the chosen `restart_mode`.
#     This function is called by AiiDA before the workchain runs.
#     """
#     # The `inputs` argument is a dictionary-like object with the user-provided inputs
#     restart_mode = inputs['restart_mode'].value # .value gives the string, e.g., 'from_phonon'

#     if restart_mode == RestartType.FROM_SCRATCH.value:
#         overrides = inputs.get('overrides', None)
#         if overrides:
#             return "For 'FROM_SCRATCH' mode, the 'overrides' namespace should not be provided."

#     elif restart_mode == RestartType.RESTART_WANNIER.value:
#         overrides = inputs.get('overrides', None)
#         if overrides:
#             if 'ph_base' not in overrides or not (
#                 'parent_folder_ph' in overrides['ph_base']
#                 ):
#                 return "For 'RESTART_WANNIER' mode, 'overrides.ph_base' must be provided."
#         else:
            # return "For 'RESTART_WANNIER' mode, the 'overrides.ph_base' namespace is required."
    # elif restart_mode == RestartType.RESTART_EPW.value:
    #     overrides = inputs.get('overrides', None)
    #     if overrides:
    #         if 'epw' not in overrides or not (
    #             'parent_folder_epw' in overrides['epw']
    #             ):
    #             return "For 'RESTART_EPW' mode, 'overrides.epw' must be provided."
    #     else:
    #         return "For 'RESTART_EPW' mode, the 'overrides' namespace is required."
        
    #     # Check that the required data for this mode is present
    #     required_keys = ['w90_intp', 'ph_base', 'epw']
    #     for key in required_keys:
    #         if key not in inputs['overrides']:
    #             return f"For 'FROM_PHONON' mode, 'overrides.{key}' must be provided."
    # if restart_mode == RestartType.RESTART_PHONON.value:
    #     overrides = inputs.get('overrides', None)
    #     if overrides:
    #         if 'w90_intp' not in overrides:
    #             return "For 'RESTART_PHONON' mode, 'overrides.ph_base' must be provided."
    #         if 'ph_base' in overrides:
    #             return "For 'RESTART_PHONON' mode, 'overrides.ph_base' should not be provided."
    #     else:
    #         return "For 'RESTART_PHONON' mode, the 'overrides' namespace is required."

    # If everything is fine, return None
    # return None
        

        
class EpwB2WWorkChain(ProtocolMixin, WorkChain):
    """Main work chain to start calculating properties using EPW.

    Has support for both the selected columns of the density matrix (SCDM) and
    (projectability-disentangled Wannier function) PDWF projection types.
    """
        
    _QFPOINTS = [1, 1, 1]
    _KFPOINTS_FACTOR = 1
    
    SOURCE_LIST = {
        'ph_base':[
            'DYN_MAT/dynamical-matrix-*',
            'out/_ph0/aiida.dvscf1',
            'out/_ph0/aiida.q_*/aiida.dvscf1', 
            ],
        'epw': [
            'crystal.fmt', 
            'dmedata.fmt', 
            'epwdata.fmt', 
            'selecq.fmt', 
            'dmedata.fmt', 
            'aiida.kgmap', 
            'aiida.kmap', 
            'aiida.ukk', 
            'out/aiida.epmatwp', 
            'save'
            ]
        }
    
    _W90_NAMESPACE = 'w90_intp'
    _PH_NAMESPACE = 'ph_base'
    _EPW_NAMESPACE = 'epw'
    _ALL_NAMESPACES = [_W90_NAMESPACE, _PH_NAMESPACE, _EPW_NAMESPACE]
    
    @classmethod
    def validate_inputs(cls, inputs, ctx=None):  # pylint: disable=unused-argument
        """Validate the inputs of the entire input namespace of `Wannier90OptimizeWorkChain`."""

        if hasattr(inputs, 'w90_intp'):
            validate_inputs_w90_intp(inputs)
        else:
            print('validate nothing')
        
        return None

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input('structure', valid_type=orm.StructureData)
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('kpoints_factor_nscf', valid_type=orm.Int, required=False)
        spec.input('qpoints_distance', valid_type=orm.Float, required=False)
        spec.input('qpoints', valid_type=orm.KpointsData, required=False)
        
        spec.expose_inputs(
            Wannier90OptimizeWorkChain, 
            namespace=cls._W90_NAMESPACE, 
            exclude=(
                'clean_workdir', 'nscf.kpoints', 'nscf.kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `Wannier90OptimizeWorkChain/Wannier90BandsWorkChain`.'
            }
        )
        spec.expose_inputs(
            PhBaseWorkChain, 
            namespace=cls._PH_NAMESPACE,
            exclude=(
                'clean_workdir', 'qpoints', 'qpoints_distance', 'ph.parent_folder'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` that does the `ph.x` calculation.'
            }
        )

        spec.expose_inputs(
            EpwBaseWorkChain, 
            namespace=cls._EPW_NAMESPACE,
            exclude=(
                'clean_workdir',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwBaseWorkChain` that does the `epw.x` calculation.'
            }
        )
        
        # spec.inputs[cls._EPW_NAMESPACE].validator = cls.validate_inputs
        # spec.inputs.validator = cls.validate_inputs

        spec.expose_outputs(
            Wannier90OptimizeWorkChain,
            namespace=cls._W90_NAMESPACE,
            namespace_options={
                'required': False,
                'help': 'Outputs from the `Wannier90OptimizeWorkChain`.'
            }
        )
        spec.expose_outputs(
            PhBaseWorkChain,
            namespace=cls._PH_NAMESPACE,
            namespace_options={
                'required': False,
            }
        )
        spec.expose_outputs(
            EpwBaseWorkChain,
            namespace=cls._EPW_NAMESPACE,
        )

        spec.outline(
            cls.generate_reciprocal_points,
            cls.setup,
            if_(cls.should_run_wannier90)(
                cls.run_wannier90,
                cls.inspect_wannier90,
            ),
            if_(cls.should_run_ph)(
                cls.run_ph,
                cls.inspect_ph,
            ),
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
        spec.exit_code(406, 'ERROR_SCF_PARENT_FOLDER_NOT_FOUND',
            message='The `scf` parent folder was not found')
        spec.exit_code(407, 'ERROR_KPOINTS_QPOINTS_NOT_COMPATIBLE',
            message='The kpoints and qpoints are not compatible')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / f'b2w.yaml'

    @staticmethod
    def set_target_base(
        inputs,
        target_base_prefix,
        ):
        
        if 'stash' not in inputs.metadata['options']:
            computer = inputs.code.computer
            if computer.transport_type == 'core.local':
                target_basepath = Path(computer.get_workdir(), f'{target_base_prefix}-stash').as_posix()
            elif computer.transport_type == 'core.ssh':
                target_basepath = Path(
                    computer.get_workdir().format(username=computer.get_configuration()['username']), f'{target_base_prefix}-stash'
                ).as_posix()
                
            inputs.metadata['options']['stash'] = {
                'target_base': target_basepath,
                'source_list': EpwB2WWorkChain.SOURCE_LIST[target_base_prefix]
                }
    
    @staticmethod
    def get_descendant_workchain(
        b2w: orm.WorkChainNode,
        link_label_filter: str
        ) -> orm.WorkChainNode:
        """
        Parses a completed EpwB2WWorkChain node and extracts its
        successful sub-workchain nodes.

        :param b2w_node: A finished and successful EpwB2WWorkChain node.
        :return: A dictionary containing the 'wannier', 'phonon', and 'epw' sub-nodes.
    """

        # Find the sub-workchains from the outputs (or links).
        # The exact way to get them depends on how you defined the call_link_labels in b2w's outline.
        # Let's assume the call links are 'w90_intp', 'ph_base', and 'epw'.
        try:
            # get_outgoing() returns a list of links, we get the first one's node
            descendant = b2w.base.links.get_outgoing(
                link_label_filter=link_label_filter
                ).first().node
            return descendant
        except AttributeError:
            # ValueError is raised by .one() if it doesn't find exactly one link
            warnings.warn(
                f"Could not find a unique sub-workflow with link label '{link_label_filter}' in <{b2w.pk}>",
                stacklevel=2
                )


    @classmethod
    def get_builder_restart(
        cls, 
        from_b2w_workchain=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        
        from .utils.overrides import get_parent_folder_chk_from_w90_workchain
        
        # Firstly check if the previous EpwB2WWorkChain is finished_ok

        if from_b2w_workchain.process_class != EpwB2WWorkChain:
            raise TypeError(f"Node <{from_b2w_workchain.pk}> is not of type EpwB2WWorkChain.")

        if from_b2w_workchain.is_finished_ok:
            raise ValueError('There is no reason to restart a finished `EpwB2WWorkChain`')
        
        
        # If a previous EpwB2WWorkChain is provided, use it to populate the builder
        builder = from_b2w_workchain.get_builder_restart()
        
        # Firstly get descendants of the previous EpwB2WWorkChain
        # Then check if the previous EpwB2WWorkChain is finished_ok

        # Get the wannier90 workchain
        from_w90_workchain = cls.get_descendant_workchain(
                from_b2w_workchain,
                EpwB2WWorkChain._W90_NAMESPACE
                )
        
        # The logic here is: if there is not w90 namespace in the previous workchain, the workchain must have been restarted that the wannier90 workchain is popped,
        # or there is one but it is finished_ok, we don't need to restart either.
        
        if cls._W90_NAMESPACE not in from_b2w_workchain.inputs:
            builder.pop(cls._W90_NAMESPACE)

            builder[cls._PH_NAMESPACE].ph.parent_folder = from_w90_workchain.inputs.ph.parent_folder
            builder[cls._EPW_NAMESPACE].parent_folder_nscf = from_w90_workchain.inputs.epw.parent_folder_nscf
            builder[cls._EPW_NAMESPACE].parent_folder_chk = from_w90_workchain.inputs.epw.parent_folder_chk

        elif from_w90_workchain.is_finished_ok:
            builder.pop(cls._W90_NAMESPACE)

            builder[cls._PH_NAMESPACE].ph.parent_folder = from_w90_workchain.outputs.scf.remote_folder
            builder[cls._EPW_NAMESPACE].parent_folder_nscf = from_w90_workchain.outputs.nscf.remote_folder
            builder[cls._EPW_NAMESPACE].parent_folder_chk = get_parent_folder_chk_from_w90_workchain(from_w90_workchain)
        else:
            builder[cls._W90_NAMESPACE]._data = from_w90_workchain._data
            # The logic here is: if there is not ph namespace in the previous workchain, it must have been restarted that the phonon workchain is popped,
            # or there is one but it is finished_ok, we don't need to restart either.
        
        from_ph_workchain = cls.get_descendant_workchain(
                from_b2w_workchain,
                EpwB2WWorkChain._PH_NAMESPACE
                )
        
        if cls._PH_NAMESPACE not in from_b2w_workchain.inputs:
            builder.pop(cls._PH_NAMESPACE)

            builder[cls._EPW_NAMESPACE].parent_folder_ph = from_ph_workchain.outputs.remote_folder
            
        elif from_ph_workchain.is_finished_ok:
            builder.pop(cls._PH_NAMESPACE)

            builder[cls._EPW_NAMESPACE].parent_folder_ph = from_ph_workchain.outputs.remote_folder

        else:
            builder[cls._PH_NAMESPACE]._data = from_ph_workchain._data

        # Get the epw workchain
        from_epw_workchain = cls.get_descendant_workchain(
            from_b2w_workchain,
            EpwB2WWorkChain._EPW_NAMESPACE
            )
        if from_epw_workchain.is_finished_ok:
            raise Warning(
                f"The `EpwB2WWorkChain` <{from_b2w_workchain.pk}> is already finished.",
                )

        return builder

    @classmethod
    def get_builder_restart_from_phonon(
        cls, 
        codes, protocol=None, overrides=None,
        from_ph_workchain=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        
        inputs = cls.get_protocol_inputs(protocol, overrides)
        
        from aiida_quantumespresso.calculations.pw import PwCalculation
        
        if not from_ph_workchain or not from_ph_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `PhBaseWorkChain`')
                
        builder = cls.get_builder()
        builder.pop(cls._PH_NAMESPACE)

        # Currently we assume that the parent folder is created by a `PwCalculation`
        # So a StructureData is always associated with it.
        # If the creator is not a `PwCalculation`, we raise an error.
        
        parent_calcjob = from_ph_workchain.inputs.ph.parent_folder.creator
        
        if parent_calcjob.process_class != PwCalculation:
            raise ValueError('Cannot find the structure associated with this `PhBaseWorkChain`')
        
        structure = parent_calcjob.inputs.structure
        
        builder.structure = structure
        w90_intp = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._W90_NAMESPACE, {}),
            pseudo_family=inputs.get(cls._W90_NAMESPACE, {}).get('pseudo_family', None),
            **kwargs
        )
        
        builder[cls._W90_NAMESPACE]._data = w90_intp._data
        
        builder[cls._W90_NAMESPACE].optimize_disproj = orm.Bool(False)
        
        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._EPW_NAMESPACE, {}),
            **kwargs
        )
        
        epw_builder.parent_folder_ph = from_ph_workchain.outputs.remote_folder
        
        builder[cls._EPW_NAMESPACE]._data = epw_builder._data
        
        if 'qpoints_distance' in from_ph_workchain.inputs:
            builder.qpoints_distance = orm.Float(from_ph_workchain.inputs.qpoints_distance)
        
        builder.kpoints_factor_nscf = orm.Int(inputs.get('kpoints_factor_nscf'))
        
        return builder

    @classmethod
    def get_builder_restart_from_wannier90(
        cls, 
        codes, protocol=None, overrides=None,
        from_w90_workchain=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        
        from .utils.overrides import get_parent_folder_chk_from_w90_workchain
        inputs = cls.get_protocol_inputs(protocol, overrides)
        
        builder = cls.get_builder()
        builder.pop(cls._W90_NAMESPACE)
        
        if not from_w90_workchain or not from_w90_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `Wannier90OptimizeWorkChain`')
        
        if from_w90_workchain.outputs.nscf.remote_folder.is_cleaned:
            raise ValueError(
                'The `nscf` remote folder is clean. '
                'Why not restart from scratch?')
        
        builder.structure = from_w90_workchain.inputs.structure
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        builder.kpoints_factor_nscf = orm.Int(inputs['kpoints_factor_nscf'])
        
        ph_base = PhBaseWorkChain.get_builder_from_protocol(
            codes['ph'],
            protocol=protocol, 
            overrides=inputs.get(cls._PH_NAMESPACE, {}), 
            **kwargs
            )
        
        ph_base.ph.parent_folder = from_w90_workchain.outputs.scf.remote_folder
        
        # NOTE: It's not good to use internal ._data property but it's quite convenient here
        # Because for first-time running, we can't provide the parent scf folder so normally we should exclude it from ph_base namespace
        # But if we restart from a finished wannier90 workchain we can provide it so it should not be excluded.
        # TODO: Find a better way to do this

        builder[cls._PH_NAMESPACE]._data = ph_base._data        
        
        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=builder.structure,
            protocol=protocol,
            overrides=inputs.get(cls._EPW_NAMESPACE, {}),
            **kwargs
        )
        
        epw_builder.parent_folder_nscf = from_w90_workchain.outputs.nscf.remote_folder
        epw_builder.parent_folder_chk = get_parent_folder_chk_from_w90_workchain(from_w90_workchain)
        builder[cls._EPW_NAMESPACE]._data = epw_builder._data
        
        return builder
    
    @classmethod
    def get_builder_from_wannier90_and_phonon(
        cls,
        codes,
        protocol=None,
        overrides=None,
        from_w90_workchain=None,
        from_ph_workchain=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        
        from .utils.overrides import get_parent_folder_chk_from_w90_workchain
        builder = cls.get_builder()
        
        builder.structure = from_w90_workchain.inputs.structure
        
        if not from_w90_workchain or not from_w90_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `Wannier90OptimizeWorkChain`')
        
        if not from_ph_workchain or not from_ph_workchain.is_finished_ok:
            raise ValueError('Currently we only accept a finished `PhBaseWorkChain`')
        
        builder.pop(cls._W90_NAMESPACE)
        builder.pop(cls._PH_NAMESPACE)
        
        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=builder.structure,
            protocol=protocol,
            overrides=overrides,
            **kwargs
        )
        
        epw_builder.parent_folder_nscf = from_w90_workchain.outputs.nscf.remote_folder
        epw_builder.parent_folder_chk = get_parent_folder_chk_from_w90_workchain(from_w90_workchain)
        
        epw_builder.parent_folder_ph = from_ph_workchain.outputs.remote_folder
        
        builder[cls._EPW_NAMESPACE]._data = epw_builder._data
        
        return builder
        
    @classmethod
    def get_builder_from_protocol(
        cls, codes, structure, protocol=None, overrides=None,
        wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        reference_bands=None, bands_kpoints=None,
        w90_chk_to_ukk_script=None,
        **kwargs
        )-> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()
        builder.structure = structure
        builder.qpoints_distance = orm.Float(inputs['qpoints_distance'])
        builder.kpoints_factor_nscf = orm.Int(inputs['kpoints_factor_nscf'])
        
        # Set up the wannier90 sub-workchain
        w90_intp_inputs = inputs.get(cls._W90_NAMESPACE, {})
        pseudo_family = w90_intp_inputs.pop('pseudo_family', None)

        w90_intp = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            codes=codes,
            structure=structure,
            overrides=w90_intp_inputs,
            pseudo_family=pseudo_family,
            projection_type=wannier_projection_type,
            reference_bands=reference_bands,
            bands_kpoints=bands_kpoints,
            print_summary=False,
        )
        w90_intp.pop('projwfc', None)
        w90_intp.pop('open_grid', None)
        
        # TODO: Only for testing, will remove later
        w90_intp.optimize_disproj = orm.Bool(False)
        builder[cls._W90_NAMESPACE] = w90_intp

        # Set up the phonon sub-workchain
        ph_base = PhBaseWorkChain.get_builder_from_protocol(
            codes['ph'],
            protocol=protocol, 
            overrides=inputs.get(cls._PH_NAMESPACE, {}), 
            **kwargs
            )
        

        builder[cls._PH_NAMESPACE] = ph_base
        
        epw = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes['epw'],
            structure=structure,
            protocol=protocol,
            overrides=inputs.get(cls._EPW_NAMESPACE, {}),
            w90_chk_to_ukk_script=w90_chk_to_ukk_script,
            **kwargs
        )

        builder[cls._EPW_NAMESPACE]._data = epw._data

        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder._inputs(prune=True)

        return builder

    def setup(self):
        """Setup the work chain."""

        inputs = AttributeDict(
            self.exposed_inputs(
                EpwBaseWorkChain, 
                namespace=self._EPW_NAMESPACE)
            )

        self.ctx.inputs = inputs
        
    def generate_reciprocal_points(self):
        """Generate the reciprocal points."""
        
        self.report('Generating q-points and k-points')
        inputs = {
            'structure': self.inputs.structure,
            'distance': self.inputs.qpoints_distance,
            'force_parity': self.inputs.get('kpoints_force_parity', orm.Bool(False)),
            'metadata': {
                'call_link_label': 'create_qpoints_from_distance'
            }
        }
        
        qpoints = create_kpoints_from_distance(**inputs)
            
        
        self.ctx.qpoints = qpoints

        qpoints_mesh = qpoints.get_kpoints_mesh()[0]
        kpoints_nscf = orm.KpointsData()
        kpoints_nscf.set_kpoints_mesh([v * self.inputs.kpoints_factor_nscf.value for v in qpoints_mesh])

        self.ctx.kpoints_nscf = kpoints_nscf
            

    def should_run_wannier90(self):
        """Check if the wannier90 workflow should be run."""

        return self._W90_NAMESPACE in self.inputs
    
    def run_wannier90(self):
        """Run the wannier90 workflow."""
        
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace=self._W90_NAMESPACE)
        )
        
        try:
            settings = inputs.wannier90.wannier90.settings.get_dict()
        except AttributeError:
            settings = {}

        settings['additional_retrieve_list'] = ['aiida.chk']  
        inputs.wannier90.wannier90.settings = orm.Dict(settings)
        
        inputs.metadata.call_link_label = self._W90_NAMESPACE
        
        set_kpoints(inputs, self.ctx.kpoints_nscf, Wannier90OptimizeWorkChain)

        workchain_node = self.submit(Wannier90OptimizeWorkChain, **inputs)
        self.report(f'launching wannier90 work chain {workchain_node.pk}')

        return ToContext(workchain_w90_intp=workchain_node)

    def inspect_wannier90(self):
        """Verify that the wannier90 workflow finished successfully."""
        from .utils.overrides import get_parent_folder_chk_from_w90_workchain
        
        workchain = self.ctx.workchain_w90_intp

        self.ctx.parent_folder_scf = workchain.outputs.scf.remote_folder
        self.ctx.inputs.parent_folder_nscf = workchain.outputs.nscf.remote_folder
        self.ctx.inputs.parent_folder_chk = get_parent_folder_chk_from_w90_workchain(workchain)
        
        if not workchain.is_finished_ok:
            self.report(f'`Wannier90BandsWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        
    def should_run_ph(self):
        """Check if the phonon workflow should be run."""
        return self._PH_NAMESPACE in self.inputs

    def run_ph(self):
        """Run the `PhBaseWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(PhBaseWorkChain, namespace=self._PH_NAMESPACE))
        
        if self.should_run_wannier90():
            inputs.ph.parent_folder = self.ctx.parent_folder_scf

        inputs.qpoints = self.ctx.qpoints
        
        inputs.metadata.call_link_label = self._PH_NAMESPACE
        
        self.set_target_base(inputs.ph, self._PH_NAMESPACE)
        workchain_node = self.submit(PhBaseWorkChain, **inputs)
        
        self.report(f'launching `ph` {workchain_node.pk}')

        return ToContext(workchain_ph=workchain_node)

    def inspect_ph(self):
        """Verify that the `PhBaseWorkChain` finished successfully."""
        workchain = self.ctx.workchain_ph

        self.ctx.inputs.parent_folder_ph = workchain.outputs.remote_folder
        
        if not workchain.is_finished_ok:
            self.report(f'Electron-phonon `PhBaseWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONON

    def run_epw(self):
        """Run the `epw.x` calculation."""       
        
        # inputs = AttributeDict(self.exposed_inputs(EpwCalculation, namespace='epw'))
        inputs = self.ctx.inputs
        
        fpoints = orm.KpointsData()
        fpoints.set_kpoints_mesh(self._QFPOINTS)
        inputs.qfpoints = fpoints
        inputs.kfpoints_factor = orm.Int(self._KFPOINTS_FACTOR)

        inputs.metadata.call_link_label = self._EPW_NAMESPACE
        self.set_target_base(inputs.epw, self._EPW_NAMESPACE)
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)
        self.report(f'launching `epw` {workchain_node.pk}')

        return ToContext(workchain_epw=workchain_node)

    def inspect_epw(self):
        """Verify that the `epw.x` calculation finished successfully."""
        workchain = self.ctx.workchain_epw

        if not workchain.is_finished_ok:
            self.report(f'`EpwBaseWorkChain` failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPW

        ## TODO: If the workchain finished OK, we will clean the remote folder:
        ## rm out/aiida.epb*, out/aiida.wfc*, out/aiida.save/*
        
    def results(self):
        """Add the most important results to the outputs of the work chain."""
        if 'workchain_w90_intp' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_w90_intp,
                    Wannier90OptimizeWorkChain,
                    namespace=self._W90_NAMESPACE,
                ),
            )
        if 'workchain_ph' in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_ph,
                    PhBaseWorkChain,
                    namespace=self._PH_NAMESPACE,
                ),
            )
        self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_epw,
                    EpwBaseWorkChain,
                    namespace=self._EPW_NAMESPACE,
                ),
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

