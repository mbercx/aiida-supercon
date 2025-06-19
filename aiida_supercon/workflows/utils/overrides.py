from pathlib import Path

from aiida import orm
from aiida.common import AttributeDict
from ...common.restart import RestartType


from aiida.engine import PortNamespace, ProcessBuilder, WorkChain, ToContext, if_, while_
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_wannier90_workflows.workflows import Wannier90BaseWorkChain, Wannier90BandsWorkChain, Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.optimize import validate_inputs as validate_inputs_w90_intp
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.common.types import WannierProjectionType

import collections.abc
from aiida.engine.processes.builder import ProcessBuilderNamespace


def recursive_copy(left: dict, right: dict) -> dict:
    """Recursively merge two dictionaries into a single dictionary.

    If any key is present in both ``left`` and ``right`` dictionaries, the value from the ``right`` dictionary is
    assigned to the key.

    :param left: first dictionary
    :param right: second dictionary
    :return: the recursively merged dictionary
    """
    import collections

    # Note that a deepcopy is not necessary, since this function is called recusively.
    right = right.copy()

    for key, value in left.items():
        if key in right:
            if isinstance(value, collections.abc.Mapping) and isinstance(right[key], collections.abc.Mapping):
                right[key] = recursive_merge(value, right[key])

    merged = left.copy()
    merged.update(right)

    return merged

def recursive_merge(namespace, dict_to_merge):
    """

    :param builder_or_ns: The builder-like object to merge into (will be modified in place).
    :param data_dict: A Python dictionary containing the data to merge.
    """
    # We still need to check the source data_dict explicitly
    if not isinstance(dict_to_merge, collections.abc.Mapping):
        raise TypeError('The data to merge must be a dictionary-like object (a Mapping).')

    for key, value in dict_to_merge.items():
        # The key check to see if we should recurse.
        is_nested_merge = (
            key in namespace and
            isinstance(namespace[key], collections.abc.Mapping) and
            isinstance(value, collections.abc.Mapping)
        )

        if is_nested_merge:
            recursive_merge(namespace[key], value)
        else:
            # Otherwise, the new value simply overwrites the old one.
            namespace[key] = value

    return namespace

def get_parent_folder_chk_from_w90_workchain(
    workchain: orm.WorkChainNode,
    ) -> orm.RemoteData:
    
    if workchain.process_class is Wannier90OptimizeWorkChain:
        if hasattr(workchain.inputs, 'optimize_disproj') and workchain.inputs.optimize_disproj:
            parent_folder_chk = workchain.outputs.wannier90_optimal.remote_folder
        else:
            parent_folder_chk = workchain.outputs.wannier90.remote_folder
    elif workchain.process_class is Wannier90BandsWorkChain:
        parent_folder_chk = workchain.outputs.wannier90.remote_folder
    else:
        raise ValueError(f"Workchain {workchain.process_label} not supported")
    
    return parent_folder_chk

def restart_from_w90_intp(
    w90_intp: orm.WorkChainNode,
    ) -> orm.Dict:
    
    assert w90_intp.is_finished_ok, "Wannier90Optimize workchain did not finish successfully"
    assert w90_intp.process_class in (
        Wannier90OptimizeWorkChain,
    ), "Wannier90OptimizeWorkChain expected"
    
    scf = w90_intp.outputs.scf
    nscf = w90_intp.outputs.nscf

    assert not nscf.remote_folder.is_cleaned, "Nscf remote folder is cleaned"
    
    parent_folder_chk = find_parent_folder_chk_from_workchain(w90_intp)
    restart = {
        'restart_mode': orm.EnumData(RestartType.RESTART_PHONON),
        'overrides': {
            'w90_intp': {
                'parent_folder_scf': scf.remote_folder,
                'parent_folder_nscf': nscf.remote_folder,
                'parent_folder_chk': parent_folder_chk,
            }
        }
    }
    
    return restart


def restart_from_ph_base(
    ph_base: orm.WorkChainNode,
    ) -> orm.Dict:
    
    assert ph_base.is_finished_ok, "PhBase workchain did not finish successfully"
    
    assert ph_base.process_class is PhBaseWorkChain, "PhBaseWorkChain expected"
    
    assert 'remote_stash' in ph_base.outputs, "ph remote stash is cleaned"

    assert not ph_base.outputs.remote_folder.is_cleaned, "ph remote folder is cleaned"
    
    if 'qpoints' in ph_base.inputs:
        qpoints = ph_base.inputs.qpoints
    else:
        create_qpoints_from_distance = ph_base.base.links.get_outgoing(
            link_label_filter='create_qpoints_from_distance'
            ).first().node
        
        qpoints = create_qpoints_from_distance.outputs.qpoints
        
    restart = {
        'overrides': {
            'ph_base': {
                'parent_folder_ph': ph_base.outputs.remote_folder,
                'qpoints': qpoints
            }
        }
    }
    
    return restart

