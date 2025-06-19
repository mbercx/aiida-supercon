# -*- coding: utf-8 -*-
"""Workchain to run a Quantum ESPRESSO pw.x calculation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common import NotExistent, InputValidationError

from aiida.common.lang import type_check
from aiida.engine import BaseRestartWorkChain, ExitCode, ProcessHandlerReport, process_handler, while_
from aiida.plugins import CalculationFactory, GroupFactory

from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.common.types import ElectronicType, RestartType, SpinType
from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90OptimizeWorkChain


from pathlib import Path

EpwCalculation = CalculationFactory('quantumespresso.epw')

def validate_inputs(inputs, ctx=None):
    """
    Validator for the logic of providing parent folder inputs.
    It enforces that either `parent_folder_epw` is provided, OR the trio of
    `nscf`, `ph`, and `chk` folders are provided together.
    """
    
    # Check if parent_folder_epw is provided
    has_parent_folder_epw = 'parent_folder_epw' in inputs
    
    # Check which of the other three folders are provided
    has_parent_folder_nscf = 'parent_folder_nscf' in inputs
    has_parent_folder_ph = 'parent_folder_ph' in inputs
    has_parent_folder_chk = 'parent_folder_chk' in inputs
    
    # --- Now, we apply your rules ---
    
    # Rule 1: Cannot provide `parent_folder_epw` AND any of the other three.
    if has_parent_folder_epw and any([
        has_parent_folder_nscf, 
        has_parent_folder_ph, 
        has_parent_folder_chk
        ]
    ):
        return "You cannot provide `parent_folder_epw` at the same time as the `nscf/ph/chk` folders."

    # Rule 2: If any of the `nscf/ph/chk` trio are provided, ALL must be provided.
    # `any()` is True if at least one is True. `all()` is True only if all are True.
    # if any([
    #     has_parent_folder_nscf, 
    #     has_parent_folder_ph, 
    #     has_parent_folder_chk
    #     ]) and not all([
    #         has_parent_folder_nscf, 
    #         has_parent_folder_ph, 
    #         has_parent_folder_chk
    #     ]):
            
    #     return "If providing `nscf/ph/chk` parent folders, you must provide all three together."

    # If the logic passes, the validator must return None
    return None
class EpwBaseWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Workchain to run a Quantum ESPRESSO pw.x calculation with automated error handling and restarts."""

    # pylint: disable=too-many-public-methods, too-many-statements

    _process_class = EpwCalculation


    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        
        spec.expose_inputs(
            EpwCalculation, namespace='epw', 
            exclude=(
                'kpoints', 
                'qpoints', 
                'kfpoints', 
                'qfpoints',
                'parent_folder_nscf',
                'parent_folder_ph',
                'parent_folder_epw',
            )
        )
        
        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The structure to compute the epw for fine grid generation'
            )
        
        spec.input(
            'qfpoints', valid_type=orm.KpointsData, required=False,
            help='fine qpoint mesh'
            )

        spec.input(
            'qfpoints_distance', valid_type=orm.Float, required=False,
            help='fine qpoint mesh distance'
            )

        spec.input(
            'kfpoints_factor', valid_type=orm.Int, 
            help='fine kpoint mesh factor'
            )       

        spec.input(
            'parent_folder_nscf', valid_type=orm.RemoteData, required=False,
            help='parent folder of the nscf calculation'
            )
        
        spec.input(
            'parent_folder_ph', valid_type=(orm.RemoteData, orm.RemoteStashFolderData), required=False,
            help='parent folder of the ph calculation'
            )
        
        spec.input(
            'parent_folder_epw', valid_type=(orm.RemoteData, orm.RemoteStashFolderData), required=False,
            help='parent folder of the epw calculation'
            )

        spec.input(
            'parent_folder_chk', valid_type=orm.RemoteData, required=False,
            help='parent folder of the chk file'
            )
        
        spec.input(
            'w90_chk_to_ukk_script', valid_type=orm.RemoteData, required=False,
            help='w90_chk_to_ukk_script'
            )
        
        # spec.inputs.validator = validate_inputs
        
        spec.outline(
            cls.setup,
            cls.validate_parent_folders,
            cls.validate_parallelization,
            cls.validate_kpoints,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(
            EpwCalculation,
            )

        spec.exit_code(201, 'ERROR_INVALID_INPUT_PSEUDO_POTENTIALS',
            message='The explicit `pseudos` or `pseudo_family` could not be used to get the necessary pseudos.')
        spec.exit_code(202, 'ERROR_INVALID_INPUT_KPOINTS',
            message='Neither the `kpoints` nor the `kpoints_distance` input was specified.')
        spec.exit_code(203, 'ERROR_INVALID_INPUT_RESOURCES',
            message='Neither the `options` nor `automatic_parallelization` input was specified. '
                    'This exit status has been deprecated as the check it corresponded to was incorrect.')
        spec.exit_code(204, 'ERROR_INVALID_INPUT_RESOURCES_UNDERSPECIFIED',
            message='The `metadata.options` did not specify both `resources.num_machines` and `max_wallclock_seconds`. '
                    'This exit status has been deprecated as the check it corresponded to was incorrect.')
        spec.exit_code(210, 'ERROR_INVALID_INPUT_AUTOMATIC_PARALLELIZATION_MISSING_KEY',
            message='Required key for `automatic_parallelization` was not specified.'
                    'This exit status has been deprecated as the automatic parallellization feature was removed.')
        spec.exit_code(211, 'ERROR_INVALID_INPUT_AUTOMATIC_PARALLELIZATION_UNRECOGNIZED_KEY',
            message='Unrecognized keys were specified for `automatic_parallelization`.'
                    'This exit status has been deprecated as the automatic parallellization feature was removed.')
        spec.exit_code(212, 'ERROR_MISSING_W90_CHK_TO_UKK_SCRIPT',
            message='w90_chk_to_ukk_script is not provided.')
        spec.exit_code(213, 'ERROR_INVALID_INPUT_PARENT_FOLDER_CHK',
            message='parent_folder_chk is not provided.')
        spec.exit_code(214, 'ERROR_INVALID_INPUT_PARENT_FOLDER_NSCF',
            message='parent_folder_nscf is not provided.')
        spec.exit_code(215, 'ERROR_INVALID_INPUT_PARENT_FOLDER_PH',
            message='parent_folder_ph is not provided.')
        spec.exit_code(216, 'ERROR_INVALID_INPUT_PARENT_FOLDER_EPW',
            message='parent_folder_epw is not provided.')
        spec.exit_code(217, 'ERROR_INCOMPATIBLE_COARSE_GRIDS',
            message='The coarse kpoints and qpoints are not compatible.')
        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unidentified unrecoverable error.')
        spec.exit_code(310, 'ERROR_KNOWN_UNRECOVERABLE_FAILURE',
            message='The calculation failed with a known unrecoverable error.')
        spec.exit_code(320, 'ERROR_INITIALIZATION_CALCULATION_FAILED',
            message='The initialization calculation failed.')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols
        return files(protocols) / 'base.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        protocol=None,
        overrides=None,
        options=None,
        parent_folder_nscf=None,
        parent_folder_ph=None,
        parent_folder_chk=None,
        parent_folder_epw=None,
        w90_chk_to_ukk_script=None,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from .utils.overrides import recursive_copy

        if isinstance(code, str):
            code = orm.load_code(code)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        # Update the parameters based on the protocol inputs
        parameters = inputs['epw']['parameters']
    
        # If overrides are provided, they are considered absolute
        if overrides:
            parameter_overrides = overrides.get('epw', {}).get('parameters', {})
            parameters = recursive_copy(parameters, parameter_overrides)

        metadata = inputs['epw']['metadata']

        if options:
            metadata['options'] = recursive_copy(inputs['epw']['metadata']['options'], options)

        # pylint: disable=no-member
        builder = cls.get_builder()
        builder.structure = structure
        builder.epw['code'] = code
        builder.epw['parameters'] = orm.Dict(parameters)
        builder.epw['metadata'] = metadata
        
        if parent_folder_nscf:
            builder.parent_folder_nscf = parent_folder_nscf
        if parent_folder_ph:
            builder.parent_folder_ph = parent_folder_ph
        if parent_folder_chk:
            builder.parent_folder_chk = parent_folder_chk
        if parent_folder_epw:
            builder.parent_folder_epw = parent_folder_epw
        if w90_chk_to_ukk_script:
            builder.w90_chk_to_ukk_script = w90_chk_to_ukk_script
        
        if 'settings' in inputs['epw']:
            builder.epw['settings'] = orm.Dict(inputs['epw']['settings'])
        if 'parallelization' in inputs['epw']:
            builder.epw['parallelization'] = orm.Dict(inputs['epw']['parallelization'])
        
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        
        if 'qfpoints' in inputs:
            builder.qfpoints = inputs['qfpoints']
        else:
            builder.qfpoints_distance = orm.Float(inputs['qfpoints_distance'])
            
        builder.kfpoints_factor = orm.Int(inputs['kfpoints_factor'])
        
        builder.max_iterations = orm.Int(inputs['max_iterations'])
        # pylint: enable=no-member

        return builder

    def setup(self):
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and create the inputs dictionary in ``self.ctx.inputs``.

        This ``self.ctx.inputs`` dictionary will be used by the ``BaseRestartWorkChain`` to submit the calculations
        in the internal loop.

        The ``parameters`` and ``settings`` input ``Dict`` nodes are converted into a regular dictionary and the
        default namelists for the ``parameters`` are set to empty dictionaries if not specified.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(EpwCalculation, 'epw'))

    def _get_kpoints_from_nscf_folder(self, nscf_folder):
        """
        A robust method to find the k-point mesh from a parent nscf folder.

        This method tries different strategies in order of reliability.

        :param nscf_folder: A RemoteData node from a PwCalculation (nscf).
        :return: A KpointsData node that has mesh information.
        :raises ValueError: If the mesh cannot be found through any strategy.
        """
        pw_calc = nscf_folder.creator

        # Check if the nscf CalcJob's input k-points already has a mesh.
        # This covers the case of a standalone PwBaseWorkChain run.
        try:
            kpoints = pw_calc.inputs.kpoints
            mesh, _ = kpoints.get_kpoints_mesh()
            return kpoints
        except (AttributeError, NotExistent):
            self.report("The input k-points do not contain a mesh.")
            pass # Move on to the next strategy

        w90_workchain = pw_calc.caller.caller
        if not w90_workchain:
            # If there's no caller, we can't use Strategies 2 & 3.
            # We must fall back to the last resort.
            return self._deduce_mesh_from_explicit_kpoints(pw_calc.inputs.kpoints)

        # Check if the caller is a Wannier90 workchain and look for `mp_grid`.
        if w90_workchain.process_class is Wannier90OptimizeWorkChain:
            if hasattr(w90_workchain.inputs, 'optimize_disproj') and w90_workchain.inputs.optimize_disproj:
                wannier_params = w90_workchain.inputs.wannier90_optimal.wannier90.get('parameters')
            else:
                wannier_params = w90_workchain.inputs.wannier90.wannier90.get('parameters')
        elif w90_workchain.process_class is Wannier90BandsWorkChain:
            wannier_params = w90_workchain.inputs.wannier.parameters.get_dict()
        else:
            return self._deduce_mesh_from_explicit_kpoints(pw_calc.inputs.kpoints)
    
        if 'mp_grid' in wannier_params:
            mp_grid = wannier_params['mp_grid']
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh(mp_grid)
            return kpoints
        else:
            return self._deduce_mesh_from_explicit_kpoints(pw_calc.inputs.kpoints)

    def _deduce_mesh_from_explicit_kpoints(self, kpoints_node):
        """The fallback strategy: deduce mesh from a list of k-points."""
        self.report("K-point mesh search: Deduce from coordinates...")
        try:
            import numpy as np
            explicit_kpoints = kpoints_node.get_kpoints()
            nk1 = len(np.unique(explicit_kpoints[:, 0].round(decimals=6)))
            nk2 = len(np.unique(explicit_kpoints[:, 1].round(decimals=6)))
            nk3 = len(np.unique(explicit_kpoints[:, 2].round(decimals=6)))
            if len(explicit_kpoints) != nk1 * nk2 * nk3:
                raise ValueError("Product of deduced dimensions does not match k-point count.")
            mesh = [nk1, nk2, nk3]
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh(mesh)
            return kpoints
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Could not deduce mesh from coordinates. Reason: {e}") from e

    def _get_qpoints_from_ph_folder(self, ph_folder):
        """
        A robust method to find the q-point mesh from a parent ph folder.

        This method tries different strategies in order of reliability.
        1. Check the inputs of the `PhBaseWorkChain`.
        2. Check the inputs of the `PhCalculation` CalcJob itself.
        3. As a last resort, deduce the mesh from the list of coordinates.

        :param ph_folder: A RemoteData node from a PhCalculation.
        :return: A KpointsData node that has mesh information.
        :raises ValueError: If the mesh cannot be found through any strategy.
        """
        ph_calc = ph_folder.creator
        qpoints = ph_calc.inputs.qpoints
        return qpoints
    
    def _get_coarse_grid_from_epw_folder(self, epw_folder):
        """
        Get the coarse grid from the epw folder.
        """
        epw_calc = epw_folder.creator
        if not epw_calc.process_class is EpwBaseWorkChain._process_class:
            self.exit_codes.ERROR_INVALID_INPUT_PARENT_FOLDER_EPW
            
        kpoints = epw_calc.inputs.kpoints
        qpoints = epw_calc.inputs.qpoints
        
        return kpoints, qpoints
    
    def validate_parent_folders(self):
        
        """Validate the parent folders."""
        
        if 'parent_folder_epw' in self.inputs:
            kpoints, qpoints = self._get_coarse_grid_from_epw_folder(self.inputs.parent_folder_epw)
            self.ctx.inputs.kpoints = kpoints
            self.ctx.inputs.qpoints = qpoints
            self.ctx.inputs.parent_folder_epw = self.inputs.parent_folder_epw
            return
        
        else:
            if self.inputs.parent_folder_nscf.is_cleaned:
                self.report("Parent folder of nscf calculation is cleaned. Skipping k-point mesh search.")
                return self.exit_codes.ERROR_INVALID_INPUT_PARENT_FOLDER_NSCF
            
            try:
                kpoints = self._get_kpoints_from_nscf_folder(self.inputs.parent_folder_nscf)
                self.ctx.inputs.kpoints = kpoints
                self.report(f"Successfully determined k-point mesh: {kpoints.get_kpoints_mesh()[0]}")
            except (ValueError, NotExistent) as e:
                self.report(f"Fatal error: Could not determine the k-points mesh. Reason: {e}")
                return self.exit_codes.ERROR_KPOINTS_MESH_NOT_FOUND

            self.ctx.inputs.parent_folder_nscf = self.inputs.parent_folder_nscf

        # if 'parent_folder_ph' in self.inputs:
            if self.inputs.parent_folder_ph.is_cleaned:
                self.report("Parent folder of ph calculation is cleaned. Skipping q-point mesh search.")
                return self.exit_codes.ERROR_INVALID_INPUT_PARENT_FOLDER_PH
            
            try:
                qpoints = self._get_qpoints_from_ph_folder(self.inputs.parent_folder_ph)
                self.ctx.inputs.qpoints = qpoints
                self.report(f"Successfully determined q-point mesh: {qpoints.get_kpoints_mesh()[0]}")
            except (ValueError, NotExistent) as e:
                self.report(f"Fatal error: Could not determine the q-points mesh. Reason: {e}")
                return self.exit_codes.ERROR_QPOINTS_MESH_NOT_FOUND

            self.ctx.inputs.parent_folder_ph = self.inputs.parent_folder_ph
        
        # if hasattr(self.inputs, 'parent_folder_chk'):
            if self.inputs.parent_folder_chk.is_cleaned:
                self.report("Parent folder of chk calculation is cleaned. Skipping k-point mesh search.")
                return self.exit_codes.ERROR_INVALID_INPUT_PARENT_FOLDER_CHK
            
            if not self.inputs.w90_chk_to_ukk_script:
                self.report("w90_chk_to_ukk_script is not provided. Skipping wannierization.")
                return self.exit_codes.ERROR_MISSING_W90_CHK_TO_UKK_SCRIPT 

    def validate_parallelization(self):
        try:
            resources = self.ctx.inputs.metadata.options['resources']
            num_machines = resources['num_machines']
            num_mpiprocs_per_machine = resources['num_mpiprocs_per_machine']
            total_procs = num_machines * num_mpiprocs_per_machine
        except KeyError as e:
            # If resource options are not defined, we cannot perform the check.
            # This is unlikely as AiiDA requires them, but it is a safe fallback.
            self.report(f'Could not determine total MPI processes from metadata.options.resources: {e}. Skipping check.')
            return

        # 2. Extract the `npool` value from either `parallelization` or `settings.CMDLINE`
        npool = None
        parallelization = self.ctx.inputs.get('parallelization', {})
        settings = self.ctx.inputs.get('settings', {})
        cmdline = settings.get('cmdline', [])
        
        if '-npool' in parallelization:
            try:
                npool = int(parallelization['-npool'])
            except (ValueError, TypeError):
                npool = None # Treat non-integer value as not set
        
        elif '-npool' in cmdline:
            try:
                # Find the index of '-npool' and get the next item in the list
                idx = cmdline.index('-npool')
                npool = int(cmdline[idx + 1])
            except (ValueError, IndexError, TypeError):
                # This handles cases like: ['-npool'] (at the end) or ['-npool', 'four']
                npool = None # Treat malformed cmdline as not set

        # 3. Perform the validation checks
        if npool is None:
            parallelization['-npool'] = total_procs
            self.ctx.inputs.parallelization = orm.Dict(parallelization)
        if npool != total_procs:
            self.report(
                f'Validation failed: `npool` value ({npool}) is not equal to the '
                f'total number of MPI processes ({total_procs}).'
            )
            return self.exit_codes.ERROR_INCONSISTENT_NPOOL_SETTING

    def validate_kpoints(self):
        """Validate the inputs related to k-points.

        Either an explicit `KpointsData` with given mesh/path, or a desired k-points distance should be specified. In
        the case of the latter, the `KpointsData` will be constructed for the input `StructureData` using the
        `create_kpoints_from_distance` calculation function.
        """
        from .utils.kpoints import is_compatible
        
        if not is_compatible(self.ctx.inputs.kpoints, self.ctx.inputs.qpoints):
            return self.exit_codes.ERROR_INCOMPATIBLE_COARSE_GRIDS
        
        if all(key not in self.inputs for key in ['qfpoints', 'qfpoints_distance']):
            return self.exit_codes.ERROR_INVALID_INPUT_KPOINTS

        try:
            qfpoints = self.inputs.qfpoints
        except AttributeError:
            inputs = {
                'structure': self.inputs.structure,
                'distance': self.inputs.qfpoints_distance,
                'force_parity': self.inputs.get('qfpoints_force_parity', orm.Bool(False)),
                'metadata': {
                    'call_link_label': 'create_qfpoints_from_distance'
                }
            }
            qfpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        qfpoints_mesh = qfpoints.get_kpoints_mesh()[0]
        kfpoints = orm.KpointsData()
        kfpoints.set_kpoints_mesh([v * self.inputs.kfpoints_factor.value for v in qfpoints_mesh])

        self.ctx.inputs.qfpoints = qfpoints
        self.ctx.inputs.kfpoints = kfpoints
        
    def prepare_process(self):
        """A placeholder for preparing inputs for the next calculation.
        
        Currently, no modifications to `self.ctx.inputs` are needed before
        submission. We rely on the parent `run_process` to create the builder.
        """
        parameters = self.ctx.inputs.parameters.get_dict()

        wannierize = parameters['INPUTEPW'].get('wannierize', False)
        
        if wannierize:
            if 'parent_folder_epw' in self.inputs:
                self.report("Should not have a parent_folder_epw if wannierize is True")
                return self.exit_codes.ERROR_INVALID_INPUT_PARENT_FOLDER_EPW

            elif 'parent_folder_chk' in self.inputs:
                self.report("Should not have a chk folder if wannierize is True")
                return self.exit_codes.ERROR_INVALID_INPUT_PARENT_FOLDER_CHK
            
        else:
            if 'parent_folder_chk' in self.inputs:
                w90_calcjob = self.inputs.parent_folder_chk.creator
                w90_params = w90_calcjob.inputs.parameters.get_dict()
                exclude_bands = w90_params.get('exclude_bands', None) #TODO check this!
            
                if exclude_bands:
                    parameters['INPUTEPW']['bands_skipped'] = f'exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}'

                parameters['INPUTEPW']['nbndsub'] = w90_params['num_wann']
                
                wannier_chk_path = Path(self.inputs.parent_folder_chk.get_remote_path(), 'aiida.chk')
                nscf_xml_path = Path(self.inputs.parent_folder_nscf.get_remote_path(), 'out/aiida.xml')

                prepend_text = self.ctx.inputs.metadata.options.get('prepend_text', '')
                prepend_text += f'\n{self.inputs.w90_chk_to_ukk_script.get_remote_path()} {wannier_chk_path} {nscf_xml_path} aiida.ukk'

                self.ctx.inputs.metadata.options.prepend_text = prepend_text
            
        self.ctx.inputs.parameters = orm.Dict(parameters)
        
    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report(f'Action taken: {action}')


    @process_handler(priority=600)
    def handle_unrecoverable_failure(self, calculation):
        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
        if calculation.is_failed and calculation.exit_status < 400:
            self.report_error_handled(calculation, 'unrecoverable error, aborting...')
            return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

    @process_handler(priority=590, exit_codes=[])
    def handle_known_unrecoverable_failure(self, calculation):
        """Handle calculations with an exit status that correspond to a known failure mode that are unrecoverable.

        These failures may always be unrecoverable or at some point a handler may be devised.
        """
        self.report_error_handled(calculation, 'known unrecoverable failure detected, aborting...')
        return ProcessHandlerReport(True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE)
