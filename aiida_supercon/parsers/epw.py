# -*- coding: utf-8 -*-
import re

from aiida import orm
import numpy

from aiida_quantumespresso.calculations.epw import EpwCalculation
from aiida_quantumespresso.parsers.base import BaseParser
from aiida_quantumespresso.utils.mapping import get_logging_container


class EpwParser(BaseParser):
    """``Parser`` implementation for the ``EpwCalculation`` calculation job."""

    success_string='EPW.bib'
    
    _PREFIX = EpwCalculation._PREFIX
    _OUTPUT_SUBFOLDER = './out/'
    _OUTPUT_DOS_FILE = _OUTPUT_SUBFOLDER + _PREFIX + '.dos'
    _OUTPUT_PHDOS_FILE = _PREFIX + '.phdos'
    _OUTPUT_PHDOS_PROJ_FILE = _PREFIX + '.phdos_proj'
    _OUTPUT_A2F_FILE = EpwCalculation._OUTPUT_A2F_FILE
    _OUTPUT_A2F_PROJ_FILE = _PREFIX + '.a2f_proj'
    _OUTPUT_LAMBDA_FS_FILE = _PREFIX + '.lambda_FS'
    _OUTPUT_LAMBDA_K_PAIRS_FILE = _PREFIX + '.lambda_k_pairs'
    
    def parse(self, **kwargs):
        """Parse the retrieved files of a completed ``EpwCalculation`` into output nodes."""
        logs = get_logging_container()

        stdout, parsed_data, logs = self.parse_stdout_from_retrieved(logs)

        base_exit_code = self.check_base_errors(logs)
        if base_exit_code:
            return self.exit(base_exit_code, logs)

        parsed_epw, logs = self.parse_stdout(stdout, logs)
        parsed_data.update(parsed_epw)

        if EpwCalculation._output_elbands_file in self.retrieved.base.repository.list_object_names():
            elbands_contents = self.retrieved.base.repository.get_object_content(EpwCalculation._output_elbands_file)
            self.out('el_band_structure', self.parse_bands(elbands_contents))

        if EpwCalculation._output_phbands_file in self.retrieved.base.repository.list_object_names():
            phbands_contents = self.retrieved.base.repository.get_object_content(EpwCalculation._output_phbands_file)
            self.out('ph_band_structure', self.parse_bands(phbands_contents))

        if self._OUTPUT_A2F_FILE in self.retrieved.base.repository.list_object_names():
            a2f_contents = self.retrieved.base.repository.get_object_content(self._OUTPUT_A2F_FILE)
            a2f_xydata, parsed_a2f = self.parse_a2f(a2f_contents)
            self.out('a2f', a2f_xydata)
            parsed_data.update(parsed_a2f)

        if self._OUTPUT_DOS_FILE in self.retrieved.base.repository.list_object_names():
            dos_contents = self.retrieved.base.repository.get_object_content(self._OUTPUT_DOS_FILE)
            self.out('dos', self.parse_dos(dos_contents))

        if self._OUTPUT_PHDOS_FILE in self.retrieved.base.repository.list_object_names():
            phdos_contents = self.retrieved.base.repository.get_object_content(self._OUTPUT_PHDOS_FILE)
            self.out('phdos', self.parse_phdos(phdos_contents))

        if self._OUTPUT_PHDOS_PROJ_FILE in self.retrieved.base.repository.list_object_names():
            phdos_proj_contents = self.retrieved.base.repository.get_object_content(self._OUTPUT_PHDOS_PROJ_FILE)
            self.out('phdos_proj', self.parse_phdos(phdos_proj_contents))

        if self._OUTPUT_A2F_PROJ_FILE in self.retrieved.base.repository.list_object_names():
            a2f_proj_contents = self.retrieved.base.repository.get_object_content(self._OUTPUT_A2F_PROJ_FILE)
            self.out('a2f_proj', self.parse_a2f_proj(a2f_proj_contents))
        
        if self._OUTPUT_LAMBDA_FS_FILE in self.retrieved.base.repository.list_object_names():
            lambda_FS_contents = self.retrieved.base.repository.get_object_content(self._OUTPUT_LAMBDA_FS_FILE)
            self.out('lambda_FS', self.parse_lambda_FS(lambda_FS_contents))
        
        if self._OUTPUT_LAMBDA_K_PAIRS_FILE in self.retrieved.base.repository.list_object_names():
            lambda_k_pairs_contents = self.retrieved.base.repository.get_object_content(self._OUTPUT_LAMBDA_K_PAIRS_FILE)
            self.out('lambda_k_pairs', self.parse_lambda_k_pairs(lambda_k_pairs_contents))

        pattern_aniso = re.compile(r'aiida\.imag_aniso_gap\d+_(\d+)\.(\d+)$')

        imag_aniso_filelist = [
            (str(float(f"{match.group(1)}.{match.group(2)}")), filename)
            for filename in self.retrieved.base.repository.list_object_names()
            if (match := pattern_aniso.match(filename))
        ]

        if imag_aniso_filelist != []:
            aniso_gap_functions_arraydata = orm.ArrayData()
            Ts = []
            aniso_gap_functions_arrays = []
            for T, imag_aniso_file in imag_aniso_filelist:
                imag_aniso_file_content = self.retrieved.base.repository.get_object_content(imag_aniso_file)
                aniso_gap_function = self.parse_aniso_gap_function(imag_aniso_file_content)
                # aniso_gap_functions_arraydata.set_array(T, aniso_gap_function)
                Ts.append(T)
                aniso_gap_functions_arrays.append(aniso_gap_function)
            
            aniso_gap_functions_arraydata.set_array('temps', numpy.array(Ts))
            aniso_gap_functions_arraydata.set_array('aniso_gap_functions', numpy.array(aniso_gap_functions_arrays))
            self.out('aniso_gap_functions', aniso_gap_functions_arraydata)
        
        if 'max_eigenvalue' in parsed_data:
            self.out('max_eigenvalue', parsed_data.pop('max_eigenvalue'))

        self.out('output_parameters', orm.Dict(parsed_data))

        if 'ERROR_OUTPUT_STDOUT_INCOMPLETE' in logs.error:
            return self.exit(self.exit_codes.get('ERROR_OUTPUT_STDOUT_INCOMPLETE'), logs)

        return self.exit(logs=logs)

    @staticmethod
    def parse_stdout(stdout, logs):
        """Parse the ``stdout``."""

        def parse_max_eigenvalue(stdout_block):
            re_pattern = re.compile(r'\s+([\d\.]+)\s+([\d\.-]+)\s+\d+\s+[\d\.]+\s+\d+\n')
            parsing_block = stdout_block.split('Finish: Solving (isotropic) linearized Eliashberg')[0]
            max_eigenvalue_array = orm.XyData()
            max_eigenvalue_array.set_array(
                'max_eigenvalue', numpy.array(re_pattern.findall(parsing_block), dtype=float)
            )
            return max_eigenvalue_array

        data_type_regex = (
            ('nbndsub', int, re.compile(r'nbndsub\s*=\s*(\d+)')),
            ('ws_vectors_electrons', int, re.compile(r'^\s*Number of WS vectors for electrons\s+(\d+)')),
            ('ws_vectors_phonons', int, re.compile(r'^\s*Number of WS vectors for phonons\s+(\d+)')),
            ('ws_vectors_electron_phonon', int, re.compile(r'^\s*Number of WS vectors for electron-phonon\s+(\d+)')),
            ('max_cores_parallelization', int, re.compile(r'^\s*Maximum number of cores for efficient parallelization\s+(\d+)')),
            ('ibndmin', int, re.compile(r'ibndmin\s*=\s*(\d+)')),
            ('ebndmin', float, re.compile(r'ebndmin\s*=\s*([+-]?[\d\.]+)')),
            ('ibndmax', int, re.compile(r'ibndmax\s*=\s*(\d+)')),
            ('ebndmax', float, re.compile(r'ebndmax\s*=\s*([+-]?[\d\.]+)')),
            ('nbnd_skip', int, re.compile(r'^\s*Skipping the first\s+(\d+)\s+bands:')),
            ('fermi_energy_coarse', float, re.compile(r'^\s*Fermi energy coarse grid =\s*([+-]?[\d\.]+)\s+eV')),
            ('fermi_energy_fine', float, re.compile(r'^\s*Fermi energy is calculated from the fine k-mesh: Ef =\s*([+-]?[\d\.]+)\s+eV')),
            ('fine_q_mesh', lambda m: [int(x) for x in m.split()], re.compile(r'^\s*Using uniform q-mesh:\s+((?:\d+\s*)+)')),
            ('fine_k_mesh', lambda m: [int(x) for x in m.split()], re.compile(r'^\s*Using uniform k-mesh:\s+((?:\d+\s*)+)')),
            ('fermi_level', lambda s: float(s.replace('D', 'E').replace('d', 'E')), re.compile(r'Fermi level \(eV\)\s*=\s*([\d\.D+-]+)')),
            ('DOS', lambda s: float(s.replace('D', 'E').replace('d', 'E')), re.compile(r'DOS\(states/spin/eV/Unit Cell\)\s*=\s*([\d\.D+-]+)')),
            ('electron_smearing', lambda s: float(s.replace('D', 'E').replace('d', 'E')), re.compile(r'Electron smearing \(eV\)\s*=\s*([\d\.D+-]+)')),
            ('fermi_window', lambda s: float(s.replace('D', 'E').replace('d', 'E')), re.compile(r'Fermi window \(eV\)\s*=\s*([\d\.D+-]+)')),
            ('lambda_coupling_strength', float, re.compile(r'Electron-phonon coupling strength\s*=\s*([\d\.]+)')),
            ('Allen_Dynes_Tc', float, re.compile(r'Estimated Allen-Dynes Tc\s*=\s*([\d\.]+) K for muc')),
            ('muc', float, re.compile(r'for muc\s*=\s*([\d\.]+)')),
            ('w_log', float, re.compile(r'Estimated w_log in Allen-Dynes Tc\s*=\s*([\d\.]+) meV')),
            ('BCS_gap', float, re.compile(r'Estimated BCS superconducting gap\s*=\s*([\d\.]+) meV')),
            ('ML_tc', float, re.compile(r'Estimated Tc from machine learning model\s*=\s*([\d\.]+) K')),
        )
        data_block_marker_parser = (
            ('max_eigenvalue', 'Superconducting transition temp. Tc', parse_max_eigenvalue),
        )
        parsed_data = {}
        stdout_lines = stdout.split('\n')

        for line_number, line in enumerate(stdout_lines):
            for data_key, type, re_pattern in data_type_regex:
                match = re.search(re_pattern, line)
                if match:
                    parsed_data[data_key] = type(match.group(1))

            for data_key, data_marker, block_parser in data_block_marker_parser:
                if data_marker in line:
                    parsed_data[data_key] = block_parser(stdout[line_number:])

        return parsed_data, logs

    @staticmethod
    def parse_a2f(content):
        """Parse the contents of the `.a2f` file."""
        a2f_array = numpy.array([line.split() for line in content.splitlines()[1:501]], dtype=float)

        a2f_xydata = orm.XyData()
        a2f_xydata.set_array(
            'frequency', a2f_array[:, 0]
        )
        a2f_xydata.set_array(
            'a2f', a2f_array[:, 1:]
        )
        a2f_xydata.set_array(
            'lambda', numpy.array([
                value for value in re.search(r'Integrated el-ph coupling\n\s+\#\s+([\d\.\s]+)', content).groups()[0].split()
            ], dtype=float
        ))
        a2f_xydata.set_array(
            'degaussq', numpy.array([
                value for value in re.search(r'Phonon smearing \(meV\)\n\s+\#\s+([\d\.\s]+)', content).groups()[0].split()
            ], dtype=float
        ))
        parsed_data = {
            'degaussw': float(re.search(r'Electron smearing \(eV\)\s+([\d\.]+)', content).groups()[0]),
            'fsthick': float(re.search(r'Fermi window \(eV\)\s+([\d\.]+)', content).groups()[0])
        }
        return a2f_xydata, parsed_data

    @staticmethod
    def parse_a2f_proj(content):
        """Parse the contents of the `.a2f_proj` file."""
        import io
        a2f_proj_xydata = orm.XyData()
        a2f_proj_array = numpy.array([line.split() for line in content.splitlines()[1:-1]], dtype=float)

        a2f_proj_xydata.set_array('frequency', a2f_proj_array[:, 0])
        a2f_proj_xydata.set_array('a2f_proj', a2f_proj_array[:, 1:])
        
        return a2f_proj_xydata
    
    @staticmethod
    def parse_dos(content):
        """Parse the contents of the `.dos` file."""
        import io
        dos_xydata = orm.XyData()
        dos = numpy.loadtxt(
            io.StringIO((content)), 
            dtype=float, 
            comments='#'
            )
        
        dos_xydata.set_array('Energy', dos[:, 0])
        dos_xydata.set_array('EDOS', dos[:, 1])
        
        return dos_xydata

    @staticmethod
    def parse_phdos(content):
        """Parse the contents of the `.phdos` file."""
        import io
        phdos_xydata = orm.XyData()
        phdos = numpy.loadtxt(
            io.StringIO((content)), 
            dtype=float, 
            skiprows=1
            )
        phdos_xydata.set_array('Frequency', phdos[:, 0])
        phdos_xydata.set_array('PHDOS', phdos[:, 1])
        
        return phdos_xydata

    @staticmethod
    def parse_lambda_FS(content):
        """Parse the contents of the `.lambda_FS` file."""
        import io
        
        lambda_FS_arraydata = orm.ArrayData()
        lambda_FS = numpy.loadtxt(
            io.StringIO((content)), 
            dtype=float, 
            comments='#'
            )
        
        lambda_FS_arraydata.set_array('kpoints', lambda_FS[:, :3])
        lambda_FS_arraydata.set_array('band', lambda_FS[:, 3])
        lambda_FS_arraydata.set_array('Enk', lambda_FS[:, 4])
        lambda_FS_arraydata.set_array('lambda', lambda_FS[:, 5])
        
        return lambda_FS_arraydata
    
    @staticmethod
    def parse_lambda_k_pairs(content):
        """Parse the contents of the `.lambda_k_pairs` file."""
        import io
        
        lambda_k_pairs_xydata = orm.XyData()
        lambda_k_pairs = numpy.loadtxt(
            io.StringIO((content)), 
            dtype=float, 
            comments='#'
            )
        lambda_k_pairs_xydata.set_array('lambda_nk', lambda_k_pairs[:, 0])
        lambda_k_pairs_xydata.set_array('rho', lambda_k_pairs[:, 1])
        
        return lambda_k_pairs_xydata

    def parse_bands(content):
        """Parse the contents of a band structure file."""
        nbnd, nks = (
            int(v) for v in re.search(
                r'&plot nbnd=\s+(\d+), nks=\s+(\d+)', content
            ).groups()
        )
        kpt_pattern = re.compile(r'\s([\s-][\d\.]+)' * 3)
        band_pattern = re.compile(r'\s+([-\d\.]+)' * nbnd)

        kpts = []
        bands = []

        for number, line in enumerate(content.splitlines()):
            match_kpt = re.search(kpt_pattern, line)
            if match_kpt and number % 2 == 1:
                kpts.append(list(match_kpt.groups()))

            match_band = re.search(band_pattern, line)
            if match_band and number % 2 == 0:
                bands.append(list(match_band.groups()))

        kpoints_data = orm.KpointsData()
        kpoints_data.set_kpoints(numpy.array(kpts, dtype=float))
        bands = numpy.array(bands, dtype=float)

        # raise ValueError('kpts', numpy.array(kpts, dtype=float).shape, 'bands', bands.shape)

        bands_data = orm.BandsData()
        bands_data.set_kpointsdata(kpoints_data)
        bands_data.set_bands(bands, units='meV')

        return bands_data

    @staticmethod
    def parse_aniso_gap_function(content):
        """Parse the contents of the `gap_function.dat` file."""
        import io
        gap_function = numpy.loadtxt(
            io.StringIO((content)), 
            dtype=float, 
            comments='#'
            )

        return gap_function
