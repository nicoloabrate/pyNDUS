"""
author: N. Abrate.

file: sensitivity.py

description: read sensitivity profiles and construct an object to perform operations.
"""
import re
import numpy as np
import serpentTools as st
from pathlib import Path
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
try:
    from ._sensitivity_algebra import SensitivityAlgebraMixin
    from .channels import ERANOS_CHANNELS, SensitivityChannel, eranos_channel
except ImportError:  # when run as a script
    from _sensitivity_algebra import SensitivityAlgebraMixin
    from channels import ERANOS_CHANNELS, SensitivityChannel, eranos_channel
try:
    import pyNDUS.utils as utils
except ModuleNotFoundError:  # when run as a script
    try:
        from . import utils
    except ImportError:
        import utils

SERPENT_NUMERIC_ALIASES = {
    452: "nubar total",
    455: "nubar delayed",
    456: "nubar prompt",
}


@dataclass(frozen=True)
class SensitivitySpatialZone:
    """Description of one spatial zone on the sensitivity material axis."""
    profile: int = None
    zone: int = None
    kind: str = "profile"
    entries: tuple = ()

    @property
    def label(self):
        """Return the label used on the existing pyNDUS material axis."""
        if self.kind == "profile":
            if self.profile is None:
                return "profile"
            return f"profile {self.profile}"

        noun = "cell" if self.kind == "cell" else "material"
        if len(self.entries) != 1:
            noun += "s"
        entries = ", ".join(str(entry) for entry in self.entries) or "unknown"
        return f"profile {self.profile} zone {self.zone} {noun} {entries}"


@dataclass(frozen=True)
class SensitivityNuclideInstance:
    """Description of one nuclide library instance on the sensitivity ZA axis."""
    zaid: int
    ace_suffix: str = None
    temperature: object = None

    @property
    def label(self):
        """Return a compact label for the nuclide library instance."""
        if self.ace_suffix is None:
            return str(self.zaid)
        if self.temperature is None:
            return f"{self.zaid}{self.ace_suffix}"
        return f"{self.zaid}{self.ace_suffix} ({self.temperature})"


class Sensitivity(SensitivityAlgebraMixin):
    """
    Class to read, store, and process multi-group sensitivity profiles from Serpent or ERANOS output files.

    Parameters
    ----------
    sensitivity_path : str or Path
        Path to the sensitivity file to be read.

    Attributes
    ----------
    filepath : Path
        Path to the sensitivity file.
            -'serpent': should end with "_sens0.m".
            -'eranos': should end with ".eranos33" or ".eranos1968".
            -'mcnp': should end with ".mcnp".
    reader : str
        Sensitivity file format ('serpent', 'eranos', or 'mcnp').
    responses : list
        List of response parameters (e.g., 'keff', 'beff', etc.).
    materials : OrderedDict
        Mapping of material/spatial-zone labels to their indices.
    spatial_zones : OrderedDict
        For MCNP readers, mapping of material-axis labels to
        :class:`SensitivitySpatialZone` metadata.
    zaid : OrderedDict
        Mapping of ZAID numbers (e.g., 942390) to their indices. Serpent's
        aggregate all-isotope profile is retained as ZAID ``0``.
    nuclide_instances : OrderedDict
        For MCNP readers, mapping of ``(ZAID, ACE suffix)`` pairs to
        :class:`SensitivityNuclideInstance` metadata.
    ace_suffixes : OrderedDict
        For MCNP readers, mapping of ZAID numbers to the ACE suffixes found in
        the file.
    zais : OrderedDict
        Mapping of ZA strings (e.g., 'Pu-239') to their indices.
    channels : OrderedDict
        Mapping of :class:`SensitivityChannel` objects to their axis indices.
    MTs : OrderedDict
        Compatibility alias for ``channels``.
    group_structure : iterable
        Energy group structure, stored in ascending order.
    n_groups : int
        Number of energy groups. Sensitivity arrays are stored in ascending
        energy order, consistently with ``group_structure``.
    sens : np.ndarray
        Sensitivity profiles whose shape is
        (nResp, nMat, nZaid, nChannels, nE). For MCNP readers, the material
        axis stores the parsed spatial zones; ZAIDs with multiple ACE suffixes
        require an explicit aggregation policy before they are selected without
        ``ace_suffix`` or ``temperature``.
    sens_nuclide_instances : np.ndarray
        For MCNP readers, sensitivity profiles before aggregation over ACE
        suffixes. The shape is (nResp, nMat, nNuclideInstances, nChannels, nE).
    sens_rsd : np.ndarray
        Relative standard deviations of the sensivity profile. When not available
        (e.g., for deterministic calculations), it is None.
        The shape is (nResp, nMat, nZaid, nChannels, nE).

    Methods
    -------
    from_serpent()
        Read and parse a Serpent sensitivity file using serpentTools.
    from_eranos()
        Read and parse an ERANOS sensitivity output file. (its extension must be either 
        ".eranos33" for the ECCO-33 group structure or ".eranos1968" for the ECCO-1968
        group structure).
    NormalizeSensProfile(sens_profile, energy_vector)
        Normalize a sensitivity profile in lethargy.
    get(resp=None, mat=None, MT=None, channel=None, za=None, ace_suffix=None,
        temperature=None, g=None)
        Extract sensitivity profiles and uncertainties for specified parameters.
        -'resp': response(s) to extract (e.g., 'keff', 'beff').
        -'mat': material(s) to extract (e.g., 'total', 'm1').
        -'MT': MT number(s) to extract when unambiguous.
        -'channel': SensitivityChannel object(s) to extract.
        -'za': ZA string(s) or number(s) to extract (e.g., 'Pu-239', 942390).
        -'ace_suffix': MCNP ACE suffix(es) to extract before ZAID aggregation.
        -'temperature': MCNP ACE temperature(s) to extract before ZAID aggregation.
        -'g': energy group(s) to extract (e.g., 1, 2, ..., n_groups).

    Raises
    ------
    SensitivityError
        If the file format is not recognized or if the file structure is not as expected.
    """

    def __init__(self,
                 sensitivity_path,
                 duplicate_policy="raise",
                 mcnp_ace_temperatures=None,
                 mcnp_ace_aggregation="raise"):
        """
        Initialize the Sensitivity object and read the sensitivity file.

        Parameters
        ----------
        sensitivity_path : str, Path, or sequence of str/Path
            Path to one sensitivity file, or multiple Serpent ``*_sens0.m``
            files to merge.
            -'serpent': should end with "_sens0.m".
            -'eranos': should end with ".eranos33" or ".eranos1968".
            -'mcnp': should end with ".mcnp".
        duplicate_policy : str, optional
            Policy for handling duplicate entries in the sensitivity file. Options are:
            -'raise': raise an error if duplicates are found (default).
            -'keep_first': keep the first occurrence and ignore subsequent duplicates.
            -'keep_last': keep the last occurrence and ignore previous duplicates.
        mcnp_ace_temperatures : dict, optional
            Mapping from MCNP ACE suffixes to temperatures. Keys may be passed
            as ``".00c"`` or ``"00c"``. Used only by the MCNP reader.
        mcnp_ace_aggregation : {"raise", "sum"}, optional
            Policy used when an MCNP file reports the same ZAID with multiple
            ACE suffixes and a profile is requested without ``ace_suffix`` or
            ``temperature``. ``"raise"`` avoids implicit aggregation (default);
            ``"sum"`` returns the coherent sum over ACE suffixes.
        """
        # --- validate and assign path
        self.filepath = sensitivity_path
        self.mcnp_ace_temperatures = self._normalize_mcnp_ace_temperatures(
            mcnp_ace_temperatures)
        self.mcnp_ace_aggregation = self._normalize_mcnp_ace_aggregation(
            mcnp_ace_aggregation)

        # single file
        if not self.is_multifile:
            # --- read the sensitivity file
            self.reader = self.get_reader()

            if self.reader == "serpent":
                self.from_serpent()
            elif self.reader == "eranos":
                self.from_eranos()
            elif self.reader == "mcnp":
                self.from_mcnp()
            return
        else:
            # multi files
            self._from_multiple(self.filepath,
                                duplicate_policy=duplicate_policy)

    def from_serpent(self):
        """
        Read and parse a Serpent sensitivity file using serpentTools.

        Raises
        ------
        SensitivityError
            If the file structure is not as expected.
        """
        sens = st.read(self.filepath)

        try:
            self.responses = list(sens.sensitivities.keys())
        except ValueError:
            raise SensitivityError(
                "The structure of serpentTools.parsers.sensitivity object might have changed!"
            )

        try:
            self.materials = list(sens.materials.keys())
        except ValueError:
            raise SensitivityError(
                "The structure of serpentTools.parsers.sensitivity object might have changed!"
            )

        if not isinstance(sens.zais, OrderedDict):
            raise SensitivityError(
                f"serpentTools.parsers.sensitivity.zais must be an OrderedDict, not of type {type(sens.zais)}."
                "The structure of serpentTools.parsers.sensitivity object might have changed!"
            )
        else:
            # sens.zais contains integers, not strings
            self.zaid = sens.zais.keys()

        self.zais = self.zaid.keys()

        if not isinstance(sens.perts, OrderedDict):
            raise SensitivityError(
                f"serpentTools.parsers.sensitivity.perts must be an OrderedDict, not of type {type(sens.perts)}."
                "The structure of serpentTools.parsers.sensitivity object might have changed!"
            )
        else:
            self.MTs = list(sens.perts.keys())

        if not isinstance(sens.sensitivities, dict):
            raise SensitivityError(
                "serpentTools.parsers.sensitivity.sensitivities must be a dict, not of type "
                f"{type(sens.sensitivities)}."
                "The structure of serpentTools.parsers.sensitivity object might have changed!"
            )
        else:
            self.sens = sens.sensitivities
            self.sens_rsd = sens.sensitivities

        if not isinstance(sens.energies, np.ndarray):
            raise SensitivityError(
                f"serpentTools.parsers.sensitivity.energies must be a np.array, not of type {type(sens.energies)}."
                "The structure of serpentTools.parsers.sensitivity object might have changed!"
            )
        else:
            self.energy_unit = "MeV"
            self.group_structure = sens.energies

    def from_eranos(self):
        """
        Read and parse an ERANOS sensitivity output file.

        Raises
        ------
        SensitivityError
            If the file format or structure is not as expected.
        """
        if 'eranos33' in self.filepath.suffix:
            energy_grid = utils.ECCO33
        elif 'eranos1968' in self.filepath.suffix:
            energy_grid = utils.ECCO1968
        else:
            raise SensitivityError(
                f"Cannot read file format {self.filepath.name}")

        nE = len(energy_grid) - 1
        header_found = False  # Flag to activate search of integral parameter
        material_found = True  # Flag to reject repeated table of results for material
        total_isotopes_found = True  # Flag to reject last review
        reactor_occurrence = 0
        columns = None
        ig = 0
        iline = 0
        iline_data = 0
        dict_read = {}

        responses = []
        responsesapp = responses.append
        materials = []
        materialsapp = materials.append
        zais = []
        zaisapp = zais.append
        perturbations = []
        perturbation_channels = []
        active_sorted_channels = []

        header_pat = "SENSITIVITY COEFFICIENTS"
        param_pat = re.compile(r'\b(\w+)\s+SENSITIVITY\b')
        mat_pat = re.compile(r'\bDOMAIN\s+(\w+)')
        iso_pat = re.compile(r'\bISOTOPE\s+(\w+)')
        data_pat = re.compile(r'\bGROUP\s+(\w+)')

        eranos_channels = list(ERANOS_CHANNELS.values())
        idx_MT_sorted = []

        # hard-coded, sanity check on order is reported later
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for iline, line in enumerate(f):
                if header_pat in line:
                    header_found = True
                    break

            for iline, line in enumerate(f):
                # Read the Integral Parameter calculated: keff, beff, leff, etc in 'Found_param'
                match = param_pat.search(line)
                if match:
                    Found_param = match.group(1)
                    resp = Found_param.lower()
                    if resp not in responses:
                        responsesapp(resp)
                        dict_read[resp] = {}
                # Read the material saved in 'mat_element'
                mat_line = mat_pat.search(line)
                # It works only in the last material which is a review of the full core
                if re.search(r'\bREACTOR\s', line):
                    reactor_occurrence += 1
                    if reactor_occurrence > 1:
                        break
                    else:
                        mat_line = True

                if mat_line:
                    if material_found:
                        if reactor_occurrence > 0:
                            mat_element = "REACTOR"
                        else:
                            mat_element = mat_line.group(1)

                        materialsapp(mat_element)
                        material_found = False
                        total_isotopes_found = False
                        dict_read[resp][mat_element] = {}
                    elif material_found == False:
                        total_isotopes_found = True
                        material_found = True
                # Read the isotope into the material saved in 'isotope'
                isotope_line = iso_pat.search(line)
                if isotope_line:
                    isotope = isotope_line.group(1)
                    m = re.match(r"([A-Za-z]+)(\d+)", isotope)
                    s = f"{m.group(1)}-{m.group(2)}"
                    if s not in zais:
                        dict_read[resp][mat_element][s] = {}
                        zaisapp(s)

                data_line = data_pat.search(line)
                # Read the tables of results excluding the total
                if data_line and not total_isotopes_found:
                    if len(perturbations) == 0:
                        perturbations = line.split()
                        perturbations.remove('GROUP')
                        perturbations.remove('SUM')
                        unknown = [
                            label for label in perturbations
                            if label not in ERANOS_CHANNELS
                        ]
                        if unknown:
                            raise SensitivityError(
                                "Unknown ERANOS perturbation column(s): "
                                f"{unknown}. The structure of the ERANOS "
                                "output file might have changed!")

                    if columns is None:
                        num_column = len(
                            perturbations
                        )  # len(perturbations) + 2 - (1 if data_line else 0)
                        columns = -np.ones((num_column, nE))
                    table_channels = [
                        eranos_channel(label, isotope=s)
                        for label in perturbations
                    ]
                    for channel in table_channels:
                        if channel not in perturbation_channels:
                            perturbation_channels.append(channel)
                    idx_MT_sorted = sorted(
                        range(len(table_channels)),
                        key=lambda index: table_channels[index].average_MT)
                    active_sorted_channels = [
                        table_channels[index] for index in idx_MT_sorted
                    ]
                    iline_data = iline + 1
                    ig = 0

                if iline >= iline_data and iline < iline_data + nE and iline_data > 0:
                    line = line.strip()
                    if not line:
                        break  # No more data
                    parts = line.split()
                    parts = parts[1:num_column + 1]
                    if len(parts) != num_column:
                        raise ValueError(
                            "Inconsistency between N of titles and N of columns"
                        )
                        continue
                    file_values = np.array([float(val) for val in parts])
                    columns[:, ig] = file_values[idx_MT_sorted]
                    ig += 1

                if ig == nE:
                    dict_read[resp][mat_element][s] = OrderedDict(
                        (channel, columns[index, :].copy())
                        for index, channel in enumerate(active_sorted_channels))
                    ig = 0

        try:
            self.responses = responses
            self.energy_unit = "eV"
            self.group_structure = energy_grid
        except ValueError:
            raise SensitivityError(
                "The structure of the ERANOS output file might have changed!")

        # --- get materials, isotopes and reactions
        perturbations = sorted(perturbation_channels or eranos_channels,
                               key=lambda channel: channel.average_MT)
        try:
            self.materials = list(materials)
            self.zaid = [utils.zais2zaid(za) for za in zais]
            self.zais = self.zaid.keys()
            self.channels = perturbations
        except ValueError:
            raise SensitivityError(
                "The structure of the ERANOS output file might have changed!")

        self.sens_rsd = None
        self.sens = dict_read

    def from_mcnp(self):
        """
        Read and parse an MCNP sensitivity output file.

        MCNP output before the first ``nuclear data ... sensitivity
        coefficients`` section is ignored. MCNP spatial zones are stored on the
        existing pyNDUS material axis and described in ``spatial_zones``.
        """
        parsed = self._parse_mcnp_sensitivity_file()
        self.responses = parsed["responses"]
        self.materials = parsed["materials"]
        self.spatial_zones = parsed["spatial_zones"]
        self.zaid = parsed["zaids"]
        self.zais = self.zaid.keys()
        self.nuclide_instances = parsed["nuclide_instances"]
        self.ace_suffixes = parsed["ace_suffixes"]
        self.channels = parsed["channels"]
        self.energy_unit = "MeV"
        self.group_structure = parsed["group_structure"]
        self._sens = parsed["sens"]
        self._sens_rsd = parsed["sens_rsd"]
        self._sens_nuclide_instances = parsed["sens_nuclide_instances"]
        self._sens_nuclide_instances_rsd = parsed["sens_nuclide_instances_rsd"]

    @staticmethod
    def _mcnp_zaid_to_pyndus(zaid):
        """Convert an MCNP ZAID such as ``94239.00c`` to pyNDUS ZAID style (i.e. that of Serpent)."""
        return int(zaid.split(".")[0]) * 10

    @staticmethod
    def _normalize_mcnp_ace_suffix(suffix):
        """Normalize an MCNP ACE suffix to the form ``.00c``."""
        suffix = str(suffix).strip().lower()
        if "." in suffix:
            return suffix[suffix.find("."):]
        return f".{suffix}"

    @classmethod
    def _normalize_mcnp_ace_temperatures(cls, temperatures):
        """Normalize user-provided MCNP ACE suffix-temperature metadata."""
        if temperatures is None:
            return {}
        if not isinstance(temperatures, dict):
            raise ValueError(
                "'mcnp_ace_temperatures' must be a dict mapping suffixes to temperatures."
            )
        return {
            cls._normalize_mcnp_ace_suffix(suffix): temperature
            for suffix, temperature in temperatures.items()
        }

    @staticmethod
    def _normalize_mcnp_ace_aggregation(policy):
        """Normalize the MCNP ACE aggregation policy."""
        allowed = {"raise", "sum"}
        if policy not in allowed:
            raise ValueError(
                f"'mcnp_ace_aggregation' must be one of {sorted(allowed)}, "
                f"not {policy!r}.")
        return policy

    @classmethod
    def _mcnp_ace_suffix(cls, zaid):
        """Return the ACE suffix from an MCNP ZAID such as ``94239.00c``."""
        return cls._normalize_mcnp_ace_suffix(zaid)

    @staticmethod
    def _mcnp_temperature_matches(value, target):
        """Return whether two MCNP ACE temperatures match."""
        if value is None:
            return False
        if isinstance(value,
                      (int, float)) and isinstance(target, (int, float)):
            return np.isclose(float(value), float(target))
        return value == target

    @staticmethod
    def _format_mcnp_available_temperatures(instances):
        """Return a compact string with the MCNP temperatures in the object."""
        temperatures = []
        for instance in instances.values():
            if instance.temperature is not None and instance.temperature not in temperatures:
                temperatures.append(instance.temperature)
        return temperatures or "none"

    @staticmethod
    def _mcnp_profile_zone(profile_number):
        """Return the fallback spatial zone used when MCNP prints no zone."""
        return SensitivitySpatialZone(int(profile_number))

    def _parse_mcnp_sensitivity_file(self):
        """Parse MCNP sensitivity coefficients into pyNDUS arrays."""
        # --- patterns for MCNP sections, profiles, spatial zones and tables
        section_re = re.compile(
            r"^\s*nuclear data\s+(.+?)\s+sensitivity coefficients\s*$",
            re.IGNORECASE)
        profile_re = re.compile(r"^\s*sensitivity profile\s+(\d+)\s*$",
                                re.IGNORECASE)
        spatial_re = re.compile(
            r"^\s*spatial zone\s+(\d+)\s+covering\s+(cell|material)\(s\):\s*$",
            re.IGNORECASE)
        channel_re = re.compile(r"^\s*(\d{4,6}\.\d{2}c)\s+(.+?)\s*$",
                                re.IGNORECASE)
        row_re = re.compile(
            r"^\s*([+-]?\d+\.\d+E[+-]\d+)\s+([+-]?\d+\.\d+E[+-]\d+)\s+"
            r"([+-]?\d+\.\d+E[+-]\d+)\s+([+-]?\d+(?:\.\d+)?(?:E[+-]\d+)?)\s*$",
            re.IGNORECASE)

        # --- containers for sparse profiles before building dense arrays
        lines = self.filepath.read_text(errors="replace").splitlines()
        data = OrderedDict()
        responses = []
        materials = []
        spatial_zones = OrderedDict()
        zaids = []
        nuclide_instances = OrderedDict()
        ace_suffixes = OrderedDict()
        channels = []
        group_structure = None
        current_resp = None
        current_profile = None
        current_zone = None
        current_mat = None
        found_section = False

        # --- scan the file after the first nuclear-data sensitivity section
        i = 0

        while i < len(lines):
            line = lines[i]
            section_match = section_re.match(line)
            if section_match:
                # --- start a new response block (keff, beta, etc.)
                current_resp = " ".join(section_match.group(1).split()).lower()
                if current_resp not in responses:
                    responses.append(current_resp)
                data.setdefault(current_resp, OrderedDict())
                current_profile = None
                current_zone = None
                current_mat = None
                found_section = True
                i += 1
                continue

            if not found_section:
                i += 1
                continue

            profile_match = profile_re.match(line)
            if profile_match:
                # --- MCNP sensitivity profile; spatial zones may follow
                current_profile = int(profile_match.group(1))
                current_zone = self._mcnp_profile_zone(current_profile)
                current_mat = None
                i += 1
                continue

            spatial_match = spatial_re.match(line)
            if spatial_match and current_resp is not None:
                # --- CELL/MAT spatial zone, stored on the material axis
                if current_profile is None:
                    current_profile = 1
                entries, next_i = self._read_mcnp_spatial_zone_entries(
                    lines, i + 1, channel_re, section_re, profile_re,
                    spatial_re)
                current_zone = SensitivitySpatialZone(
                    current_profile, int(spatial_match.group(1)),
                    spatial_match.group(2).lower(), tuple(entries))
                current_mat = current_zone.label
                if current_mat not in materials:
                    materials.append(current_mat)
                    spatial_zones[current_mat] = current_zone
                data[current_resp].setdefault(current_mat, OrderedDict())
                i = next_i
                continue

            channel_match = channel_re.match(line)
            if channel_match and current_resp is not None:
                # --- isotope/channel table; fallback to profile when no zone exists
                if current_mat is None:
                    if current_zone is None:
                        current_zone = self._mcnp_profile_zone(current_profile or 1)
                    current_mat = current_zone.label
                    if current_mat not in materials:
                        materials.append(current_mat)
                        spatial_zones[current_mat] = current_zone
                    data[current_resp].setdefault(current_mat, OrderedDict())
                mcnp_zaid = channel_match.group(1)
                zaid = self._mcnp_zaid_to_pyndus(mcnp_zaid)
                ace_suffix = self._mcnp_ace_suffix(mcnp_zaid)
                temperature = self.mcnp_ace_temperatures.get(ace_suffix)
                instance = SensitivityNuclideInstance(zaid, ace_suffix,
                                                      temperature)
                channel = self._coerce_channel(channel_match.group(2))
                lower, upper, sens, rsd, next_i = self._read_mcnp_channel_rows(
                    lines, i + 1, row_re)
                if sens:
                    # --- register axes and check that all tables share one grid
                    if zaid not in zaids:
                        zaids.append(zaid)
                    if (zaid, ace_suffix) not in nuclide_instances:
                        nuclide_instances[(zaid, ace_suffix)] = instance
                    ace_suffixes.setdefault(zaid, [])
                    if ace_suffix not in ace_suffixes[zaid]:
                        ace_suffixes[zaid].append(ace_suffix)
                    if channel not in channels:
                        channels.append(channel)
                    current_grid = np.array([lower[0], *upper])
                    if group_structure is None:
                        group_structure = current_grid
                    elif not np.allclose(group_structure, current_grid):
                        raise SensitivityError(
                            "Inconsistent MCNP sensitivity energy grid.")
                    current_data = data[current_resp][current_mat].setdefault(
                        (zaid, ace_suffix), OrderedDict())
                    values = np.array(sens)
                    rel_unc = np.array(rsd)
                    if channel in current_data:
                        # --- repeated ZA/channel, possibly from another ACE suffix
                        values, rel_unc = self._combine_mcnp_channel_data(
                            current_data[channel], values, rel_unc)
                    current_data[channel] = (values, rel_unc)
                i = next_i
                continue

            i += 1

        if not found_section:
            raise SensitivityError(
                "No MCNP 'nuclear data ... sensitivity coefficients' section found."
            )
        if group_structure is None:
            raise SensitivityError(
                "No MCNP sensitivity coefficient rows found.")

        return self._build_mcnp_sensitivity_arrays(data, responses, materials,
                                                   spatial_zones, zaids,
                                                   nuclide_instances,
                                                   ace_suffixes, channels,
                                                   group_structure)

    @staticmethod
    def _read_mcnp_spatial_zone_entries(lines, start, channel_re, section_re,
                                        profile_re, spatial_re):
        """Read CELL/MAT entries belonging to one MCNP spatial zone."""
        entries = []
        i = start
        while i < len(lines):
            line = lines[i]
            if (channel_re.match(line) or section_re.match(line)
                    or profile_re.match(line) or spatial_re.match(line)):
                break
            entries.extend(
                int(value) for value in re.findall(r"\b\d+\b", line))
            i += 1
        return entries, i

    @staticmethod
    def _read_mcnp_channel_rows(lines, start, row_re):
        """Read one MCNP channel table after its isotope/channel header."""
        lower = []
        upper = []
        sens = []
        rsd = []
        started = False
        i = start
        while i < len(lines):
            row_match = row_re.match(lines[i])
            if row_match:
                started = True
                values = [float(row_match.group(j)) for j in range(1, 5)]
                lower.append(values[0])
                upper.append(values[1])
                sens.append(values[2])
                rsd.append(values[3])
                i += 1
                continue
            if started:
                break
            i += 1
        return lower, upper, sens, rsd, i

    @staticmethod
    def _combine_mcnp_channel_data(existing, values, rel_unc):
        """Combine repeated MCNP blocks belonging to the same sensitivity profile."""
        old_values, old_rel_unc = existing
        total = old_values + values
        old_abs_unc = np.abs(old_values) * old_rel_unc
        new_abs_unc = np.abs(values) * rel_unc
        total_abs_unc = np.sqrt(old_abs_unc**2 + new_abs_unc**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            total_rel_unc = np.where(total != 0, total_abs_unc / np.abs(total),
                                     0)
        return total, total_rel_unc

    @staticmethod
    def _build_mcnp_sensitivity_arrays(data, responses, materials,
                                       spatial_zones, zaids, nuclide_instances,
                                       ace_suffixes, channels,
                                       group_structure):
        """Build dense sensitivity/RSD arrays from sparse MCNP profile data."""
        nResp = len(responses)
        nMat = len(materials)
        nZaid = len(zaids)
        instance_keys = list(nuclide_instances.keys())
        nInstances = len(instance_keys)
        nChannels = len(channels)
        nE = len(group_structure) - 1
        sens = np.zeros((nResp, nMat, nZaid, nChannels, nE))
        sens_rsd = np.zeros_like(sens)
        sens_instances = np.zeros((nResp, nMat, nInstances, nChannels, nE))
        sens_instances_rsd = np.zeros_like(sens_instances)
        aggregate_set = np.zeros((nResp, nMat, nZaid, nChannels), dtype=bool)
        for iResp, resp in enumerate(responses):
            for iMat, mat in enumerate(materials):
                for iInstance, instance_key in enumerate(instance_keys):
                    zaid = instance_key[0]
                    iZaid = zaids.index(zaid)
                    for iChannel, channel in enumerate(channels):
                        try:
                            values, rsd = data[resp][mat][instance_key][channel]
                        except KeyError:
                            continue
                        sens_instances[iResp, iMat, iInstance,
                                       iChannel, :] = values
                        sens_instances_rsd[iResp, iMat, iInstance,
                                           iChannel, :] = rsd
                        if aggregate_set[iResp, iMat, iZaid, iChannel]:
                            values, rsd = Sensitivity._combine_mcnp_channel_data(
                                (sens[iResp, iMat, iZaid, iChannel, :],
                                 sens_rsd[iResp, iMat, iZaid, iChannel, :]),
                                values, rsd)
                        sens[iResp, iMat, iZaid, iChannel, :] = values
                        sens_rsd[iResp, iMat, iZaid, iChannel, :] = rsd
                        aggregate_set[iResp, iMat, iZaid, iChannel] = True

        if Sensitivity._should_collapse_bare_mcnp_profiles(materials, spatial_zones):
            materials, spatial_zones, sens, sens_rsd = Sensitivity._collapse_bare_mcnp_profiles(
                materials, spatial_zones, sens, sens_rsd)
            _, _, sens_instances, sens_instances_rsd = Sensitivity._collapse_bare_mcnp_profiles(
                materials, spatial_zones, sens_instances, sens_instances_rsd)

        return {
            "responses": responses,
            "materials": materials,
            "spatial_zones": spatial_zones,
            "zaids": zaids,
            "nuclide_instances": nuclide_instances,
            "ace_suffixes": OrderedDict((zaid, tuple(suffixes))
                                        for zaid, suffixes in ace_suffixes.items()),
            "channels": channels,
            "group_structure": group_structure,
            "sens": sens,
            "sens_rsd": sens_rsd,
            "sens_nuclide_instances": sens_instances,
            "sens_nuclide_instances_rsd": sens_instances_rsd,
        }

    @staticmethod
    def _should_collapse_bare_mcnp_profiles(materials, spatial_zones):
        """Return whether MCNP profiles carry no explicit cell/material zones."""
        if len(materials) <= 1:
            return False
        return all(material in spatial_zones and spatial_zones[material].kind
                   == "profile" and spatial_zones[material].zone is None
                   and len(spatial_zones[material].entries) == 0
                   for material in materials)

    @staticmethod
    def _collapse_bare_mcnp_profiles(materials, spatial_zones, values, rsd):
        """Sum bare MCNP profiles over the material axis."""
        total = values.sum(axis=1, keepdims=True)
        total_abs_std = np.sqrt(
            np.sum((np.abs(values) * rsd)**2, axis=1, keepdims=True))
        with np.errstate(divide="ignore", invalid="ignore"):
            total_rsd = np.where(total != 0.0, total_abs_std / np.abs(total),
                                 0.0)
        collapsed_zones = OrderedDict([("profile", SensitivitySpatialZone())])
        return ["profile"], collapsed_zones, total, total_rsd

    @staticmethod
    def NormalizeSensProfile(sens_profile, energy_vector):
        """
        Normalize a sensitivity profile in lethargy.

        Parameters
        ----------
        sens_profile : array-like
            Sensitivity profile (vector of G elements).
        energy_vector : array-like
            Energy vector of G+1 limits (not bins).

        Returns
        -------
        vector : list
            Sensitivity profile per unit lethargy.
        """
        sens_profile = np.asarray(sens_profile, dtype=float)
        energy_vector = np.asarray(energy_vector, dtype=float)

        if len(sens_profile) != (len(energy_vector) - 1):
            raise ValueError('Energy grid is not the same')
        elif np.any(energy_vector <= 0.0):
            raise ValueError('Energy grid values must be positive')

        delta_lethargy = np.abs(np.log(energy_vector[1:] / energy_vector[:-1]))
        return list(sens_profile / delta_lethargy)

    @property
    def filepath(self):
        """
        Path to the sensitivity file.

        Returns
        -------
        Path
            File path.
        """
        if hasattr(self, "_filepaths"):
            return self._filepaths
        return self._filepath

    @filepath.setter
    def filepath(self, value):
        """
        Set the file path(s) for the sensitivity file(s).

        Parameters
        ----------
        value : str, Path, or list of str/Path

        Raises
        ------
        ValueError
            If value is not a string or Path.
        SensitivityError
            If the file does not exist.
        """
        # ---- MULTI-FILE CASE ----
        if isinstance(value, (list, tuple)):
            paths = []
            for v in value:
                if isinstance(v, str):
                    v = Path(v)
                elif not isinstance(v, Path):
                    raise ValueError(
                        f"All elements must be str or Path, not {type(v)}")

                if not v.exists():
                    raise SensitivityError(
                        f"File {v} does not exist, cannot create Sensitivity object."
                    )

                paths.append(v)

            if len(paths) == 0:
                raise SensitivityError("Empty list of sensitivity files.")

            self._filepaths = paths

            # # keep first for backward compatibility ?
            # self._filepath = paths[0]

            return

        # ---- SINGLE FILE CASE ----
        if isinstance(value, str):
            value = Path(value)
        elif not isinstance(value, Path):
            raise ValueError(
                f"Input arg to class Sensitivity must be either str, Path or list, not {type(value)}"
            )

        if not value.exists():
            raise SensitivityError(
                f"File {value} does not exist, cannot create Sensitivity object."
            )

        self._filepath = value

        # remove multi-file attribute if switching mode
        if hasattr(self, "_filepaths"):
            del self._filepaths

    @property
    def is_multifile(self):
        """
        Whether this object was initialized from multiple sensitivity files.

        Returns
        -------
        bool
            ``True`` when ``filepath`` stores a sequence of paths.
        """
        return hasattr(self, "_filepaths")

    @property
    def reader(self):
        """
        Sensitivity file format ('serpent', 'eranos', or 'mcnp').

        Returns
        -------
        str
            File format.
        """
        return self._reader

    @reader.setter
    def reader(self, value):
        """
        Set the sensitivity file format.

        Parameters
        ----------
        value : str
            File format ('serpent', 'eranos', or 'mcnp').

        Raises
        ------
        ValueError
            If value is not a string.
        """
        if isinstance(value, str):
            self._reader = value
        elif not isinstance(value, str):
            raise ValueError(
                f"Attribute 'reader' to class Sensitivity must be a str, not of type {type(value)}"
            )

    def get_reader(self):
        """
        Determine the reader type based on the file extension.

        Returns
        -------
        str
            Reader type ('serpent', 'eranos', or 'mcnp').

        Raises
        ------
        SensitivityError
            If the file format is not recognized.
        """
        if self.filepath.suffix == ".m" and self.filepath.stem.endswith(
                "_sens0"):
            return "serpent"
        elif self.filepath.suffix in [".eranos33", ".eranos1968"]:
            return "eranos"
        elif self.filepath.suffix == ".mcnp":
            return "mcnp"
        elif self.filepath.suffix == ".txt" and self._is_mcnp_sensitivity_output():
            return "mcnp"
        else:
            raise SensitivityError(
                f"Cannot determine reader type from file extension {self.filepath.suffix}"
            )

    def _is_mcnp_sensitivity_output(self):
        """Return whether a text file contains MCNP sensitivity coefficients."""
        section_re = re.compile(
            r"nuclear data\s+.+?\s+sensitivity coefficients", re.IGNORECASE)
        with open(self.filepath, "r", errors="replace") as file:
            for line in file:
                if section_re.search(line):
                    return True
        return False

    @property
    def responses(self):
        """
        List of response parameters.

        Returns
        -------
        list
            List of responses.
        """
        return self._responses

    @responses.setter
    def responses(self, value):
        """
        Set the list of response parameters.

        Parameters
        ----------
        value : iterable of str
            List of response parameter names.

        Raises
        ------
        ValueError
            If value is not iterable or contains non-string elements.
        """
        if not isinstance(value, Iterable):
            raise ValueError(
                f"'responses' must be of type list, not of type {type(value)}")
        else:
            for iv, v in enumerate(value):
                if not isinstance(v, str):
                    raise ValueError(
                        f"Element n.{iv} of the responses list is not string!")

        self._responses = tuple(value)

    @property
    def materials(self):
        """
        Mapping of material names to their indices.

        Returns
        -------
        OrderedDict
            Material names and indices.
        """
        return self._materials

    @materials.setter
    def materials(self, value):
        """
        Set the mapping of material names to their indices.

        Parameters
        ----------
        value : iterable of str
            List of material names.

        Raises
        ------
        ValueError
            If value is not iterable or contains non-string elements.
        """
        out = OrderedDict()
        if not isinstance(value, Iterable):
            raise ValueError(
                f"'materials' must be of type list, not of type {type(value)}")
        else:
            for iv, v in enumerate(value):
                if not isinstance(v, str):
                    raise ValueError(
                        f"Element n.{iv} of the materials list is not string!")
                else:
                    out[v] = iv

        self._materials = out

    @property
    def zaid(self):
        """
        Mapping of ZAID numbers to their indices.

        Returns
        -------
        OrderedDict
            ZAID numbers and indices.
        """
        return self._zaid

    @zaid.setter
    def zaid(self, value):
        """
        Set the mapping of ZAID numbers to their indices.

        Parameters
        ----------
        value : iterable of int
            List of ZAID numbers.

        Raises
        ------
        ValueError
            If value is not iterable.
        """
        if not isinstance(value, Iterable):
            raise ValueError(
                f"Expected an Iterable instead of type {type(value)} for 'Sensitivity.zaid'"
            )

        self._zaid = OrderedDict()
        for iv, v in enumerate(value):
            self._zaid[v] = iv

    @property
    def zais(self):
        """
        Mapping of ZA strings to their indices.

        Returns
        -------
        OrderedDict
            ZA strings and indices.
        """
        return self._zais

    @zais.setter
    def zais(self, value):
        """
        Set the mapping of ZA strings to their indices.

        Parameters
        ----------
        value : iterable of int
            List of ZAID numbers to be converted to ZA strings.

        Raises
        ------
        ValueError
            If value is not iterable.
        """
        if not isinstance(value, Iterable):
            raise ValueError(
                f"Expected an Iterable instead of type {type(value)} for 'Sensitivity.zais'"
            )

        out = OrderedDict()
        for iv, v in enumerate(value):
            out[utils.zaid2zais(v)] = iv

        self._zais = out

    @property
    def channels(self):
        """
        Mapping of sensitivity channels to their axis indices.

        Returns
        -------
        OrderedDict
            :class:`SensitivityChannel` objects and indices.
        """
        if hasattr(self, "_channels"):
            return self._channels
        if hasattr(self, "_MTs"):
            return self._MTs
        raise ValueError("'channels' attribute is not defined.")

    @channels.setter
    def channels(self, value):
        if not isinstance(value, Iterable):
            raise ValueError(
                f"Expected an Iterable instead of type {type(value)} "
                "for 'Sensitivity.channels'")
        channels = OrderedDict()
        for i, channel in enumerate(value):
            channels[self._coerce_channel(channel)] = i
        self._channels = channels

    @property
    def MTs(self):
        """
        Compatibility alias for :attr:`channels`.

        Returns
        -------
        OrderedDict
            Sensitivity channels and indices.
        """
        return self.channels

    @MTs.setter
    def MTs(self, value):
        """
        Set the channel mapping from reader MT numbers or descriptors.

        Parameters
        ----------
        value : iterable
            List of MT numbers or descriptors.

        Raises
        ------
        ValueError
            If the reader type is unknown or input is invalid.
        """
        if self.reader == 'serpent':
            self._serpent_MTs(value)
        elif self.reader == 'eranos':
            self._eranos_MTs(value)
        else:
            raise ValueError(f"Unknown reader {self.reader}")

    @staticmethod
    def _channel_to_endf_dict(channel):
        return {
            "label": channel.name,
            "MF": channel.average_MF,
            "MT": channel.average_MT,
            "L": channel.L,
            "covariance_MF": channel.covariance_MF,
            "covariance_MTs": channel.covariance_MTs,
        }

    @staticmethod
    def _coerce_channel(value):
        if isinstance(value, SensitivityChannel):
            return value
        if isinstance(value, (int, np.integer)):
            mt = int(value)
            if mt in SERPENT_NUMERIC_ALIASES:
                return SensitivityChannel.from_alias(SERPENT_NUMERIC_ALIASES[mt])
            return SensitivityChannel.from_endf(average_MF=3, average_MT=mt)
        if isinstance(value, str):
            label = " ".join(value.split()).lower()
            if "xs" in label:
                mt = 1 if "total" in label else int(label.split()[1])
                return SensitivityChannel.from_endf(average_MF=3, average_MT=mt)
            try:
                return SensitivityChannel.from_alias(label)
            except ValueError:
                return SensitivityChannel(
                    average_MF=None, average_MT=None, covariance_MF=None,
                    covariance_MT=None, name=label)
        raise ValueError(
            "Sensitivity channels must be SensitivityChannel, int, or str, "
            f"not {type(value)}.")

    def _eranos_MTs(self, value):
        """
        Set the channel mapping for ERANOS files.

        Parameters
        ----------
        value : iterable of int
            List of MT numbers from ERANOS output.

        Raises
        ------
        ValueError
            If any MT is not an integer.
        """
        if not isinstance(value, Iterable):
            raise ValueError(
                f"Expected an Iterable instead of type {type(value)} for 'Sensitivity.MTs'"
            )
        else:
            value = sorted(value)

        channels = OrderedDict()
        for imt, mt in enumerate(value):
            if not isinstance(mt, int):
                raise ValueError(
                    f"Input MTs from ERANOS for attribute MTs must be of type int, not {type(mt)}!"
                )
            if mt == 452:
                channel = SensitivityChannel.from_alias("nubar total")
            else:
                channel = SensitivityChannel.from_endf(average_MF=3, average_MT=mt)
            channels[channel] = imt
        self._channels = channels

    def _serpent_MTs(self, value):
        """
        Set the channel mapping for Serpent files.

        Parameters
        ----------
        value : iterable
            List of MT descriptors from Serpent output.

        Raises
        ------
        ValueError
            If the input is not iterable or channels cannot be parsed.
        """
        if not isinstance(value, Iterable):
            raise ValueError(
                f"Expected an Iterable instead of type {type(value)} "
                "for 'Sensitivity.MTs'")

        channels = OrderedDict()

        for imt, mt in enumerate(value):
            if isinstance(mt, SensitivityChannel):
                channels[mt] = imt
                continue
            # numeric Serpent perturbations
            if isinstance(mt, (int, np.integer)):
                mt = int(mt)
                channel = self._coerce_channel(mt)
                channels[channel] = imt
                continue

            if not isinstance(mt, str):
                raise ValueError(
                    f"Serpent MT descriptors must be strings or integers, "
                    f"not {type(mt)}")

            # string Serpent perturbations
            label = " ".join(mt.split()).lower()
            channel = self._coerce_channel(label)
            channels[channel] = imt

        self._channels = channels

    @property
    def endf_channels(self):
        """
        Mapping from sensitivity keys to their ENDF quantity/covariance channel.

        Serpent uses descriptive perturbation names for quantities that do not
        have a unique ENDF MT. This metadata keeps the Serpent key separate
        from the ENDF covariance channel used by sandwich calculations.
        """
        return OrderedDict((channel, self._channel_to_endf_dict(channel))
                           for channel in self.channels)

    def get_covariance_sensitivity_keys(self, covariance_MF, covariance_MT):
        """
        Return sensitivity keys compatible with one ENDF covariance channel.

        Parameters
        ----------
        covariance_MF : int
            Covariance file number, e.g. 31, 33, 34, or 35.
        covariance_MT : int
            Covariance MT number.

        Returns
        -------
        list
            Sensitivity keys accepted by :meth:`get`.
        """
        keys = []
        for key, channel in self.endf_channels.items():
            if channel.get("covariance_MF") != covariance_MF:
                continue
            if covariance_MT in channel.get("covariance_MTs", ()):
                keys.append(key)
        return keys

    @property
    def energy_unit(self):
        """
        Unit used by the stored energy group structure.

        Returns
        -------
        str
            Canonical unit label, either ``"eV"`` or ``"MeV"``.
        """
        if hasattr(self, "_energy_unit"):
            return self._energy_unit

        if getattr(self, "reader", None) in {"serpent", "mcnp"}:
            return "MeV"
        elif getattr(self, "reader", None) == "eranos":
            return "eV"
        return "eV"

    @energy_unit.setter
    def energy_unit(self, value):
        """
        Set the unit used by the stored energy group structure.

        Parameters
        ----------
        value : str
            Energy unit. Supported values are ``"eV"`` and ``"MeV"``.
        """
        self._energy_unit = utils.normalize_energy_unit(value)

    @property
    def energy_grid(self):
        """
        Energy group structure with unit metadata.

        Returns
        -------
        utils.EnergyGrid
            Stored energy group boundaries and their unit.
        """
        if self.group_structure is None:
            raise ValueError(
                "Group structure is not defined, cannot build an EnergyGrid.")
        return utils.EnergyGrid(self.group_structure, self.energy_unit)

    def group_structure_as(self, unit):
        """
        Return the energy group structure converted to a requested unit.

        Parameters
        ----------
        unit : str
            Target energy unit. Supported values are ``"eV"`` and ``"MeV"``.

        Returns
        -------
        numpy.ndarray
            Energy group boundaries converted to ``unit``.
        """
        return self.energy_grid.to(unit)

    @property
    def group_structure_ev(self):
        """
        Energy group structure converted to eV.

        Returns
        -------
        numpy.ndarray
            Energy group boundaries in eV.
        """
        return self.group_structure_as("eV")

    @property
    def group_structure_mev(self):
        """
        Energy group structure converted to MeV.

        Returns
        -------
        numpy.ndarray
            Energy group boundaries in MeV.
        """
        return self.group_structure_as("MeV")

    @property
    def group_structure(self):
        """
        Energy group structure.

        Returns
        -------
        iterable
            Energy group boundaries.
        """
        return self._group_structure

    @group_structure.setter
    def group_structure(self, value):
        """
        Set the energy group structure.

        Parameters
        ----------
        value : iterable
            Energy group boundaries. Values are stored in ascending order.

        Raises
        ------
        ValueError
            If value is not iterable or is not monotonic.
        """
        if value is not None:
            if isinstance(value, utils.EnergyGrid):
                self.energy_unit = value.unit
                value = value.values.copy()

            if not isinstance(value, Iterable):
                raise ValueError(
                    f"Expected an iterable instead of type {type(value)} for arg 'group_structure'"
                )
            value = np.asarray(value, dtype=float)
            diff = np.diff(value)

            if value.ndim != 1:
                raise ValueError(
                    "Energy group structure must be one-dimensional.")
            elif len(value) > 1 and np.all(diff < 0):
                value = value[::-1]
            elif len(value) > 1 and not np.all(diff > 0):
                raise ValueError(
                    "Energy group structure must be strictly monotonic.")

        self._group_structure = value

    @property
    def n_groups(self):
        """
        Number of energy groups.

        Returns
        -------
        int
            Number of groups.
        """
        if self.group_structure is not None:
            return len(self.group_structure) - 1
        else:
            raise ValueError(
                "Group structure is not defined, cannot get number of groups.")

    @property
    def n_resp(self):
        """
        Number of responses.

        Returns
        -------
        int
            Number of responses.
        """
        if hasattr(self, 'responses'):
            return len(self.responses)
        else:
            raise ValueError(
                "'responses' attribute is not defined, cannot get number of responses 'n_resp'."
            )

    @property
    def n_mat(self):
        """
        Number of materials.

        Returns
        -------
        int
            Number of materials.
        """
        if hasattr(self, 'materials'):
            return len(self.materials)
        else:
            raise ValueError(
                "'materials' attribute is not defined, cannot get number of materials 'n_mat'."
            )

    @property
    def n_zai(self):
        """
        Number of isotopes.

        Returns
        -------
        int
            Number of isotopes.
        """
        if hasattr(self, 'zaid'):
            return len(self.zaid)
        else:
            raise ValueError(
                "'zaid' attribute is not defined, cannot get number of isotopes 'n_zai'."
            )

    @property
    def n_MTs(self):
        """
        Number of sensitivity channels.

        Returns
        -------
        int
            Number of sensitivity channels.
        """
        if hasattr(self, 'MTs'):
            return len(self.MTs)
        else:
            raise ValueError(
                "'channels' attribute is not defined, cannot get number of channels 'n_MTs'."
            )

    @property
    def sens(self):
        """
        Sensitivity profiles.

        Returns
        -------
        np.ndarray
            Sensitivity array whose shape is
            (nResp, nMat, nZaid, nChannels, nE).
        """
        return self._sens

    @sens.setter
    def sens(self, value):
        """
        Set the sensitivity profiles.

        Parameters
        ----------
        value : dict or np.ndarray
            Sensitivity data from Serpent or ERANOS.

        Raises
        ------
        ValueError
            If the reader type is unknown.
        """
        if self.reader == "serpent":
            self._sens_serpent(value)
        elif self.reader == "eranos":
            self._sens_eranos(value)
        else:
            raise ValueError(f"Unknown reader {self.reader}")

    @property
    def sens_nuclide_instances(self):
        """
        MCNP sensitivity profiles before aggregation over ACE suffixes.

        Returns
        -------
        np.ndarray
            Sensitivity array whose shape is
            (nResp, nMat, nNuclideInstances, nChannels, nE).
        """
        if not hasattr(self, "_sens_nuclide_instances"):
            raise ValueError(
                "'sens_nuclide_instances' is available only for MCNP readers.")
        return self._sens_nuclide_instances

    @property
    def sens_nuclide_instances_rsd(self):
        """
        MCNP sensitivity RSDs before aggregation over ACE suffixes.

        Returns
        -------
        np.ndarray
            Sensitivity RSD array whose shape is
            (nResp, nMat, nNuclideInstances, nChannels, nE).
        """
        if not hasattr(self, "_sens_nuclide_instances_rsd"):
            raise ValueError(
                "'sens_nuclide_instances_rsd' is available only for MCNP readers."
            )
        return self._sens_nuclide_instances_rsd

    def _sens_eranos(self, value):
        """
        Build the sensitivity array for ERANOS files.

        Parameters
        ----------
        value : dict
            Nested dictionary with sensitivity data from ERANOS. ERANOS tables
            are read in descending energy order and are reversed here before
            being stored.

        Sets
        ----
        self._sens : np.ndarray
            Sensitivity array with shape (nResp, nMat, nZaid, nChannels, nE).
        """
        nResp = len(self.responses)
        nMat = len(self.materials)
        nZaid = len(self.zaid.keys())
        nMTs = len(self.MTs.keys())
        nE = self.n_groups
        sens = np.zeros((nResp, nMat, nZaid, nMTs, nE))
        dummy = np.zeros((nE, ))
        for iResp, resp in enumerate(self.responses):
            for mat, iMat in self.materials.items():
                for zais, iZaid in self.zais.items():
                    isotope_values = value[resp][mat].get(zais)
                    for channel, iMT in self.MTs.items():
                        if isotope_values is None:
                            sens[iResp, iMat, iZaid, iMT, :] = dummy
                        elif isinstance(isotope_values, dict):
                            profile = isotope_values.get(channel)
                            if profile is None:
                                sens[iResp, iMat, iZaid, iMT, :] = dummy
                            else:
                                sens[iResp, iMat, iZaid, iMT, :] = profile[::-1]
                        else:
                            profile = isotope_values[iMT, ::-1]
                            sens[iResp, iMat, iZaid, iMT, :] = profile

        self._sens = sens

    def _sens_serpent(self, value):
        """
        Build the sensitivity array for Serpent files.

        Parameters
        ----------
        value : dict
            Dictionary with sensitivity data from Serpent.

        Sets
        ----
        self._sens : np.ndarray
            Sensitivity array with shape (nResp, nMat, nZaid, nChannels, nE).
        """
        arr_shape = None
        for k, v in value.items():
            if not isinstance(v, np.ndarray):
                raise ValueError(
                    f"The keys of the sensitivity dict must be np.array, not of type {type(v)}."
                    "The structure of serpentTools.parsers.sensitivity object might have changed!"
                )
            if arr_shape is None:
                arr_shape = v.shape
            elif arr_shape != v.shape:
                raise ValueError(
                    f"All sensitivity arrays must have the same shape, but found {arr_shape} and {v.shape} for {k}."
                    "The structure of serpentTools.parsers.sensitivity object might have changed!"
                )

        sens_avg = np.zeros((len(self.responses), *arr_shape[:-1]))
        for i, resp in enumerate(self.responses):
            if not isinstance(value[resp], np.ndarray):
                raise ValueError(
                    f"The keys of the sensitivity dict must be np.array, not of type {type(value[resp])}"
                )
            else:
                sens_avg[i, :, :, :, :] = value[resp][:, :, :, :, 0]

        self._sens = sens_avg

    @property
    def sens_rsd(self):
        """
        Relative standard deviations of the sensitivity profiles.

        Returns
        -------
        np.ndarray
            Sensitivity array whose shape is
            (nResp, nMat, nZaid, nChannels, nE).
        """
        return self._sens_rsd

    @sens_rsd.setter
    def sens_rsd(self, value):
        """
        Set the relative standard deviations of the sensitivity profiles.

        Parameters
        ----------
        value : dict or np.ndarray
            Sensitivity RSD data from Serpent or ERANOS.

        Raises
        ------
        ValueError
            If the reader type is unknown.
        """
        if self.reader == "serpent":
            self._sens_rsd_serpent(value)
        elif self.reader == "eranos":
            self._sens_rsd_eranos()
        else:
            raise ValueError(f"Unknown reader {self.reader}")

    def _sens_rsd_eranos(self):
        """
        Set the sensitivity relative standard deviations for ERANOS files.

        Sets
        ----
        self._sens_rsd : None
            ERANOS files do not provide RSDs.
        """
        self._sens_rsd = None

    def _sens_rsd_serpent(self, value):
        """
        Build the sensitivity RSD array for Serpent files.

        Parameters
        ----------
        value : dict
            Dictionary with sensitivity data from Serpent.

        Sets
        ----
        self._sens_rsd : np.ndarray
            Sensitivity RSD array with shape
            (nResp, nMat, nZaid, nChannels, nE).

        Raises
        ------
        ValueError
            If the sensitivity data is not a numpy array.
        """
        arr_shape = None
        for k, v in value.items():
            if not isinstance(v, np.ndarray):
                raise ValueError(
                    f"The keys of the sensitivity dict must be np.array, not of type {type(v)}."
                    "The structure of serpentTools.parsers.sensitivity object might have changed!"
                )
            if arr_shape is None:
                arr_shape = v.shape
            elif arr_shape != v.shape:
                raise ValueError(
                    f"All sensitivity arrays must have the same shape, but found {arr_shape} and {v.shape} for {k}."
                    "The structure of serpentTools.parsers.sensitivity object might have changed!"
                )

        sens_rsd = np.zeros((len(self.responses), *arr_shape[:-1]))
        for i, resp in enumerate(self.responses):
            if not isinstance(value[resp], np.ndarray):
                raise ValueError(
                    f"The keys of the sensitivity dict must be np.array, not of type {type(value[resp])}"
                )
            else:
                sens_rsd[i, :, :, :, :] = value[resp][:, :, :, :, 1]

        self._sens_rsd = sens_rsd

    def _select_channels(self, *, MT=None, channel=None, average_MF=None,
                         covariance_MF=None, covariance_MT=None, L=None):
        selected = list(self.channels.keys())

        if channel is not None:
            values = channel if isinstance(channel, list) else [channel]
            selected = [self._coerce_channel(value) for value in values]
            missing = [value for value in selected if value not in self.channels]
            if missing:
                raise ValueError(f"Channel(s) {missing} not available!")

        if MT is not None:
            values = MT if isinstance(MT, list) else [MT]
            selected_from_mt = []
            for value in values:
                if isinstance(value, SensitivityChannel):
                    matches = [value] if value in self.channels else []
                else:
                    matches = [
                        candidate for candidate in self.channels
                        if candidate.average_MT == value
                        or candidate.covariance_MT == value
                    ]
                if not matches:
                    raise ValueError(f"MT {value} not available!")
                if len(matches) > 1:
                    raise ValueError(
                        f"MT {value} is ambiguous; matching channels are "
                        f"{matches}. Use channel=... or specify MF/L.")
                selected_from_mt.extend(matches)
            selected = selected_from_mt

        filters = {
            "average_MF": average_MF,
            "covariance_MF": covariance_MF,
            "covariance_MT": covariance_MT,
            "L": L
        }
        for attr, value in filters.items():
            if value is None:
                continue
            if attr == "covariance_MT":
                selected = [
                    candidate for candidate in selected
                    if value in candidate.covariance_MTs
                ]
            else:
                selected = [
                    candidate for candidate in selected
                    if getattr(candidate, attr) == value
                ]

        if not selected:
            raise ValueError(
                "No sensitivity channels match the requested filters.")
        return selected

    def _resolve_eranos_capture_channel(self, channel, selected_zais):
        """Resolve ERANOS capture requests whose ENDF MT depends on isotope."""
        if self.reader != "eranos" or channel is None:
            return channel

        values = channel if isinstance(channel, list) else [channel]
        resolved = []
        changed = False
        for value in values:
            candidate = self._coerce_channel(value)
            if candidate == ERANOS_CHANNELS["CAPTURE"]:
                changed = True
                for zais in selected_zais:
                    eranos_capture = eranos_channel("CAPTURE", isotope=zais)
                    if eranos_capture not in resolved:
                        resolved.append(eranos_capture)
            elif candidate not in resolved:
                resolved.append(candidate)

        return resolved if changed else channel

    def get(self, resp=None, mat=None, MT=None, channel=None, average_MF=None,
            covariance_MF=None, covariance_MT=None, L=None, za=None,
            ace_suffix=None, temperature=None, g=None, group_order='ascending'):
        """
        Extract sensitivity profiles and uncertainties for specified parameters.

        Parameters
        ----------
        resp : str or list, optional
            Response(s) to extract, e.g., 'keff', 'beff'.
            If None, all responses are extracted.
        mat : str or list, optional
            Material(s) to extract, e.g., 'total', 'm1'.
            If None, all materials are extracted.
        MT : int or list, optional
            Average MT number(s) to extract. Accepted only when the MT maps to
            one unambiguous channel.
        channel : SensitivityChannel or list, optional
            Canonical channel(s) to extract.
        average_MF, covariance_MF, covariance_MT, L : int, optional
            ENDF-aware channel filters.
        za : str, int, or list, optional
            ZA string(s) or number(s) to extract, e.g., 'Pu-239', 942390 or ['Pu-239', 'Pu-240'].
            If None, all ZAIDs are extracted.
        ace_suffix : str or list, optional
            MCNP ACE suffix(es), e.g. ``".00c"`` or ``".02c"``. If provided,
            profiles are extracted before aggregation over ACE suffixes.
        temperature : int, float, str, or list, optional
            MCNP ACE temperature(s) provided through ``mcnp_ace_temperatures``.
            If provided, profiles are extracted before aggregation over ACE
            suffixes. Numeric values are compared with ``numpy.isclose``.
        g : int or list, optional
            Energy group(s) to extract, e.g., 1, 2, ..., n_groups.
            If None, all groups are extracted.
        group_order : {"ascending", "descending"}, optional
            Order used for the returned energy groups. ``"ascending"``
            preserves the internal ordering; ``"descending"`` reverses it.

        Returns
        -------
        S_avg : np.ndarray or dict
            Sensitivity profile(s).
        S_rsd : np.ndarray or dict, optional
            Sensitivity RSD(s), if available.
        """
        allowed_group_orders = {"ascending", "descending"}
        if group_order not in allowed_group_orders:
            raise ValueError(
                f"'group_order' must be one of {sorted(allowed_group_orders)}, "
                f"not {group_order!r}")

        # get indexes
        iM = []
        iZ = []
        iP = []
        iG = []

        # --- get material indexes
        if mat is not None:
            if isinstance(mat, str):
                if mat not in self.materials:
                    raise ValueError(f"Material {mat} not available!")
                else:
                    iM.append(self.materials[mat])
            elif not isinstance(mat, list):
                raise ValueError(
                    f"'mat' should be of type str or list, not of type {type(mat)}"
                )
            else:
                for val in mat:
                    iM.append(self.materials[val])
        else:
            for val in self.materials:
                iM.append(self.materials[val])

        # --- get ZA indexes
        if za is not None:
            if isinstance(za, str):
                if za not in self.zais:
                    raise ValueError(f"'za' {za} not available!")
                else:
                    iZ = [self.zais[za]]
            elif isinstance(za, int):
                if za not in self.zaid:
                    raise ValueError(f"'za' {za} not available!")
                else:
                    iZ = [self.zaid[za]]
            elif not isinstance(za, list):
                raise ValueError(
                    f"'za' should be of type str or int, not of type {type(za)}"
                )
            else:
                for val in za:
                    if isinstance(val, str):
                        iZ.append(self.zais[val])
                    elif isinstance(val, int):
                        iZ.append(self.zaid[val])
                    else:
                        raise ValueError(
                            f"Elements in list 'za' should be of type str or int, not of type {type(za)}"
                        )
        else:
            for k, v in self.zaid.items():
                iZ.append(v)

        # --- get channel indexes
        selected_zais = [list(self.zais.keys())[idx] for idx in iZ]
        effective_channel = self._resolve_eranos_capture_channel(
            channel, selected_zais)
        for selected_channel in self._select_channels(
                MT=MT, channel=effective_channel, average_MF=average_MF,
                covariance_MF=covariance_MF, covariance_MT=covariance_MT, L=L):
            iP.append(self.channels[selected_channel])

        use_nuclide_instances = ace_suffix is not None or temperature is not None
        selected_zaids = [list(self.zaid.keys())[idx] for idx in iZ]
        if use_nuclide_instances:
            if self.reader != "mcnp":
                raise ValueError(
                    "'ace_suffix' and 'temperature' can be used only with MCNP sensitivities."
                )
            if ace_suffix is None:
                suffixes = None
            else:
                suffixes = ace_suffix if isinstance(ace_suffix,
                                                    list) else [ace_suffix]
                suffixes = [
                    self._normalize_mcnp_ace_suffix(suffix)
                    for suffix in suffixes
                ]
            temperatures = (None if temperature is None else temperature if
                            isinstance(temperature, list) else [temperature])
            instance_keys = list(self.nuclide_instances.keys())
            iZ = [
                i for i, (zaid, suffix) in enumerate(instance_keys)
                if zaid in selected_zaids and
                (suffixes is None or suffix in suffixes) and (
                    temperatures is None or any(
                        self._mcnp_temperature_matches(
                            self.nuclide_instances[(zaid,
                                                    suffix)].temperature, temp)
                        for temp in temperatures))
            ]
            if not iZ:
                available_temperatures = self._format_mcnp_available_temperatures(
                    self.nuclide_instances)
                raise ValueError(
                    f"No MCNP nuclide instance matches za={za!r}, "
                    f"ace_suffix={ace_suffix!r}, temperature={temperature!r}. "
                    f"Available temperatures: {available_temperatures}.")
        elif (getattr(self, "_reader", None) == "mcnp"
              and getattr(self, "mcnp_ace_aggregation", "raise") == "raise"):
            ambiguous = OrderedDict(
                (zaid, self.ace_suffixes[zaid]) for zaid in selected_zaids
                if len(self.ace_suffixes.get(zaid, ())) > 1)
            if ambiguous:
                details = ", ".join(f"{zaid}: {suffixes}"
                                    for zaid, suffixes in ambiguous.items())
                raise ValueError(
                    "MCNP sensitivity data contain multiple ACE suffixes for "
                    f"the selected ZAID(s): {details}. Select one profile with "
                    "'ace_suffix' or 'temperature', or instantiate Sensitivity "
                    "with mcnp_ace_aggregation='sum' if summing the suffix "
                    "contributions is the intended coherent perturbation.")

        # --- get resp
        if resp is not None:
            if isinstance(resp, str):
                if resp not in self.responses:
                    raise ValueError(f"Response {resp} not available!")
                else:
                    iR = [self.responses.index(resp)]
            elif not isinstance(resp, list):
                raise ValueError(
                    f"'resp' should be of type str or list, not of type {type(resp)}"
                )
            else:
                iR = []
                for r in resp:
                    if r not in self.responses:
                        raise ValueError(f"Response {resp} not available!")
                    else:
                        iR.append(self.responses.index(r))
        else:
            resp = self.responses
            iR = [ir for ir in range(len(self.responses))]

        # --- get group indexes
        def group_number_to_index(group_number):
            if not isinstance(group_number, int):
                raise ValueError(
                    f"Elements in list 'g' should be of type int, not of type {type(group_number)}"
                )
            elif group_number < 1 or group_number > self.n_groups:
                raise ValueError(f"Energy group {group_number} not available!")

            if group_order == 'ascending':
                return group_number - 1
            else:
                return self.n_groups - group_number

        if g is not None:
            if isinstance(g, int):
                iG = [group_number_to_index(g)]
            elif not isinstance(g, list):
                raise ValueError(
                    f"'g' must be of type int or list, not of type {type(g)}")
            else:
                for val in g:
                    iG.append(group_number_to_index(val))
        else:
            iG = [ig for ig in range(len(self.group_structure) - 1)]
            if group_order == 'descending':
                iG = iG[::-1]

        # --- get sensitivity vector and uncertainty
        sens_array = self.sens_nuclide_instances if use_nuclide_instances else self.sens
        sens_rsd_array = (self.sens_nuclide_instances_rsd
                          if use_nuclide_instances else self.sens_rsd)
        S_avg = sens_array[np.ix_(iR, iM, iZ, iP, iG)]
        if sens_rsd_array is not None:
            S_rsd = sens_rsd_array[np.ix_(iR, iM, iZ, iP, iG)]
            return S_avg, S_rsd
        else:
            return S_avg

    def collapse(self, fewgrp, weight=None, egridname=None):
        """
        Collapse sensitivity coefficients onto a coarser energy grid.

        Parameters
        ----------
        fewgrp : iterable
            Few-group boundaries used for the collapse.
        weight : array-like, optional
            Weighting function used for the energy collapse. If ``None``, a
            unit weighting vector is used.
        egridname : str, optional
            Name of the energy grid, by default ``None``.

        Raises
        ------
        SensitivityError
            If the target grid is incompatible with the current energy grid or
            the weighting vector length is inconsistent.

        Returns
        -------
        None
            The object is updated in place.
        """
        if weight is None:
            weight = np.ones((self.n_groups))

        if len(weight) != self.n_groups:
            raise SensitivityError(
                f"len of the weighting function, {len(weight)} not consistent with the number of "
                f"groups in sensitivities {self.n_groups}")

        multigrp = self.group_structure

        if isinstance(fewgrp, utils.EnergyGrid):
            fewgrp = fewgrp.to(self.energy_unit)

        if isinstance(fewgrp, list):
            fewgrp = np.asarray(fewgrp)

        # ensure ascending order
        fewgrp = np.asarray(sorted(fewgrp), dtype=float)
        H = len(multigrp) - 1
        G = len(fewgrp) - 1
        # sanity checks
        if G >= H:
            raise SensitivityError(
                f'Collapsing failed: few-group structure should ',
                f' have less than {H} group')
        if multigrp[0] != fewgrp[0] or multigrp[-1] != fewgrp[-1]:
            raise SensitivityError(
                'Collapsing failed: few-group structure '
                'boundaries do not match with multi-group one:'
                f'multi_group lower bound: {multigrp[0]}, few_group lower bound: {fewgrp[0]} ',
                f'multi_group lower bound: {multigrp[-1]}, few_group lower bound: {fewgrp[-1]}'
            )
        # map fewgroup onto multigroup
        few_into_multigrp = np.zeros((G + 1, ), dtype=int)
        # multigrp_bin = np.zeros((H+1,), dtype=int)
        for ig, g in enumerate(fewgrp):
            reldiff = abs(multigrp - g) / g
            idx = np.argmin(reldiff)
            if (reldiff[idx] > 1E-5):
                raise SensitivityError(
                    f'Group boundary n.{ig}, {g} {self.energy_unit} not present in fine grid!'
                )
            else:
                few_into_multigrp[ig] = idx

        dims = (self.n_resp, self.n_mat, self.n_zai, self.n_MTs,
                len(fewgrp) - 1)

        collapsed_sens = np.zeros(dims)
        if self.sens_rsd is not None:
            collapsed_sens_rsd = np.zeros(dims)

        for ig in range(G):
            # select fine groups in g
            G1, G2 = fewgrp[ig], fewgrp[ig + 1]
            iS = few_into_multigrp[ig]
            iE = few_into_multigrp[ig + 1]
            # --- collapse
            for iresp in range(self.n_resp):
                for imat in range(self.n_mat):
                    for izai in range(self.n_zai):
                        for iMT in range(self.n_MTs):
                            if self.sens_rsd is None:
                                sensitivity = self.sens[iresp, imat, izai,
                                                        iMT, :]
                                collapsed_values = weight[iS:iE].dot(
                                    sensitivity[iS:iE])
                                collapsed_sens[iresp, imat, izai, iMT,
                                               ig] = collapsed_values
                            else:
                                S_avg = self.sens[iresp, imat, izai, iMT, :]
                                S_rsd = self.sens_rsd[iresp, imat, izai,
                                                      iMT, :]
                                sensitivity = utils.np2unp(S_avg, 2 * S_rsd)
                                collapsed_values = weight[iS:iE].dot(
                                    sensitivity[iS:iE])
                                collapsed_sens[iresp, imat, izai, iMT,
                                               ig] = collapsed_values.n
                                collapsed_sens_rsd[
                                    iresp, imat, izai, iMT,
                                    ig] = collapsed_values.s / collapsed_values.n if collapsed_values.n != 0 else 0.0

        # update attributes data
        self._sens = collapsed_sens
        if self.sens_rsd is not None:
            self._sens_rsd = collapsed_sens_rsd

        self.fine_energygrid = self.group_structure
        self.fine_energygrid_unit = self.energy_unit
        self.group_structure = fewgrp
        self.egridname = egridname if egridname else f'{G}G'

    # @staticmethod
    # def _normalize_paths(sensitivity_path):
    #     if isinstance(sensitivity_path, (str, Path)):
    #         return [Path(sensitivity_path)]
    #     elif isinstance(sensitivity_path, Iterable):
    #         paths = [Path(p) if isinstance(p, str) else p for p in sensitivity_path]
    #         if len(paths) == 0:
    #             raise SensitivityError("Empty list of sensitivity files.")
    #         return paths
    #     else:
    #         raise ValueError("Invalid sensitivity path provided.")

    def _from_multiple(self, paths, duplicate_policy="raise"):
        """
        Read and merge multiple sensitivity files into this object.

        Parameters
        ----------
        paths : sequence of pathlib.Path
            Sensitivity files to read and merge. Currently all files must be
            Serpent ``*_sens0.m`` files.
        duplicate_policy : {"raise", "keep_first", "keep_last"}, optional
            Policy used when the same response/material/ZA/MT profile appears
            in more than one input file.

        Raises
        ------
        SensitivityError
            If readers differ, non-Serpent files are provided, or energy grids
            are inconsistent.
        """
        objs = [Sensitivity(p) for p in paths]

        readers = {obj.reader for obj in objs}
        if len(readers) != 1:
            raise SensitivityError(
                "Cannot merge sensitivity files from different readers.")

        if readers != {"serpent"}:
            raise SensitivityError(
                "Merging multiple files is currently supported only for Serpent _sens0.m files."
            )

        self.reader = "serpent"

        self._check_same_energy_grid(objs)

        self._merge_serpent_sensitivities(objs,
                                          duplicate_policy=duplicate_policy)

    @staticmethod
    def _check_same_energy_grid(objs, rtol=0.0, atol=0.0):
        """
        Validate that all sensitivity objects share the same energy grid.

        Parameters
        ----------
        objs : sequence of Sensitivity
            Sensitivity objects to compare.
        rtol : float, optional
            Relative tolerance passed to ``numpy.allclose``.
        atol : float, optional
            Absolute tolerance passed to ``numpy.allclose``.

        Raises
        ------
        SensitivityError
            If any object has an energy grid with a different shape or values.
        """
        ref = objs[0].group_structure
        ref_unit = objs[0].energy_unit
        for i, obj in enumerate(objs[1:], start=1):
            if obj.energy_unit != ref_unit:
                raise SensitivityError(
                    f"Inconsistent energy unit between {objs[0].filepath} and {obj.filepath}."
                )

            if ref.shape != obj.group_structure.shape or not np.allclose(
                    ref, obj.group_structure, rtol=rtol, atol=atol):
                raise SensitivityError(
                    f"Inconsistent energy grid between {objs[0].filepath} and {obj.filepath}."
                )

    def _merge_serpent_sensitivities(self, objs, duplicate_policy="raise"):
        """
        Merge multiple Serpent Sensitivity objects into one Sensitivity object.

        Parameters
        ----------
        objs : list[Sensitivity]
            Already parsed Sensitivity objects coming from Serpent _sens0.m files.
        duplicate_policy : str, optional
            Policy for duplicate entries:
            - "raise"
            - "keep_first"
            - "keep_last"

        Raises
        ------
        SensitivityError
            If readers differ, energy grids differ, or duplicates are found
            and duplicate_policy="raise".
        ValueError
            If duplicate_policy is invalid.
        """
        if len(objs) == 0:
            raise SensitivityError(
                "Cannot merge an empty list of Sensitivity objects.")

        allowed_policies = {"raise", "keep_first", "keep_last"}
        if duplicate_policy not in allowed_policies:
            raise ValueError(f"Invalid duplicate_policy: {duplicate_policy}. "
                             f"Allowed values are {sorted(allowed_policies)}.")

        # reader consistency
        readers = {obj.reader for obj in objs}
        if readers != {"serpent"}:
            raise SensitivityError(
                "Can only merge Sensitivity objects read from Serpent _sens0.m files."
            )

        # energy-grid consistency
        self._check_same_energy_grid(objs)

        # ---- union of metadata ----
        responses = []
        materials = []
        zaids = []
        mts = []

        for obj in objs:
            for resp in obj.responses:
                if resp not in responses:
                    responses.append(resp)

            for mat in obj.materials.keys():
                if mat not in materials:
                    materials.append(mat)

            for za in obj.zaid.keys():
                if za not in zaids:
                    zaids.append(za)

            for mt in obj.MTs.keys():
                if mt not in mts:
                    mts.append(mt)

        ref_grid = np.asarray(objs[0].group_structure)
        # ---- assign merged metadata using existing setters ----
        self.reader = "serpent"
        self.energy_unit = objs[0].energy_unit
        self.responses = responses
        self.materials = materials
        self.zaid = zaids
        self.zais = self.zaid.keys()
        self.MTs = mts
        self.group_structure = ref_grid.copy()

        nResp = len(self.responses)
        nMat = len(self.materials)
        nZaid = len(self.zaid)
        nMTs = len(self.MTs)
        nE = self.n_groups

        # detect whether any object has rsd
        rsd_availability = [obj.sens_rsd is not None for obj in objs]

        if any(rsd_availability) and not all(rsd_availability):
            raise SensitivityError(
                "Cannot merge Serpent sensitivity objects with inconsistent "
                "RSD availability.")

        has_rsd = all(rsd_availability)

        merged_avg = np.zeros((nResp, nMat, nZaid, nMTs, nE))
        merged_rsd = np.zeros((nResp, nMat, nZaid, nMTs, nE)) if has_rsd else None

        filled = {}

        for obj_idx, obj in enumerate(objs):
            for resp in obj.responses:
                for mat in obj.materials:
                    for za in obj.zaid:
                        for mt in obj.MTs:
                            try:
                                out = obj.get(resp=[resp], mat=[mat], MT=[mt],
                                              za=[za], group_order="ascending")
                            except (KeyError, ValueError) as exc:
                                raise SensitivityError(
                                    "Could not retrieve sensitivity profile "
                                    f"(response={resp}, material={mat}, "
                                    f"ZA={za}, MT={mt}) from {obj.filepath}."
                                ) from exc

                            if has_rsd:
                                s_avg, s_rsd = out
                            else:
                                s_avg = out
                                s_rsd = None

                            key = (resp, mat, za, mt)

                            if key in filled:
                                if duplicate_policy == "raise":
                                    raise SensitivityError(
                                        "Duplicate sensitivity profile found for "
                                        f"(response={resp}, material={mat}, "
                                        f"ZA={za}, MT={mt}). "
                                        f"First source: {objs[filled[key]].filepath}; "
                                        f"second source: {obj.filepath}.")

                                if duplicate_policy == "keep_first":
                                    continue

                                # keep_last: proceed and overwrite

                            iR = self.responses.index(resp)
                            iM = self.materials[mat]
                            iZ = self.zaid[za]
                            iP = self.MTs[mt]

                            avg = np.asarray(s_avg).reshape(-1)

                            if avg.size != nE:
                                raise SensitivityError(
                                    f"Sensitivity profile {key} contains {avg.size} "
                                    f"groups, expected {nE}.")

                            merged_avg[iR, iM, iZ, iP, :] = avg

                            if has_rsd:
                                rsd = np.asarray(s_rsd).reshape(-1)

                                if rsd.size != nE:
                                    raise SensitivityError(
                                        f"RSD profile {key} contains {rsd.size} "
                                        f"groups, expected {nE}.")

                                merged_rsd[iR, iM, iZ, iP, :] = rsd

                            filled[key] = obj_idx

        self._sens = merged_avg
        self._sens_rsd = merged_rsd


class SensitivityError(Exception):
    """Custom exception raised for invalid sensitivity input or processing."""

    pass


def _expose_public_aliases():
    """Expose class symbols on the parent package for clean from-imports."""
    import sys as _sys

    parent = _sys.modules.get(__package__)
    if parent is not None:
        parent.Sensitivity = Sensitivity
        parent.SensitivityError = SensitivityError
        parent.SensitivityChannel = SensitivityChannel


_expose_public_aliases()
