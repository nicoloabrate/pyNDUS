"""
author: N. Abrate.

file: Getensitivity.py

description: read sensitivity profiles and construct an object to perform operations.
"""
import re
import numpy as np
import serpentTools as st
from pathlib import Path
from collections import OrderedDict
from collections.abc import Iterable
try:
    import pyNDUS.utils as utils
except ModuleNotFoundError: # when run as a script
    try:
        from . import utils
    except ImportError:
        import utils

class Sensitivity:
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
    reader : str
        Sensitivity file format ('serpent' or 'eranos').
    responses : list
        List of response parameters (e.g., 'keff', 'beff', etc.).
    materials : OrderedDict
        Mapping of material names to their indices.
    zaid : OrderedDict
        Mapping of ZAID numbers (e.g., 942390) to their indices.
    zais : OrderedDict
        Mapping of ZA strings (e.g., 'Pu-239') to their indices.
    MTs : OrderedDict
        Mapping of MT numbers (see ENDF-6 format for the numbers) to their indices.
    group_structure : iterable
        Energy group structure.
    n_groups : int
        Number of energy groups.
    sens : np.ndarray
        Sensitivity profiles whose shape is (nResp, nMat, nZaid, nMTs, nE).
    sens_rsd : np.ndarray
        Relative standard deviations of the sensivity profile. When not available
        (e.g., for deterministic calculations), it is None.
        The shape is (nResp, nMat, nZaid, nMTs, nE).

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
    get(resp=None, mat=None, MT=None, za=None, g=None)
        Extract sensitivity profiles and uncertainties for specified parameters.
        -'resp': response(s) to extract (e.g., 'keff', 'beff').
        -'mat': material(s) to extract (e.g., 'total', 'm1').
        -'MT': MT number(s) to extract (e.g., 2, 4, 18).
        -'za': ZA string(s) or number(s) to extract (e.g., 'Pu-239', 942390).
        -'g': energy group(s) to extract (e.g., 1, 2, ..., n_groups).

    Raises
    ------
    SensitivityError
        If the file format is not recognized or if the file structure is not as expected.
    """
    def __init__(self, sensitivity_path):
        """
        Initialize the Sensitivity object and read the sensitivity file.

        Parameters
        ----------
        sensitivity_path : str or Path
            Path to the sensitivity file.
            -'serpent': should end with "_sens0.m".
            -'eranos': should end with ".eranos33" or ".eranos1968".
        """
        # --- validate and assign path
        self.filepath = sensitivity_path

        # --- read the sensitivity file
        if "_sens0.m" in self.filepath.name:
            self.reader = "serpent"
        elif ".eranos" in self.filepath.name:
            self.reader = "eranos"
        else:
            raise SensitivityError(f"Cannot read file {self.filepath} since it is not produced by Serpent!")

        if self.reader == "serpent":
            self.from_serpent()
        elif self.reader == "eranos":
            self.from_eranos()
        else:
            raise SensitivityError(f"Sensitivity reader for {reader} not available!")

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
            raise SensitivityError("The structure of serpentTools.parsers.sensitivity object might have changed!")

        try:
            self.materials = list(sens.materials.keys())
        except ValueError:
            raise SensitivityError("The structure of serpentTools.parsers.sensitivity object might have changed!")

        if not isinstance(sens.zais, OrderedDict):
            raise SensitivityError(f"serpentTools.parsers.sensitivity.zais must be an OrderedDict, not of type {type(sens.zais)}."
                                    "The structure of serpentTools.parsers.sensitivity object might have changed!")
        else:
            # sens.zais contains integers, not strings
            self.zaid = sens.zais.keys()

        self.zais = self.zaid.keys()

        if not isinstance(sens.perts, OrderedDict):
            raise SensitivityError(f"serpentTools.parsers.sensitivity.perts must be an OrderedDict, not of type {type(sens.perts)}."
                                    "The structure of serpentTools.parsers.sensitivity object might have changed!")
        else:
            self.MTs = list(sens.perts.keys())

        if not isinstance(sens.sensitivities, dict):
            raise SensitivityError(f"serpentTools.parsers.sensitivity.sensitivities must be a dict, not of type {type(sens.sensitivities)}."
                                    "The structure of serpentTools.parsers.sensitivity object might have changed!")
        else:
            self.sens = sens.sensitivities
            self.sens_rsd = sens.sensitivities

        if not isinstance(sens.energies, np.ndarray):
            raise SensitivityError(f"serpentTools.parsers.sensitivity.energies must be a np.array, not of type {type(sens.energies)}."
                                    "The structure of serpentTools.parsers.sensitivity object might have changed!")
        else:
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
            raise SensitivityError(f"Cannot read file format {self.filepath.name}")

        nE = len(energy_grid) - 1
        header_found = False # Flag to activate search of integral parameter
        material_found = True # Flag to reject repeated table of results for material
        total_isotopes_found = True # Flag to reject last review
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
        perturbationsapp = perturbations.append

        header_pat = "SENSITIVITY COEFFICIENTS"
        param_pat = re.compile(r'\b(\w+)\s+SENSITIVITY\b')
        mat_pat = re.compile(r'\bDOMAIN\s+(\w+)')
        iso_pat = re.compile(r'\bISOTOPE\s+(\w+)')
        data_pat = re.compile(r'\bGROUP\s+(\w+)')

        str2MT = OrderedDict({"CAPTURE": 102, "FISSION": 18, "ELASTIC": 2,
                             "INELASTIC": 4, "N,XN": 106, "NU": 452})
        MT_str = list(str2MT.keys())
        MT_int = OrderedDict(zip(str2MT.values(), [i for i in range(len(MT_str))]))
                # idx_MT_sorted = [2, 3, 1, 0, 4, 5]
        MT_sorted = tuple(sorted(MT_int.keys()))
        idx_MT_sorted = []
        for mt in MT_sorted:
            idx_MT_sorted.append(MT_int[mt])

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

                    if columns is None:
                        num_column = len(perturbations) # len(perturbations) + 2 - (1 if data_line else 0)
                        columns = -np.ones((num_column, nE))
                        if perturbations != MT_str:
                            raise SensitivityError("The MT order does not match."
                                                   "The structure of the ERANOS output file might have changed!")
                    iline_data = iline + 1
                    ig = 0

                if iline >= iline_data and iline < iline_data + nE and iline_data > 0:
                    line = line.strip()
                    if not line:
                        break  # No more data
                    parts = line.split()
                    parts = parts[1:num_column + 1]
                    if len(parts) != num_column:
                        raise ValueError("Inconsistency between N of titles and N of columns")
                        continue
                    file_values = np.array([float(val) for val in parts])
                    columns[:, ig] = file_values[idx_MT_sorted]
                    ig += 1
                
                if ig == nE:
                    dict_read[resp][mat_element][s] = columns.copy()
                    ig = 0

        try:
            self.responses = responses
            self.group_structure = energy_grid
        except ValueError:
            raise SensitivityError("The structure of the ERANOS output file might have changed!")

        # --- get materials, isotopes and reactions
        perturbations = [2, 4, 18, 102, 106, 452]
        try:
            self.materials = list(materials)
            self.zaid = [utils.zais2zaid(za) for za in zais]
            self.zais = self.zaid.keys()
            self.MTs = perturbations
        except ValueError:
            raise SensitivityError("The structure of the ERANOS output file might have changed!")

        self.sens_rsd = None
        self.sens = dict_read

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
        if len(sens_profile) != (len(energy_vector)-1):
            ValueError ('Energy grid is not the same')
        vector = []
        for i in range(0,len(sens_profile)):
            vector.append(sens_profile[i]/np.log(energy_vector[i]/energy_vector[i+1]))
        return vector

    @property
    def filepath(self):
        """
        Path to the sensitivity file.

        Returns
        -------
        Path
            File path.
        """
        return self._filepath

    @filepath.setter
    def filepath(self, value):
        """
        Set the file path for the sensitivity file.

        Parameters
        ----------
        value : str or Path
            Path to the sensitivity file.

        Raises
        ------
        ValueError
            If value is not a string or Path.
        SensitivityError
            If the file does not exist.
        """
        if isinstance(value, str):
            value = Path(value)
        elif not isinstance(value, Path):
            raise ValueError(f"Input arg to class Sensitivity must be either str or Path object, not of type {type(value)}")

        if not value.exists():
            raise SensitivityError(f"File {value} does not exist, cannot create Sensitivity object.")

        self._filepath = value

    @property
    def reader(self):
        """
        Sensitivity file format ('serpent' or 'eranos').

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
            File format ('serpent' or 'eranos').

        Raises
        ------
        ValueError
            If value is not a string.
        """
        if isinstance(value, str):
            self._reader = value
        elif not isinstance(value, str):
            raise ValueError(f"Attribute 'reader' to class Sensitivity must be a str, not of type {type(value)}")

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
            raise ValueError(f"'responses' must be of type list, not of type {type(value)}")
        else:
            for iv, v in enumerate(value):
                if not isinstance(v, str):
                    raise ValueError(f"Element n.{iv} of the responses list is not string!")

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
            raise ValueError(f"'materials' must be of type list, not of type {type(value)}")
        else:
            for iv, v in enumerate(value):
                if not isinstance(v, str):
                    raise ValueError(f"Element n.{iv} of the materials list is not string!")
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
            raise ValueError(f"Expected an Iterable instead of type {type(value)} for 'Sensitivity.zaid'")

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
            raise ValueError(f"Expected an Iterable instead of type {type(value)} for 'Sensitivity.zais'")

        out = OrderedDict()
        for iv, v in enumerate(value):
            out[utils.zaid2zais(v)] = iv

        self._zais = out

    @property
    def MTs(self):
        """
        Mapping of MT numbers to their indices.

        Returns
        -------
        OrderedDict
            MT numbers and indices.
        """
        return self._MTs

    @MTs.setter
    def MTs(self, value):
        """
        Set the mapping of MT numbers to their indices.

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

    def _eranos_MTs(self, value):
        """
        Set the MTs attribute for ERANOS files.

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
            raise ValueError(f"Expected an Iterable instead of type {type(value)} for 'Sensitivity.MTs'")
        else:
            value = sorted(value)

        MTs = OrderedDict()
        for imt, mt in enumerate(value):
            if not isinstance(mt, int):
                raise ValueError(f"Input MTs from ERANOS for attribute MTs must be of type int, not {type(mt)}!")
            MTs[mt] = imt
        self._MTs = MTs

    def _serpent_MTs(self, value):
        """
        Set the MTs attribute for Serpent files.

        Parameters
        ----------
        value : iterable
            List of MT descriptors from Serpent output.

        Raises
        ------
        ValueError
            If the input is not iterable or MTs cannot be parsed.
        """
        if not isinstance(value, Iterable):
            raise ValueError(f"Expected an Iterable instead of type {type(value)} for 'Sensitivity.MTs'")

        MTs = OrderedDict()
        for imt, mt in enumerate(value):
            if "xs" in mt:
                if "total" not in mt:
                    MTs[int(mt.split(" ")[1])] = imt
                else: # due to Serpent definition for MT1
                    MTs[1] = imt
            elif "nubar" in mt:
                if mt == 'nubar total':
                    MTs[452] = imt
                elif mt == 'nubar delayed':
                    MTs[455] = imt
                elif mt == 'nubar prompt':
                    MTs[456] = imt
            else:
                # TODO FIXME define an MT number for fission emission spectra
                MTs[mt] = imt

        self._MTs = MTs

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
            Energy group boundaries.

        Raises
        ------
        ValueError
            If value is not iterable.
        """
        if value is not None:
            if not isinstance(value, Iterable):
                raise ValueError(f"Expected an iterable instead of type {type(value)} for arg 'group_structure'")

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
            raise ValueError("Group structure is not defined, cannot get number of groups.")

    @property
    def sens(self):
        """
        Sensitivity profiles.

        Returns
        -------
        np.ndarray
            Sensitivity array whose shape is (nResp, nMat, nZaid, nMTs, nE).
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

    def _sens_eranos(self, value):
        """
        Build the sensitivity array for ERANOS files.

        Parameters
        ----------
        value : dict
            Nested dictionary with sensitivity data from ERANOS.

        Sets
        ----
        self._sens : np.ndarray
            Sensitivity array with shape (nResp, nMat, nZaid, nMTs, nE).
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
                    for mt, iMT in self.MTs.items():
                        if zais in value[resp][mat].keys():
                            sens[iResp, iMat, iZaid, iMT, :] = value[resp][mat][zais][iMT, :]
                        else:
                            sens[iResp, iMat, iZaid, iMT, :] = dummy

        self._sens = sens

    def _sens_serpent(self, value):
        """
        Build the sensitivity dictionary for Serpent files.

        Parameters
        ----------
        value : dict
            Dictionary with sensitivity data from Serpent.

        Sets
        ----
        self._sens : dict
            Dictionary of sensitivity arrays for each response.
        """
        arr_shape = None
        for k, v in value.items():
            if not isinstance(v, np.ndarray):
                raise ValueError(f"The keys of the sensitivity dict must be np.array, not of type {type(v)}."
                                "The structure of serpentTools.parsers.sensitivity object might have changed!")
            if arr_shape is None:
                arr_shape = v.shape
            elif arr_shape != v.shape:
                raise ValueError(f"All sensitivity arrays must have the same shape, but found {arr_shape} and {v.shape} for {k}."
                                "The structure of serpentTools.parsers.sensitivity object might have changed!")

        sens_avg = np.zeros((len(self.responses), *arr_shape[:-1]))
        for i, resp in enumerate(self.responses):
            if not isinstance(value[resp], np.ndarray):
                raise ValueError(f"The keys of the sensitivity dict must be np.array, not of type {type(value[resp])}")
            else:
                sens_avg[i, :, :, :, :] = value[resp][:, :, :, ::-1, 0]

        self._sens = sens_avg

    @property
    def sens_rsd(self):
        """
        Relative standard deviations of the sensitivity profiles.

        Returns
        -------
        np.ndarray
            Sensitivity array whose shape is (nResp, nMat, nZaid, nMTs, nE).
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
        Build the sensitivity RSD dictionary for Serpent files.

        Parameters
        ----------
        value : dict
            Dictionary with sensitivity data from Serpent.

        Sets
        ----
        self._sens_rsd : dict
            Dictionary of sensitivity RSD arrays for each response.

        Raises
        ------
        ValueError
            If the sensitivity data is not a numpy array.
        """
        arr_shape = None
        for k, v in value.items():
            if not isinstance(v, np.ndarray):
                raise ValueError(f"The keys of the sensitivity dict must be np.array, not of type {type(v)}."
                                "The structure of serpentTools.parsers.sensitivity object might have changed!")
            if arr_shape is None:
                arr_shape = v.shape
            elif arr_shape != v.shape:
                raise ValueError(f"All sensitivity arrays must have the same shape, but found {arr_shape} and {v.shape} for {k}."
                                "The structure of serpentTools.parsers.sensitivity object might have changed!")

        sens_rsd = np.zeros((len(self.responses), *arr_shape[:-1]))
        for i, resp in enumerate(self.responses):
            if not isinstance(value[resp], np.ndarray):
                raise ValueError(f"The keys of the sensitivity dict must be np.array, not of type {type(value[resp])}")
            else:
                sens_rsd[i, :, :, :, :] = value[resp][:, :, :, ::-1, 1]

        self._sens_rsd = sens_rsd

    def get(self, resp=None, mat=None, MT=None, za=None, g=None, group_order='descending'):
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
            MT number(s) to extract, e.g., 2, 4, 18.
        If None, all MTs are extracted.
        za : str, int, or list, optional
            ZA string(s) or number(s) to extract, e.g., 'Pu-239', 942390 or ['Pu-239', 'Pu-240'].
            If None, all ZAIDs are extracted.
        g : int or list, optional
            Energy group(s) to extract, e.g., 1, 2, ..., n_groups.
            If None, all groups are extracted.

        Returns
        -------
        S_avg : np.ndarray or dict
            Sensitivity profile(s).
        S_rsd : np.ndarray or dict, optional
            Sensitivity RSD(s), if available.
        """
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
                raise ValueError(f"'mat' should be of type str or list, not of type {type(mat)}")
            else:
                for val in mat:
                    iM.append(self.materials[val])
        else:
            for val in self.materials:
                iM.append(self.materials[val])

        # --- get MT indexes
        if MT is not None:
            if isinstance(MT, int):
                if MT not in self.MTs:
                    raise ValueError(f"MT {MT} not available!")
                else:
                    iP = [self.MTs[MT]]
            elif not isinstance(MT, list):
                raise ValueError(f"'MT' should be of type int or list, not of type {type(MT)}")
            else:
                for val in MT:
                    iP.append(self.MTs[val])
        else:
            for mt in self.MTs:
                iP.append(self.MTs[mt])

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
                raise ValueError(f"'za' should be of type str or int, not of type {type(za)}")
            else:
                for val in za:
                    if isinstance(val, str):
                        iZ.append(self.zais[val])
                    elif isinstance(val, int):
                        iZ.append(self.zaid[val])
                    else:
                        raise ValueError(f"Elements in list 'za' should be of type str or int, not of type {type(za)}")
        else:
            for k, v in self.zaid.items():
                iZ.append(v)

        # --- get resp
        if resp is not None:
            if isinstance(resp, str):
                if resp not in self.responses:
                    raise ValueError(f"Response {resp} not available!")
                else:
                    iR = [self.responses.index(resp)]
            elif not isinstance(resp, list):
                raise ValueError(f"'resp' should be of type str or list, not of type {type(resp)}")
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
        if g is not None:
            if isinstance(g, int):
                if group_order == 'ascending':
                    iG = [self._groups - 1]
                else:
                    iG = [g - 1]
            elif not isinstance(g, list):
                raise ValueError(f"'g' must be of type int or list, not of type {type(g)}")
        else:
            iG = [ig for ig in range(len(self.group_structure) - 1)]
            if group_order == 'ascending':
                iG = iG[::-1]

        # --- get sensitivity vector and uncertainty
        S_avg = self.sens[np.ix_(iR, iM, iZ, iP, iG)]
        if self.sens_rsd is not None:
            S_rsd = self.sens_rsd[np.ix_(iR, iM, iZ, iP, iG)]
            return S_avg, S_rsd
        else:
            return S_avg

class SensitivityError(Exception):
    pass


