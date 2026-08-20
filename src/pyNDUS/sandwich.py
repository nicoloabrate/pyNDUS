"""
author: N. Abrate.

file: Sandwich.py

description: propagate sensitivity profiles and covariance matrices with
             sandwich formulas.
"""
import numpy as np
from numpy import trapz
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy as unp
from collections import OrderedDict
from itertools import permutations
from copy import deepcopy as copy
try:
    from .sensitivity import Sensitivity
    from . import utils
except ImportError:  # when run as a script
    from sensitivity import Sensitivity
    import utils

try:
    from .covariance import Covariance
except ModuleNotFoundError:
    try:
        from covariance import Covariance
    except ModuleNotFoundError:
        Covariance = None


def _normalize_errorr_mf(MF):
    """
    Normalize an MF identifier to the internal ``"errorr<MF>"`` string form.

    Parameters
    ----------
    MF : int or str
        MF identifier such as ``33``, ``"33"``, ``"MF33"``, or
        ``"errorr33"``.

    Returns
    -------
    str
        Normalized ERRORR key.
    """
    if isinstance(MF, int):
        return f"errorr{MF}"
    if not isinstance(MF, str):
        raise ValueError(f"MF must be int or str, not {type(MF)}")
    if MF.startswith("errorr"):
        return MF
    if MF.startswith("MF"):
        return f"errorr{int(MF[2:])}"
    return f"errorr{int(MF)}"


def _normalize_errorr_mfs(list_MFs):
    """
    Normalize one or more MF identifiers.

    Parameters
    ----------
    list_MFs : int, str, list, or None
        MF selection. ``None`` is returned unchanged.

    Returns
    -------
    list[str] or None
        Normalized ERRORR keys.
    """
    if list_MFs is None:
        return None
    if isinstance(list_MFs, (int, str)):
        list_MFs = [list_MFs]
    elif not isinstance(list_MFs, list):
        raise ValueError(
            f"list_MFs argument must be int, str or list, not {type(list_MFs)}"
        )
    return [_normalize_errorr_mf(MF) for MF in list_MFs]


def _errorr_mf_to_int(MF):
    """
    Convert an MF identifier to its integer ENDF MF number.

    Parameters
    ----------
    MF : int or str
        MF identifier accepted by :func:`_normalize_errorr_mf`.

    Returns
    -------
    int
        Integer MF number.
    """
    return int(_normalize_errorr_mf(MF).replace("errorr", ""))


def _merge_or_set(container, key, value, MF, sum_MFs=False):
    """
    Store one MF/MT contribution, optionally summing duplicate entries.

    Parameters
    ----------
    container : dict
        Dictionary receiving values keyed by ``(MF, MT_row, MT_col)``.
    key : tuple[int, int]
        MT pair before adding the MF level.
    value : scalar
        Value to store or add.
    MF : int or str
        MF identifier associated with ``value``.
    sum_MFs : bool, optional
        If True, duplicate MF/MT keys are summed. Otherwise duplicates raise an
        error.
    """
    key = (_errorr_mf_to_int(MF), key[0], key[1])
    if key in container:
        if not sum_MFs:
            raise ValueError(
                f"The MF/MT pair {key} is present more than once. "
                f"Select a single MF with list_MFs=[...] or set sum_MFs=True. "
                f"Last attempted MF: {MF}.")
        container[key] = container[key] + value
    else:
        container[key] = value


def _build_result_frames(records, columns, index_columns, include_MF=False,
                         sum_MFs=False):
    """
    Build wide result matrices and keep the long table in DataFrame metadata.

    Parameters
    ----------
    records : list[tuple]
        Rows used to build the long result table.
    columns : list[str]
        Column names for ``records``.
    index_columns : list[str]
        Leading columns to keep in the returned matrix index.
    include_MF : bool, optional
        If True, keep MF as an index level in the returned matrix.
    sum_MFs : bool, optional
        If True, sum duplicate MT pairs across MF sections when ``include_MF``
        is False.

    Returns
    -------
    pandas.DataFrame
        Wide matrix indexed by response/material/ZA/MT row. The long table is
        stored in ``attrs["table"]`` and the MF-resolved wide matrix in
        ``attrs["by_mf"]``.
    """
    df = pd.DataFrame(records, columns=columns)
    by_mf_index = index_columns + ["MF", "MT_row", "MT_col"]
    by_mf_table = df.set_index(by_mf_index).sort_index()
    by_mf_matrix = by_mf_table["value"].unstack("MT_col")

    if include_MF:
        matrix = by_mf_matrix.copy()
    else:
        legacy_index = index_columns + ["MT_row", "MT_col"]
        duplicated = df.duplicated(legacy_index, keep=False)
        if duplicated.any() and not sum_MFs:
            raise ValueError(
                "The same MT pair is present in more than one MF section. "
                "Select a single MF with list_MFs=[...], set include_MF=True, "
                "or set sum_MFs=True.")
        if sum_MFs:
            legacy_df = df.groupby(legacy_index, sort=False,
                                   as_index=False)["value"].sum()
        else:
            legacy_df = df.drop(columns=["MF"])
        matrix = legacy_df.set_index(legacy_index)["value"].unstack("MT_col")

    matrix.attrs["table"] = by_mf_table
    matrix.attrs["by_mf"] = by_mf_matrix
    return matrix


class Sandwich:
    """
    Compute uncertainty, representativity, or similarity from sensitivities.

    Parameters
    ----------
    sens : Sensitivity
        Reference sensitivity object.
    sens2 : Sensitivity, optional
        Second sensitivity object, required for representativity and
        similarity calculations.
    covmat : dict[int, Covariance], optional
        Covariance objects keyed by ZAID. Required for uncertainty and
        representativity calculations.
    sigma : int or float, optional
        Multiplier applied to Monte Carlo sensitivity relative standard
        deviations. Ignored when sensitivities do not include RSDs.
    verbosity : bool, optional
        Retained for API compatibility.
    list_resp : str or list[str], optional
        Responses to process. If ``None``, use all responses available in the
        provided sensitivity objects.
    list_mat : str or list[str], optional
        Materials to process for uncertainty calculations.
    list_za : str, int, or list, optional
        Isotopes to process, either as ZAID integers or isotope labels.
    list_MTs : int or list[int], optional
        MT reactions to process. If ``None``, use the intersection available in
        sensitivities and covariances.
    list_MFs : int, str, list, or None, optional
        MF sections to process. The default uses the historical MF31/MF33
        selection when covariances are available.
    sum_MFs : bool, optional
        Sum contributions that share the same MT pair across MF sections.
    include_MF : bool, optional
        Keep MF as an explicit output index level.
    uncertainty_output : {"variance", "signed_sqrt"}, optional
        Representation exposed through ``uncertainty``. ``"variance"`` keeps
        the raw contributions :math:`S_i^T C_{ij} S_j`; ``"signed_sqrt"``
        returns ``sign(q) * sqrt(abs(q))`` for each contribution ``q``.
        The default is ``"variance"``.
    representativity : bool, optional
        If True, compute representativity between ``sens`` and ``sens2``.
    similarity : bool, optional
        If True, compute similarity between ``sens`` and ``sens2`` without
        covariance weighting.

    Attributes
    ----------
    calculation_type : {"uncertainty", "representativity", "similarity"}
        Type of calculation performed.
    uncertainty, representativity, similarity : pandas.DataFrame
        Result matrix for the selected calculation type.
    uncertainty_variance, uncertainty_signed_sqrt : pandas.DataFrame
        Variance-contribution and signed-square-root matrices. Available for
        uncertainty calculations regardless of ``uncertainty_output``.
    uncertainty_standard_deviation : pandas.DataFrame
        Total variance and standard deviation for each response/material,
        obtained by summing all selected ZA/MF/MT contributions before taking
        the square root.
    representativity_total : pandas.Series
        Total representativity coefficient for each response, obtained by
        summing the normalized ZA/MF/MT-pair contributions in
        ``representativity_table``.
    *_table : pandas.DataFrame
        Long MF-resolved result table.
    *_by_mf : pandas.DataFrame
        MF-resolved wide result matrix.
    dict_map : dict
        Mapping from response names to the metadata indices used during the
        calculation.
    """

    def __init__(self, sens, sens2=None, covmat=None, sigma=2, verbosity=False,
                 list_resp=None, list_mat=None, list_za=None, list_MTs=None,
                 list_MFs=None, sum_MFs=False, include_MF=False,
                 representativity=False, similarity=False,
                 uncertainty_output="variance"):
        """
        Initialize a sandwich-formula calculation and store the result.

        See the class docstring for parameter descriptions. Exactly one
        calculation mode is selected: uncertainty by default, representativity
        when ``representativity=True``, or similarity when ``similarity=True``.
        """

        # --- validate input arguments
        if similarity:
            self.calculation_type = "similarity"
            if sens2 is None:
                raise ValueError(
                    f"'sens2' arg is needed to perform representativity calculations!"
                )
        elif representativity:
            if covmat is None:
                raise ValueError(
                    f"'covmat' arg is needed to perform representativity calculations!"
                )
            if sens2 is None:
                raise ValueError(
                    f"'sens2' arg is needed to perform representativity calculations!"
                )
            self.calculation_type = "representativity"
        else:
            if covmat is None:
                raise ValueError(
                    f"'covmat' arg is needed to perform uncertainty calculations!"
                )
            self.calculation_type = "uncertainty"

        if representativity and similarity:
            raise ValueError(
                "Cannot perform 'representativity' and 'similariy' calculation at the same time. "
                "Just one argument can be True")

        if not isinstance(uncertainty_output, str):
            raise ValueError(
                "'uncertainty_output' must be 'variance' or 'signed_sqrt', "
                f"not {type(uncertainty_output)}")
        uncertainty_output = uncertainty_output.lower()
        if uncertainty_output not in {"variance", "signed_sqrt"}:
            raise ValueError(
                "'uncertainty_output' must be 'variance' or 'signed_sqrt', "
                f"not {uncertainty_output!r}")
        if self.calculation_type != "uncertainty" and (
                uncertainty_output != "variance"):
            raise ValueError(
                "'uncertainty_output' applies only to uncertainty "
                "calculations.")
        self.uncertainty_output = uncertainty_output

        if not isinstance(sens, Sensitivity):
            raise ValueError(
                f"'sens' arg must be of type pyNDUS.Sensitivity, not of type {type(sens)}"
            )
        else:
            if sens.sens_rsd is not None:
                sens_MC = True
            else:
                sens_MC = False

        sens2_MC = None
        if sens2 is not None:
            if not isinstance(sens2, Sensitivity):
                raise ValueError(
                    f"'sens2' arg must be of type pyNDUS.Sensitivity, not of type {type(sens2)}"
                )
            else:
                if sens.sens_rsd is not None:
                    sens2_MC = True

        if covmat is not None:
            if not isinstance(covmat, dict):
                raise ValueError(
                    f"'covmat' arg must be of type dict, not of type {type(covmat)}"
                )
            else:
                if Covariance is None:
                    raise SandwichError(
                        "Covariance is unavailable because its optional "
                        "dependencies are not installed.")
                if len(covmat) == 0:
                    raise ValueError(
                        f"'covmat' dict is empty! Check the dict with the covariances."
                    )
                else:
                    for k, v in covmat.items():
                        if not isinstance(v, Covariance):
                            raise ValueError(
                                f"'covmat' items must be of type Covariance, not of type {type(covmat)}"
                            )
            is_covmat = True
        else:
            is_covmat = False

        if sens_MC or sens2_MC:
            self.sens_MC = True
        else:
            self.sens_MC = False

        self.sigma = sigma

        # --- check responses
        if list_resp is None:
            list_resp = list(sens.responses)
            if sens2 is not None:
                list_resp += list(sens2.responses)
            list_resp = list(set(list_resp))
        elif isinstance(list_resp, str):
            if list_resp not in sens.responses:
                raise ValueError(
                    f"{list_resp} not available in 'Sensitivity' object provided!"
                )
            if sens2 is not None:
                if list_resp not in sens2.responses:
                    raise ValueError(
                        f"{list_resp} not available in 'Sensitivity' object 'sens2' provided!"
                    )
            list_resp = [list_resp]
        elif not isinstance(list_resp, list):
            raise ValueError(
                f"'list_resp' arg must be str or list, not {type(list_resp)}")
        else:
            for resp in list_resp:
                if resp not in sens.responses:
                    raise ValueError(
                        f"{resp} not available in 'Sensitivity' object provided!"
                    )
                if sens2 is not None:
                    if resp not in sens2.responses:
                        raise ValueError(
                            f"{resp} not available in 'Sensitivity' object 'sens2' provided!"
                        )

        # --- check materials
        if not representativity:
            if list_mat is None:
                list_mat = list(sens.materials.keys())
                if sens2 is not None:
                    list_mat += list(sens2.materials.keys())
                list_mat = list(set(list_mat))
            elif isinstance(list_mat, str):
                if list_mat not in sens.materials.keys():
                    raise ValueError(
                        f"{list_mat} not available in 'Sensitivity' object provided!"
                    )
                if sens2 is not None:
                    if list_mat not in sens2.materials.keys():
                        raise ValueError(
                            f"{list_mat} not available in 'Sensitivity' object 'sens2' provided!"
                        )
                list_mat = [list_mat]
            elif not isinstance(list_mat, list):
                raise ValueError(
                    f"'list_mat' arg must be str or list, not {type(list_mat)}"
                )
            else:
                for mat in list_mat:
                    if mat not in sens.materials.keys():
                        raise ValueError(
                            f"{mat} not available in 'Sensitivity' object provided!"
                        )
                    if sens2 is not None:
                        if mat not in sens2.materials.keys():
                            raise ValueError(
                                f"{mat} not available in 'Sensitivity' object 'sens2' provided!"
                            )

        # --- check responses
        if list_za is None:
            list_za = list(sens.zaid.keys())
            if sens2 is not None:
                list_za += list(sens2.zaid.keys())
            # if is_covmat:
            #     list_za += list(covmat.keys())
            list_za = list(set(list_za))
        elif isinstance(list_za, str):
            if list_za not in sens.zais.keys():
                raise ValueError(
                    f"{list_za} not available in 'Sensitivity' object provided!"
                )
            if sens2 is not None:
                if list_za not in sens2.zais.keys():
                    raise ValueError(
                        f"{list_za} not available in 'Sensitivity' object 'sens2' provided!"
                    )
            list_za = [sens.zais[list_za]]
        elif isinstance(list_za, int):
            if list_za not in sens.zaid.keys():
                raise ValueError(
                    f"{list_za} not available in 'Sensitivity' object provided!"
                )
            if sens2 is not None:
                if list_za not in sens2.zaid.keys():
                    raise ValueError(
                        f"{list_za} not available in 'Sensitivity' object 'sens2' provided!"
                    )
            list_za = [sens.zaid[list_za]]
        elif not isinstance(list_za, list):
            raise ValueError(
                f"'list_za' arg must be str or list, not {type(list_za)}")
        else:
            for zaid in list_za:
                if isinstance(zaid, int):
                    if zaid not in sens.zaid.keys():
                        raise ValueError(
                            f"{zaid} not available in 'Sensitivity' object provided!"
                        )
                    if sens2 is not None:
                        if zaid not in sens2.zaid.keys():
                            raise ValueError(
                                f"{zaid} not available in 'Sensitivity' object 'sens2' provided!"
                            )
                elif isinstance(zaid, str):
                    if zaid not in sens.zais.keys():
                        raise ValueError(
                            f"{zaid} not available in 'Sensitivity' object provided!"
                        )
                    if sens2 is not None:
                        if zaid not in sens2.zais.keys():
                            raise ValueError(
                                f"{zaid} not available in 'Sensitivity' object 'sens2' provided!"
                            )

        # --- check MTs
        if list_MTs is None:
            get_MTs = True
        elif isinstance(list_MTs, int):
            list_MTs = [list_MTs]
            get_MTs = False
        elif isinstance(list_MTs, list):
            get_MTs = False
        else:
            raise ValueError(
                f"list_MTs argument must be of type 'list', not {type(list_MTs)}"
            )

        selected_MFs = _normalize_errorr_mfs(list_MFs)
        if not isinstance(sum_MFs, bool):
            raise ValueError(
                f"sum_MFs argument must be bool, not {type(sum_MFs)}")
        if not isinstance(include_MF, bool):
            raise ValueError(
                f"include_MF argument must be bool, not {type(include_MF)}")

        if not similarity and not representativity:
            if not is_covmat:
                raise SandwichError(
                    "covmat optional argument needed to get the uncertainty propagated with the sandwich formula!"
                )

        # --- enforce consistent set of MTs for any isotope
        sens_MTs = {}
        rcov_MTs = {}
        map_MF2MT = {}

        self.MTs = {}
        za_dict = {}
        for iza, za in enumerate(list_za):

            if not isinstance(za, int):
                za = utils.zais2zaid(za)

            if za in sens.zaid.keys():
                idx = sens.zaid[za]
                zais = list(sens.zais.keys())[idx]

            elif sens2 is not None:
                if za in sens2.zaid.keys():
                    idx = sens2.zaid[za]
                    zais = list(sens2.zais.keys())[idx]

            if is_covmat:
                if not representativity:
                    if za in covmat.keys():
                        za_dict[za] = zais
                else:
                    za_dict[za] = zais
            else:
                za_dict[za] = zais

            # --- get MTs in covariance
            if is_covmat:
                if za in covmat.keys():
                    map_MF2MT[za] = copy(covmat[za].MFs2MTs)
                    if selected_MFs is None:
                        # Historical default: cross sections/nubar only.
                        # MF34/MF35 are available on explicit request through list_MFs.
                        allowed_MFs = {"errorr31", "errorr33"}
                    else:
                        allowed_MFs = set(selected_MFs)
                    map_MF2MT[za] = {
                        mf: mts
                        for mf, mts in map_MF2MT[za].items()
                        if mf in allowed_MFs
                    }
                else:
                    map_MF2MT[za] = {}
                    default_MFs = selected_MFs or ["errorr33"]
                    for mf in default_MFs:
                        map_MF2MT[za][mf] = []

            # --- get MTs in sensitivity
            sens_MTs[za] = list(sens.MTs.keys())
            if sens2 is not None:
                sens_MTs[za] += list(sens2.MTs.keys())
                sens_MTs[za] = list(set(sens_MTs[za]))

            # --- enforce consistency between covariance (if any) and sensitivities
            if is_covmat:
                covMTs = []
                if za in covmat.keys():
                    for mf in map_MF2MT[za].keys():
                        covMTs += map_MF2MT[za][mf][:]
                    covMTs.sort()

                if get_MTs:
                    intersection = list(set(covMTs) & set(sens_MTs[za]))
                    intersection.sort()

                    sens_MTs[za] = intersection.copy()
                    if za in za_dict.keys():
                        if not representativity:
                            self.MTs[za] = intersection.copy()
                        else:
                            self.MTs[za] = intersection

                        if 1 in self.MTs[za]:
                            if len(self.MTs[za]) > 2:
                                self.MTs[za].remove(1)
                            elif max(self.MTs[za]) < 110 and len(self.MTs[za]) == 2:
                                self.MTs[za].remove(1)
                        # remove MF=31 (452) from serpent sensitivities to avoid double counting the profiles
                        if sens.reader == 'serpent':
                            if 452 in self.MTs[za]:
                                if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                    self.MTs[za].remove(452)
                        if sens2 is not None:
                            if sens2.reader == 'serpent':
                                if 452 in self.MTs[za]:
                                    if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                        self.MTs[za].remove(452)

                        for mf in map_MF2MT[za].keys():
                            map_MF2MT[za][mf] = []

                        for mf in map_MF2MT[za].keys():
                            map_MF2MT[za][mf] = [
                                mt for mt in intersection
                                if mt in covmat[za].MFs2MTs.get(mf, [])
                            ]

                else:

                    intersection = list(
                        set(list_MTs) & set(covMTs) & set(sens_MTs[za]))
                    intersection.sort()
                    sens_MTs[za] = intersection.copy()

                    if za in za_dict.keys():
                        self.MTs[za] = intersection.copy()

                        if 1 in self.MTs[za]:
                            if len(self.MTs[za]) > 2:
                                self.MTs[za].remove(1)
                            elif max(self.MTs[za]) < 110 and len(self.MTs[za]) == 2:
                                self.MTs[za].remove(1)
                        # remove MF=31 (452) from serpent sensitivities to avoid double counting the profiles
                        if sens.reader == 'serpent':
                            if 452 in self.MTs[za]:
                                if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                    self.MTs[za].remove(452)
                        if sens2 is not None:
                            if sens2.reader == 'serpent':
                                if 452 in self.MTs[za]:
                                    if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                        self.MTs[za].remove(452)

                    if za in map_MF2MT.keys():
                        for mf in map_MF2MT[za].keys():
                            map_MF2MT[za][mf] = []

                    if za in covmat.keys():
                        for mf in map_MF2MT[za].keys():
                            map_MF2MT[za][mf] = [
                                mt for mt in intersection
                                if mt in covmat[za].MFs2MTs.get(mf, [])
                            ]

            else:
                if get_MTs:
                    if za in za_dict.keys():
                        self.MTs[za] = sens_MTs[za]
                        # remove total XS to avoid double counting the profiles
                        if 1 in self.MTs[za]:
                            if len(self.MTs[za]) > 2:
                                self.MTs[za].remove(1)
                            elif max(self.MTs[za]) < 110 and len(self.MTs[za]) == 2:
                                self.MTs[za].remove(1)
                        # remove MF=31 (452) from serpent sensitivities to avoid double counting the profiles
                        if sens.reader == 'serpent':
                            if 452 in self.MTs[za]:
                                if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                    self.MTs[za].remove(452)
                        if sens2 is not None:
                            if sens2.reader == 'serpent':
                                if 452 in self.MTs[za]:
                                    if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                        self.MTs[za].remove(452)

                else:
                    intersection = list(set(list_MTs) & set(sens_MTs[za]))
                    sens_MTs[za] = intersection.copy()
                    if za in za_dict.keys():
                        self.MTs[za] = intersection.copy()

                        if 1 in self.MTs[za]:
                            if len(self.MTs[za]) > 2:
                                self.MTs[za].remove(1)
                            elif max(self.MTs[za]) < 110 and len(self.MTs[za]) == 2:
                                self.MTs[za].remove(1)
                        # remove MF=31 (452) from serpent sensitivities to avoid double counting the profiles
                        if sens.reader == 'serpent':
                            if 452 in self.MTs[za]:
                                if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                    self.MTs[za].remove(452)
                        if sens2 is not None:
                            if sens2.reader == 'serpent':
                                if 452 in self.MTs[za]:
                                    if 455 in self.MTs[za] and 456 in self.MTs[za]:
                                        self.MTs[za].remove(452)

                if za in self.MTs:
                    default_MFs = selected_MFs or ["errorr33"]
                    map_MF2MT[za] = {mf: self.MTs[za].copy() for mf in default_MFs}

        if len(za_dict) == 0:
            raise SandwichError(
                "No valid ZAIDs found. Check that there is a non-null intersection between ZAIDs "
                "provided in the sensitivity and covariance (if any) objects.")
        else:
            self.za = za_dict

        self.MFs2MTs = map_MF2MT
        self.include_MF = include_MF

        # --- assign output
        if representativity:
            representativity, dict_map = self.compute_representativity(
                sens, sens2, covmat, list_resp, map_MF2MT, self.za, self.sens_MC,
                sigma=self.sigma, sum_MFs=sum_MFs, include_MF=include_MF)
            self.representativity = representativity
            self.representativity_table = representativity.attrs.get("table")
            self.representativity_by_mf = representativity.attrs.get("by_mf")
            self.dict_map = dict_map
        elif similarity:
            similarity, dict_map = self.compute_similarity(
                sens, sens2, list_resp, map_MF2MT, self.za, self.sens_MC,
                sigma=self.sigma, sum_MFs=sum_MFs, include_MF=include_MF)
            self.similarity = similarity
            self.similarity_table = similarity.attrs.get("table")
            self.similarity_by_mf = similarity.attrs.get("by_mf")
            self.dict_map = dict_map
        else:
            uncertainty_variance, dict_map = self.compute_uncertainty(
                sens, covmat, list_resp, list_mat, map_MF2MT, self.za, self.sens_MC,
                sigma=self.sigma, sum_MFs=sum_MFs, include_MF=include_MF)
            uncertainty_signed_sqrt = self._signed_sqrt_uncertainty(
                uncertainty_variance)
            self.uncertainty_variance = uncertainty_variance
            self.uncertainty_variance_table = uncertainty_variance.attrs.get(
                "table")
            self.uncertainty_variance_by_mf = uncertainty_variance.attrs.get(
                "by_mf")
            self.uncertainty_signed_sqrt = uncertainty_signed_sqrt
            self.uncertainty_signed_sqrt_table = (
                uncertainty_signed_sqrt.attrs.get("table"))
            self.uncertainty_signed_sqrt_by_mf = (
                uncertainty_signed_sqrt.attrs.get("by_mf"))
            self.uncertainty_standard_deviation = (
                self._standard_deviation_from_variance(
                    uncertainty_variance))

            if uncertainty_output == "signed_sqrt":
                uncertainty = uncertainty_signed_sqrt
            else:
                uncertainty = uncertainty_variance
            self.uncertainty = uncertainty
            self.uncertainty_table = uncertainty.attrs.get("table")
            self.uncertainty_by_mf = uncertainty.attrs.get("by_mf")
            self.dict_map = dict_map

    @property
    def sigma(self):
        """
        Sensitivity RSD multiplier used when Monte Carlo uncertainties exist.

        Returns
        -------
        int, float, or None
            Sigma multiplier, or ``None`` for deterministic sensitivities.
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """
        Set the sensitivity RSD multiplier.

        Parameters
        ----------
        value : int or float
            Non-negative multiplier. Ignored and stored as ``None`` when the
            sensitivity inputs do not include RSD data.
        """

        if self.sens_MC:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"'sigma' must be a number, not {type(value)}")
            if value < 0:
                raise ValueError(f"'sigma' must be positive, not {value}")
            self._sigma = value
        else:
            self._sigma = None

    @property
    def za(self):
        """
        Mapping from selected ZAID integers to isotope labels.

        Returns
        -------
        dict[int, str]
            ZAID-to-label mapping used in result indices.
        """
        return self._za

    @za.setter
    def za(self, value):
        """
        Validate and store the ZAID-to-label mapping.

        Parameters
        ----------
        value : dict[int, str]
            ZAID-to-label mapping.
        """
        if not isinstance(value, dict):
            raise ValueError(f"'za' must be a dict, not {type(value)}")
        for za, zais in value.items():
            if not isinstance(za, int):
                raise ValueError(
                    f"keys of 'za' dict must be int, not {type(za)}")
            if not isinstance(zais, str):
                raise ValueError(
                    f"values of 'za' dict must be str, not {type(zais)}")
        self._za = value

    @property
    def representativity_total(self):
        """
        Return the total representativity coefficient for each response.

        Entries in ``representativity_table`` are contributions to one
        globally normalized coefficient. The physical representativity is
        therefore their sum over all selected ZA, MF, MT-row, and MT-column
        values. The returned Series always keeps RESPONSE as its index, even
        when only one response was requested.

        Returns
        -------
        pandas.Series
            Total representativity indexed by response.
        """
        if not hasattr(self, "representativity_table"):
            raise AttributeError(
                "'representativity_total' is available only after a "
                "representativity calculation.")

        totals = {}
        values = self.representativity_table["value"]
        for response, contributions in values.groupby(
                level="RESPONSE", sort=False):
            totals[response] = sum(contributions.tolist(), 0.0)

        return pd.Series(
            totals, name="representativity_total",
            index=pd.Index(totals.keys(), name="RESPONSE"))

    def get_result_table(self, quantity=None, response=None, material=None, ZA=None,
                         MF=None, MT=None):
        """
        Return the long result table, optionally filtered by response/material/ZA/MF/MT.

        The returned table always keeps MF and both MT axes in the index.

        Parameters
        ----------
        quantity : {"uncertainty", "representativity", "similarity"}, optional
            Quantity to retrieve. If ``None``, use the calculation performed by
            this object.
        response : str or list[str], optional
            Response filter.
        material : str or list[str], optional
            Material filter. Only available for uncertainty results.
        ZA : str or list[str], optional
            Isotope-label filter.
        MF : int, str, or list, optional
            MF filter.
        MT : int, tuple[int, int], or list[int], optional
            MT filter. A two-value tuple filters one MT-row/MT-column pair.

        Returns
        -------
        pandas.DataFrame
            Filtered long table indexed by the original table index levels.
        """
        if quantity is None:
            quantity = self.calculation_type
        quantity = quantity.lower()
        attr = f"{quantity}_table"
        if not hasattr(self, attr):
            raise ValueError(f"No table available for quantity '{quantity}'.")

        table = getattr(self, attr)
        if table is None:
            raise ValueError(f"No table available for quantity '{quantity}'.")

        df = table.reset_index()

        if response is not None:
            values = response if isinstance(response, list) else [response]
            df = df[df["RESPONSE"].isin(values)]

        if material is not None:
            if "MATERIAL" not in df.columns:
                raise ValueError(
                    f"Quantity '{quantity}' has no MATERIAL level.")
            values = material if isinstance(material, list) else [material]
            df = df[df["MATERIAL"].isin(values)]

        if ZA is not None:
            values = ZA if isinstance(ZA, list) else [ZA]
            df = df[df["ZA"].isin(values)]

        if MF is not None:
            values = MF if isinstance(MF, list) else [MF]
            values = [_errorr_mf_to_int(value) for value in values]
            df = df[df["MF"].isin(values)]

        if MT is not None:
            if isinstance(MT, tuple):
                if len(MT) != 2:
                    raise ValueError("MT tuple must have length 2.")
                df = df[(df["MT_row"] == MT[0]) & (df["MT_col"] == MT[1])]
            else:
                values = MT if isinstance(MT, list) else [MT]
                df = df[df["MT_row"].isin(values) & df["MT_col"].isin(values)]

        return df.set_index(table.index.names).sort_index()

    @staticmethod
    def _fallback_cov(n_groups: int, rsd: float, mt1: int, mt2: int,
                      corr: float = 0.0):
        """
        Build a fallback relative covariance matrix for missing data.

        Parameters
        ----------
        n_groups : int
            Number of energy groups.
        rsd : float
            Relative standard deviation used on the diagonal.
        mt1, mt2 : int
            MT pair represented by the matrix.
        corr : float, optional
            Correlation coefficient used for off-diagonal MT pairs.

        Returns
        -------
        numpy.ndarray
            Relative covariance matrix. Diagonal MT pairs use ``rsd**2`` on the
            diagonal; off-diagonal MT pairs use ``corr * rsd**2``.
        """
        var = float(rsd)**2
        if mt1 == mt2:
            return var * np.eye(n_groups)
        else:
            return float(corr) * var * np.eye(n_groups)

    @staticmethod
    def _signed_sqrt_value(value):
        """
        Return the sign-preserving square root of one variance contribution.

        ``uncertainties`` scalars retain their propagated uncertainty. Exact
        zeros are returned unchanged to avoid the singular derivative of the
        square root at zero.
        """
        nominal = getattr(value, "nominal_value", value)
        if pd.isna(nominal):
            return value
        if nominal < 0:
            if hasattr(value, "nominal_value"):
                return -unp.sqrt(-value)
            return -np.sqrt(-value)
        if nominal > 0:
            if hasattr(value, "nominal_value"):
                return unp.sqrt(value)
            return np.sqrt(value)
        return value

    @staticmethod
    def _signed_sqrt_uncertainty(variance):
        """
        Transform every variance contribution to ``sign(q)*sqrt(abs(q))``.

        The returned matrix retains the same index, columns, and MF-resolved
        metadata tables as the input variance matrix.
        """
        def transform(frame):
            if frame is None:
                return None
            return frame.apply(
                lambda column: column.map(Sandwich._signed_sqrt_value))

        signed_sqrt = transform(variance)
        table = variance.attrs.get("table")
        if table is not None:
            table = table.copy()
            table["value"] = table["value"].map(
                Sandwich._signed_sqrt_value)
        by_mf = transform(variance.attrs.get("by_mf"))
        signed_sqrt.attrs["table"] = table
        signed_sqrt.attrs["by_mf"] = by_mf
        return signed_sqrt

    @staticmethod
    def _standard_deviation_from_variance(variance):
        """
        Sum all selected contributions before taking the square root.

        Returns one row per response/material. A negative total variance has no
        real standard deviation and is reported as ``NaN``.
        """
        records = []
        levels = ["RESPONSE", "MATERIAL"]
        for (response, material), group in variance.groupby(
                level=levels, sort=False):
            values = [
                value
                for value in group.to_numpy(dtype=object).ravel()
                if not pd.isna(value)
            ]
            total_variance = sum(values, 0.0)
            nominal = getattr(
                total_variance, "nominal_value", total_variance)
            if nominal < 0:
                standard_deviation = np.nan
            elif nominal > 0:
                if hasattr(total_variance, "nominal_value"):
                    standard_deviation = unp.sqrt(total_variance)
                else:
                    standard_deviation = np.sqrt(total_variance)
            else:
                standard_deviation = total_variance
            records.append(
                (response, material, total_variance, standard_deviation))

        return pd.DataFrame(
            records,
            columns=[
                "RESPONSE", "MATERIAL", "variance",
                "standard_deviation",
            ],
        ).set_index(["RESPONSE", "MATERIAL"])

    @staticmethod
    def compute_similarity(sens, sens2, list_resp, list_MTs, za_dict, sens_MC,
                           sigma=None, sum_MFs=False, include_MF=False):
        """
        Compute pairwise normalized sensitivity-vector similarity.

        For every requested reaction pair ``(MT_i, MT_j)``, the returned
        coefficient is the cosine similarity

        ``S_i.T @ S_j / (sqrt(S_i.T @ S_i) * sqrt(S_j.T @ S_j))``.

        The two denominator terms therefore refer to the same two vectors used
        in that element's numerator; no global norm over other reactions is
        used.  This is the unweighted form of the TSUNAMI coefficient (unit
        covariance), so every coefficient with non-zero norms lies in
        ``[-1, 1]``.  A comparison involving a missing or zero-norm profile is
        reported as zero.

        Parameters
        ----------
        sens, sens2 : Sensitivity
            Sensitivity objects to compare.
        list_resp : list[str]
            Responses to process.
        list_MTs : dict
            Mapping from ZAID to MT lists, or to MF-to-MT mappings.
        za_dict : dict[int, str]
            ZAID-to-label mapping used in the output.
        sens_MC : bool
            Whether sensitivity RSDs should be propagated with
            ``uncertainties``.
        sigma : int or float, optional
            Multiplier applied to sensitivity RSDs.
        sum_MFs : bool, optional
            Sum duplicate MT pairs across MF sections.
        include_MF : bool, optional
            Keep MF as an explicit output index level.

        Returns
        -------
        similarity : pandas.DataFrame
            Pairwise normalized dot-product matrix. Rows identify sensitivity
            vectors from ``sens`` and columns identify vectors from ``sens2``.
        dict_map : dict
            Metadata-to-index mapping used during the calculation.
        """

        if sens.reader == 'serpent':
            mat = 'total'
        elif sens.reader == 'eranos':
            mat = 'REACTOR'

        if sens2.reader == 'serpent':
            mat2 = 'total'
        elif sens2.reader == 'eranos':
            mat2 = 'REACTOR'

        map_MF2MT = {}
        for za, mts_or_mfs in list_MTs.items():
            if isinstance(mts_or_mfs, dict):
                map_MF2MT[za] = {
                    _normalize_errorr_mf(mf): mts
                    for mf, mts in mts_or_mfs.items()
                }
            else:
                map_MF2MT[za] = {"errorr33": mts_or_mfs}

        # Apply the scalar product (i.e., the TSUNAMI expression with unit
        # covariance). Each MT pair is normalized with the two profiles in its
        # own numerator, rather than with a norm accumulated over all MTs.
        output = {}
        dict_map = {}

        for resp in list_resp:

            output[resp] = {}
            dict_map[resp] = {}

            for key in ['zaid', 'MTs']:
                dict_map[resp][key] = {}

            for iza, za in enumerate(za_dict.keys()):

                dict_map[resp]["zaid"][za] = iza
                output[resp][za] = {}

                for mf, mts in map_MF2MT[za].items():
                    vectors1 = {}
                    vectors2 = {}

                    for mt in mts:
                        exist1 = (
                            resp in sens.responses
                            and mat in sens.materials
                            and mt in sens.MTs
                            and za in sens.zaid
                        )
                        exist2 = (
                            resp in sens2.responses
                            and mat2 in sens2.materials
                            and mt in sens2.MTs
                            and za in sens2.zaid
                        )

                        if exist1:
                            result1 = sens.get(
                                resp=[resp], mat=[mat], MT=[mt], za=[za],
                                group_order="ascending")
                            if sens_MC and isinstance(result1, tuple):
                                avg1, rsd1 = result1
                                vector1 = utils.np2unp(
                                    np.squeeze(avg1),
                                    sigma * np.squeeze(rsd1))
                            else:
                                vector1 = np.squeeze(result1)
                        else:
                            vector1 = np.zeros((sens.n_groups, ))

                        if exist2:
                            result2 = sens2.get(
                                resp=[resp], mat=[mat2], MT=[mt], za=[za],
                                group_order="ascending")
                            if sens_MC and isinstance(result2, tuple):
                                avg2, rsd2 = result2
                                vector2 = utils.np2unp(
                                    np.squeeze(avg2),
                                    sigma * np.squeeze(rsd2))
                            else:
                                vector2 = np.squeeze(result2)
                        else:
                            vector2 = np.zeros((sens2.n_groups, ))

                        vectors1[mt] = (vector1, exist1)
                        vectors2[mt] = (vector2, exist2)

                    for mt1 in mts:
                        vector1, exist1 = vectors1[mt1]
                        for mt2 in mts:
                            vector2, exist2 = vectors2[mt2]

                            if exist1 and exist2:
                                norm1 = unp.sqrt(np.dot(vector1.T, vector1))
                                norm2 = unp.sqrt(np.dot(vector2.T, vector2))
                                denominator = norm1 * norm2
                                denominator_nominal = getattr(
                                    denominator, "nominal_value", denominator)
                                if denominator_nominal != 0:
                                    value = (
                                        np.dot(vector1.T, vector2)
                                        / denominator
                                    )
                                else:
                                    value = 0
                            else:
                                value = 0

                            _merge_or_set(
                                output[resp][za], (mt1, mt2), value, mf,
                                sum_MFs=sum_MFs)

        # Convert the already pairwise-normalized output to a DataFrame.
        records = []
        for resp, zas in output.items():
            for za, mt_matrix in zas.items():
                for (mf, mt1, mt2), value in mt_matrix.items():
                    records.append(
                        (resp, za_dict[za], mf, mt1, mt2, value))

        similarity = _build_result_frames(
            records,
            ["RESPONSE", "ZA", "MF", "MT_row", "MT_col", "value"],
            ["RESPONSE", "ZA"],
            include_MF=include_MF,
            sum_MFs=sum_MFs,
            ).fillna(0)
        dict_map = dict_map

        return similarity, dict_map

    @staticmethod
    def compute_uncertainty(sens, covmat, list_resp, list_mat, map_MF2MT, za_dict,
                            sens_MC, sigma=None, sum_MFs=False, include_MF=False,
                            missing_cov="zero", missing_cov_rsd=0.20,
                            missing_cov_corr=0.0):
        """
        Propagate nuclear data covariance to response uncertainty.

        Parameters
        ----------
        sens : Sensitivity
            Sensitivity object.
        covmat : dict[int, Covariance]
            Covariance objects keyed by ZAID.
        list_resp : list[str]
            Responses to process.
        list_mat : list[str]
            Materials to process.
        map_MF2MT : dict
            Mapping from ZAID to MF-to-MT selections.
        za_dict : dict[int, str]
            ZAID-to-label mapping used in the output.
        sens_MC : bool
            Whether sensitivity RSDs should be propagated with
            ``uncertainties``.
        sigma : int or float, optional
            Multiplier applied to sensitivity RSDs.
        sum_MFs : bool, optional
            Sum duplicate MT pairs across MF sections.
        include_MF : bool, optional
            Keep MF as an explicit output index level.
        missing_cov : {"zero", "raise", "assume"}, optional
            Policy for missing covariance data.
        missing_cov_rsd : float, optional
            RSD used by the ``"assume"`` missing-covariance policy.
        missing_cov_corr : float, optional
            Off-diagonal correlation used by the ``"assume"`` policy.

        Returns
        -------
        uncertainty : pandas.DataFrame
            Uncertainty contribution matrix.
        dict_map : dict
            Metadata-to-index mapping used during the calculation.
        """
        # --- apply sandwich rule
        output = {}
        dict_map = {}

        for resp in list_resp:

            output[resp] = {}
            dict_map[resp] = {}
            for key in ['materials', 'zaid', 'MTs']:
                dict_map[resp][key] = {}

            for imat, mat in enumerate(list_mat):

                output[resp][mat] = {}
                dict_map[resp]["materials"][mat] = imat

                for iza, za in enumerate(za_dict.keys()):

                    if za in covmat.keys():
                        e6_mat_id = covmat[za].mat
                    else:
                        e6_mat_id = None

                    dict_map[resp]["zaid"][za] = iza
                    output[resp][mat][za] = {}

                    for mf in map_MF2MT[za].keys():
                        # --- get covariance matrix
                        cov_df = None
                        if za in covmat.keys():
                            if mf in covmat[za].rcov.keys():
                                cov_df = covmat[za]

                        # --- diagonal terms
                        for mt in map_MF2MT[za][mf]:
                            exist = (resp in sens.responses and mat in sens.materials and
                                     mt in sens.MTs and za in sens.zaid)
                            # get group-wise sensitivity vector
                            if sens_MC:
                                if exist:
                                    S_avg, S_rsd = sens.get(
                                        resp=[resp], mat=[mat], MT=[mt], za=[za],
                                        group_order="ascending")
                                else:
                                    S_avg = np.zeros((sens.n_groups, ))
                                    S_rsd = None

                                if S_rsd is not None:
                                    S = utils.np2unp(np.squeeze(S_avg),
                                                     sigma * np.squeeze(S_rsd))
                                else:
                                    S = np.squeeze(S_avg)

                            else:
                                if exist:
                                    S = np.squeeze(
                                        sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za],
                                                 group_order="ascending"))
                                else:
                                    S = np.zeros((sens.n_groups, ))
                            # get covariance matrix
                            if cov_df is not None:
                                C = cov_df.get((mt, mt), MF=mf, to_numpy=True)

                            # apply sandwich rule
                            if cov_df is not None:
                                if exist:
                                    _merge_or_set(output[resp][mat][za], (mt, mt),
                                                  np.dot(S.T, np.dot(C, S)), mf,
                                                  sum_MFs=sum_MFs)
                                else:
                                    _merge_or_set(output[resp][mat][za], (mt, mt),
                                                  0, mf,
                                                  sum_MFs=sum_MFs)
                            else:
                                if missing_cov == "zero":
                                    _merge_or_set(output[resp][mat][za], (mt, mt),
                                                  0, mf,
                                                  sum_MFs=sum_MFs)
                                elif missing_cov == "raise":
                                    raise ValueError(
                                        f"Missing covariance for ZA={za}, MF={mf}, MT={mt} (diag). "
                                        f"resp={resp}, mat={mat}")
                                elif missing_cov == "assume":
                                    if exist:
                                        C_fb = Sandwich._fallback_cov(
                                            sens.n_groups, missing_cov_rsd, mt, mt,
                                            corr=missing_cov_corr)
                                        _merge_or_set(
                                            output[resp][mat][za], (mt, mt),
                                            np.dot(S.T, np.dot(C_fb, S)),
                                            mf,
                                            sum_MFs=sum_MFs)
                                    else:
                                        _merge_or_set(output[resp][mat][za],
                                                      (mt, mt), 0, mf,
                                                      sum_MFs=sum_MFs)
                                else:
                                    raise ValueError(
                                        f"Invalid missing_cov policy: {missing_cov}"
                                    )

                        # --- covariances
                        cov_combos = list(permutations(map_MF2MT[za][mf], 2))
                        for nm in cov_combos:
                            exist = (resp in sens.responses and mat in sens.materials and
                                     nm[0] in sens.MTs and za in sens.zaid)
                            exist2 = (resp in sens.responses and mat in sens.materials and
                                      nm[1] in sens.MTs and za in sens.zaid)
                            # get covariance matrix
                            if cov_df is not None:
                                C = cov_df.get(nm, MF=mf, to_numpy=True)

                            # get group-wise sensitivity vector
                            if sens_MC:
                                if exist:
                                    S_avg_r, S_rsd_r = sens.get(
                                        [resp], [mat], [nm[0]], [za],
                                        group_order="ascending")
                                else:
                                    S_avg_r = np.zeros((sens.n_groups, ))
                                    S_rsd_r = None

                                if exist2:
                                    S_avg_l, S_rsd_l = sens.get(
                                        [resp], [mat], [nm[1]], [za],
                                        group_order="ascending")
                                else:
                                    S_avg_l = np.zeros((sens.n_groups, ))
                                    S_rsd_l = None

                                if S_rsd_r is not None:
                                    S_r = utils.np2unp(
                                        np.squeeze(S_avg_r),
                                        sigma * np.squeeze(S_rsd_r))
                                else:
                                    S_r = np.squeeze(S_avg_r)

                                if S_rsd_l is not None:
                                    S_l = utils.np2unp(
                                        np.squeeze(S_avg_l),
                                        sigma * np.squeeze(S_rsd_l))
                                else:
                                    S_l = np.squeeze(S_avg_l)
                            else:
                                if exist:
                                    S_r = np.squeeze(
                                        sens.get([resp], [mat], [nm[0]], [za],
                                                 group_order="ascending"))
                                else:
                                    S_r = np.zeros((sens.n_groups, ))

                                if exist2:
                                    S_l = np.squeeze(
                                        sens.get([resp], [mat], [nm[1]], [za],
                                                 group_order="ascending"))
                                else:
                                    S_l = np.zeros((sens.n_groups, ))

                            # apply sandwich rule
                            # if cov_df is not None:
                            #     if exist and exist2:
                            #         output[resp][mat][za][nm] = np.dot(S_r.T, np.dot(C, S_l))
                            #     else:
                            #         output[resp][mat][za][nm] = 0
                            # else:
                            #     output[resp][mat][za][nm] = 0
                            if cov_df is not None:
                                if exist and exist2:
                                    _merge_or_set(output[resp][mat][za], nm,
                                                  np.dot(S_r.T, np.dot(C, S_l)),
                                                  mf,
                                                  sum_MFs=sum_MFs)
                                else:
                                    _merge_or_set(output[resp][mat][za], nm, 0, mf,
                                                  sum_MFs=sum_MFs)
                            else:
                                if missing_cov == "zero":
                                    _merge_or_set(output[resp][mat][za], nm, 0, mf,
                                                  sum_MFs=sum_MFs)
                                elif missing_cov == "raise":
                                    raise ValueError(
                                        f"Missing covariance for ZA={za}, MF={mf}, MTs={nm} (off-diag). "
                                        f"resp={resp}, mat={mat}")
                                elif missing_cov == "assume":
                                    if exist and exist2:
                                        C_fb = Sandwich._fallback_cov(
                                            sens.n_groups, missing_cov_rsd, nm[0], nm[1],
                                            corr=missing_cov_corr)
                                        _merge_or_set(output[resp][mat][za], nm,
                                                      np.dot(S_r.T,
                                                             np.dot(C_fb, S_l)), mf,
                                                      sum_MFs=sum_MFs)
                                    else:
                                        _merge_or_set(output[resp][mat][za], nm, 0, mf,
                                                      sum_MFs=sum_MFs)
                                else:
                                    raise ValueError(
                                        f"Invalid missing_cov policy: {missing_cov}"
                                    )

        # --- convert in pandas.DataFrame
        records = []
        for resp, mats in output.items():
            for mat, zas in mats.items():
                for za, mt_matrix in zas.items():
                    for (mf, mt1, mt2), value in mt_matrix.items():
                        records.append(
                            (resp, mat, za_dict[za], mf, mt1, mt2, value))

        uncertainty = _build_result_frames(
            records,
            ["RESPONSE", "MATERIAL", "ZA", "MF", "MT_row", "MT_col", "value"],
            ["RESPONSE", "MATERIAL", "ZA"],
            include_MF=include_MF,
            sum_MFs=sum_MFs,
        )

        return uncertainty, dict_map

    @staticmethod
    def compute_representativity(sens, sens2, covmat, list_resp, map_MF2MT, za_dict,
                                 sens_MC, sigma=None, sum_MFs=False, include_MF=False,
                                 missing_cov="zero", missing_cov_rsd=0.20,
                                 missing_cov_corr=0.0):
        """
        Compute covariance-weighted representativity between two sensitivities.

        Parameters
        ----------
        sens, sens2 : Sensitivity
            Sensitivity objects to compare.
        covmat : dict[int, Covariance]
            Covariance objects keyed by ZAID.
        list_resp : list[str]
            Responses to process.
        map_MF2MT : dict
            Mapping from ZAID to MF-to-MT selections.
        za_dict : dict[int, str]
            ZAID-to-label mapping used in the output.
        sens_MC : bool
            Whether sensitivity RSDs should be propagated with
            ``uncertainties``.
        sigma : int or float, optional
            Multiplier applied to sensitivity RSDs.
        sum_MFs : bool, optional
            Sum duplicate MT pairs across MF sections.
        include_MF : bool, optional
            Keep MF as an explicit output index level.
        missing_cov : {"zero", "raise", "assume"}, optional
            Policy for missing covariance data.
        missing_cov_rsd : float, optional
            RSD used by the ``"assume"`` missing-covariance policy.
        missing_cov_corr : float, optional
            Off-diagonal correlation used by the ``"assume"`` policy.

        Returns
        -------
        representativity : pandas.DataFrame
            Covariance-weighted representativity matrix.
        dict_map : dict
            Metadata-to-index mapping used during the calculation.
        """

        if sens.reader == 'serpent':
            mat = 'total'
        elif sens.reader == 'eranos':
            mat = 'REACTOR'

        if sens2.reader == 'serpent':
            mat2 = 'total'
        elif sens2.reader == 'eranos':
            mat2 = 'REACTOR'

        # --- apply sandwich rule
        output = {}
        dict_map = {}

        for resp in list_resp:

            output[resp] = {}
            dict_map[resp] = {}
            for key in ['zaid', 'MTs']:
                dict_map[resp][key] = {}

            for iza, za in enumerate(za_dict.keys()):
                if za in covmat.keys():
                    e6_mat_id = covmat[za].mat
                else:
                    e6_mat_id = None

                dict_map[resp]["zaid"][za] = iza
                output[resp][za] = {}

                for mf in map_MF2MT[za].keys():
                    # --- get covariance matrix
                    cov_df = None
                    if za in covmat.keys():
                        if mf in covmat[za].rcov.keys():
                            cov_df = covmat[za]

                    # --- diagonal terms
                    for mt in map_MF2MT[za][mf]:
                        exist = (resp in sens.responses and mat in sens.materials and
                                 mt in sens.MTs and za in sens.zaid)
                        exist2 = (resp in sens2.responses and mat in sens2.materials and
                                  mt in sens2.MTs and za in sens2.zaid)
                        # get group-wise sensitivity vector
                        if sens_MC:
                            if exist:
                                S_avg_r, S_rsd_r = sens.get(
                                    resp=[resp], mat=[mat], MT=[mt], za=[za],
                                    group_order="ascending")
                            else:
                                S_avg_r = np.zeros((sens.n_groups, ))
                                S_rsd_r = None

                            if exist2:
                                S_avg_l, S_rsd_l = sens2.get(
                                    resp=[resp], mat=[mat2], MT=[mt], za=[za],
                                    group_order="ascending")
                            else:
                                S_avg_l = np.zeros((sens.n_groups, ))
                                S_rsd_l = None

                            if S_rsd_r is not None:
                                S_r = utils.np2unp(np.squeeze(S_avg_r),
                                                   sigma * np.squeeze(S_rsd_r))
                            else:
                                S_r = np.squeeze(S_avg_r)

                            if S_rsd_l is not None:
                                S_l = utils.np2unp(np.squeeze(S_avg_l),
                                                   sigma * np.squeeze(S_rsd_l))
                            else:
                                S_l = np.squeeze(S_avg_l)

                        else:
                            if exist:
                                S_r = np.squeeze(
                                    sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za],
                                             group_order="ascending"))
                            else:
                                S_r = np.zeros((sens.n_groups, ))

                            if exist2:
                                S_l = np.squeeze(
                                    sens2.get(resp=[resp], mat=[mat2], MT=[mt],
                                              za=[za], group_order="ascending"))
                            else:
                                S_l = np.zeros((sens.n_groups, ))

                        # get covariance matrix
                        if cov_df is not None:
                            C = cov_df.get((mt, mt), MF=mf, to_numpy=True)

                        # apply sandwich rule
                        if cov_df is not None:
                            if exist and exist2:
                                _merge_or_set(output[resp][za], (mt, mt),
                                              np.dot(S_r.T, np.dot(C, S_l)), mf,
                                              sum_MFs=sum_MFs)
                            else:
                                _merge_or_set(output[resp][za], (mt, mt), 0, mf,
                                              sum_MFs=sum_MFs)
                        else:
                            if missing_cov == "zero":
                                _merge_or_set(output[resp][za], (mt, mt), 0, mf,
                                              sum_MFs=sum_MFs)

                            elif missing_cov == "raise":
                                raise ValueError(
                                    f"Missing covariance for ZA={za}, MF={mf}, MT={mt} "
                                    f"(resp={resp})")

                            elif missing_cov == "assume":
                                if exist and exist2:
                                    C_fb = Sandwich._fallback_cov(
                                        sens.n_groups, missing_cov_rsd, mt, mt,
                                        corr=missing_cov_corr)
                                    _merge_or_set(output[resp][za], (mt, mt),
                                                  np.dot(S_r.T,
                                                         np.dot(C_fb, S_l)), mf,
                                                  sum_MFs=sum_MFs)
                                else:
                                    _merge_or_set(output[resp][za], (mt, mt), 0, mf,
                                                  sum_MFs=sum_MFs)

                            else:
                                raise ValueError(
                                    f"Invalid missing_cov policy: {missing_cov}"
                                )

                    # --- covariances
                    cov_combos = list(permutations(map_MF2MT[za][mf], 2))
                    for nm in cov_combos:
                        exist = (resp in sens.responses and mat in sens.materials and
                                 nm[0] in sens.MTs and za in sens.zaid)
                        exist2 = (resp in sens2.responses and mat in sens2.materials and
                                  nm[1] in sens2.MTs and za in sens2.zaid)
                        # get group-wise sensitivity vector
                        if sens_MC:
                            if exist:
                                S_avg_r, S_rsd_r = sens.get(
                                    [resp], [mat], [nm[0]], [za],
                                    group_order="ascending")
                            else:
                                S_avg_r = np.zeros((sens.n_groups, ))
                                S_rsd_r = None

                            if exist2:
                                S_avg_l, S_rsd_l = sens2.get(
                                    [resp], [mat2], [nm[1]], [za],
                                    group_order="ascending")
                            else:
                                S_avg_l = np.zeros((sens.n_groups, ))
                                S_rsd_l = None

                            if S_rsd_r is not None:
                                S_r = utils.np2unp(np.squeeze(S_avg_r),
                                                   sigma * np.squeeze(S_rsd_r))
                            else:
                                S_r = np.squeeze(S_avg_r)

                            if S_rsd_l is not None:
                                S_l = utils.np2unp(np.squeeze(S_avg_l),
                                                   sigma * np.squeeze(S_rsd_l))
                            else:
                                S_l = np.squeeze(S_avg_l)
                        else:
                            if exist:
                                S_r = np.squeeze(
                                    sens.get([resp], [mat], [nm[0]], [za],
                                             group_order="ascending"))
                            else:
                                S_r = np.zeros((sens.n_groups, ))

                            if exist2:
                                S_l = np.squeeze(
                                    sens2.get([resp], [mat2], [nm[1]], [za],
                                              group_order="ascending"))
                            else:
                                S_l = np.zeros((sens.n_groups, ))

                        # get covariance matrix
                        if cov_df is not None:
                            C = cov_df.get(nm, MF=mf, to_numpy=True)

                        # apply sandwich rule
                        if cov_df is not None:
                            if exist and exist2:
                                _merge_or_set(output[resp][za], nm,
                                              np.dot(S_r.T, np.dot(C, S_l)), mf,
                                              sum_MFs=sum_MFs)
                            else:
                                _merge_or_set(output[resp][za], nm, 0, mf,
                                              sum_MFs=sum_MFs)
                        else:
                            if missing_cov == "zero":
                                _merge_or_set(output[resp][za], nm, 0, mf,
                                              sum_MFs=sum_MFs)

                            elif missing_cov == "raise":
                                raise ValueError(
                                    f"Missing covariance for ZA={za}, MF={mf}, MTs={nm} "
                                    f"(resp={resp})")

                            elif missing_cov == "assume":
                                if exist and exist2:
                                    C_fb = Sandwich._fallback_cov(
                                        sens.n_groups, missing_cov_rsd, nm[0], nm[1],
                                        corr=missing_cov_corr)
                                    _merge_or_set(output[resp][za], nm,
                                                  np.dot(S_r.T,
                                                         np.dot(C_fb, S_l)), mf,
                                                  sum_MFs=sum_MFs)
                                else:
                                    _merge_or_set(output[resp][za], nm, 0, mf,
                                                  sum_MFs=sum_MFs)

                            else:
                                raise ValueError(
                                    f"Invalid missing_cov policy: {missing_cov}"
                                )

        # --- get normalisation coefficients
        za_dict_1 = dict(zip(sens.zaid.keys(), sens.zais.keys()))
        unc1, dict_map1 = Sandwich.compute_uncertainty(
            sens, covmat, list_resp, [mat], map_MF2MT, za_dict_1, sens_MC,
            sigma=sigma, sum_MFs=sum_MFs, include_MF=include_MF,
            missing_cov=missing_cov, missing_cov_rsd=missing_cov_rsd,
            missing_cov_corr=missing_cov_corr)
        za_dict_2 = dict(zip(sens2.zaid.keys(), sens2.zais.keys()))
        unc2, dict_map2 = Sandwich.compute_uncertainty(
            sens2, covmat, list_resp, [mat2], map_MF2MT, za_dict_2, sens_MC,
            sigma=sigma, sum_MFs=sum_MFs, include_MF=include_MF,
            missing_cov=missing_cov, missing_cov_rsd=missing_cov_rsd,
            missing_cov_corr=missing_cov_corr)

        # --- assign normalised output and convert in pandas.DataFrame
        norm = {}
        for resp in list_resp:
            val1 = unc1.loc[(resp, mat)].sum().sum()
            val2 = unc2.loc[(resp, mat2)].sum().sum()
            if hasattr(val1, 'nominal_value'):
                if val1.n < 0:
                    val1 = -unp.sqrt(-val1)
                else:
                    val1 = unp.sqrt(val1)
            else:
                if val1 < 0:
                    val1 = -np.sqrt(-val1)
                else:
                    val1 = np.sqrt(val1)

            if hasattr(val2, 'nominal_value'):
                if val2.n < 0:
                    val2 = -unp.sqrt(-val2)
                else:
                    val2 = unp.sqrt(val2)
            else:
                if val2 < 0:
                    val2 = -np.sqrt(-val2)
                else:
                    val2 = np.sqrt(val2)

            norm[resp] = val1 * val2

        records = []
        for resp, zas in output.items():
            for za, mt_matrix in zas.items():
                for (mf, mt1, mt2), value in mt_matrix.items():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        records.append((resp, za_dict[za], mf, mt1, mt2,
                                        value / norm[resp]))

        representativity = _build_result_frames(
            records,
            ["RESPONSE", "ZA", "MF", "MT_row", "MT_col", "value"],
            ["RESPONSE", "ZA"],
            include_MF=include_MF,
            sum_MFs=sum_MFs,
        )
        dict_map = dict_map

        return representativity, dict_map


class SandwichError(Exception):
    """Custom exception raised for sandwich-formula calculation errors."""

    pass


SandwhichError = SandwichError
sandwhichError = SandwichError
