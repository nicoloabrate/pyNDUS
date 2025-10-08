"""
author: N. Abrate.

file: NDUQ.py

description: GPT and XGPT classes to perform uncertainty quantification for
            Serpent-2 output responses.
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
    import pyNDUS.Sensitivity as Sensitivity
    import pyNDUS.GetCovariance as GetCovariance
    import pyNDUS.utils as utils
except ModuleNotFoundError: # when run as a script
    try:
        from . import Sensitivity, GetCovariance
        from . import utils
    except ImportError:
        import Sensitivity
        import GetCovariance
        import utils


MT2MF = {452: 31, 455: 31, 456: 31}
for mt in range(1, 117):
    MT2MF[mt] = 33

class Sandwich:
    def __init__(self, sens, sens2=None, covmat=None, sigma=2, verbosity=False, 
                 list_resp=None, list_mat=None, list_za=None, list_MTs=None, 
                 list_MFs=None, representativity=False, similarity=False):

        # --- validate input arguments
        if similarity:
            self.calculation_type = "similarity"
            if sens2 is None:
                raise ValueError(f"'sens2' arg is needed to perform representativity calculations!")
        elif representativity:
            if covmat is None:
                raise ValueError(f"'covmat' arg is needed to perform representativity calculations!")
            if sens2 is None:
                raise ValueError(f"'sens2' arg is needed to perform representativity calculations!")
            self.calculation_type = "representativity"
        else:
            if covmat is None:
                raise ValueError(f"'covmat' arg is needed to perform uncertainty calculations!")
            self.calculation_type = "uncertainty"

        if representativity and similarity:
            raise ValueError("Cannot perform 'representativity' and 'similariy' calculation at the same time. Just one argument can be True")

        if not isinstance(sens, Sensitivity):
            raise ValueError(f"'sens' arg must be of type pyNDUS.Sensitivity, not of type {type(sens)}")
        else:
            if sens.sens_rsd is not None:
                sens_MC = True
            else:
                sens_MC = False

        sens2_MC = None
        if sens2 is not None:
            if not isinstance(sens2, Sensitivity):
                raise ValueError(f"'sens2' arg must be of type pyNDUS.Sensitivity, not of type {type(sens2)}")
            else:
                if sens.sens_rsd is not None:
                    sens2_MC = True

        if covmat is not None:
            if not isinstance(covmat, dict):
                raise ValueError(f"'covmat' arg must be of type dict, not of type {type(covmat)}")
            else:
                if len(covmat) == 0:
                    raise ValueError(f"'covmat' dict is empty! Check the dict with the covariances.")
                else:
                    for k, v in covmat.items():
                        if not isinstance(v, GetCovariance):
                            raise ValueError(f"'covmat' items must be of type GetCovariance, not of type {type(covmat)}")
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
                raise ValueError(f"{list_resp} not available in 'Sensitivity' object provided!")
            if sens2 is not None:
                if list_resp not in sens2.responses:
                    raise ValueError(f"{list_resp} not available in 'Sensitivity' object 'sens2' provided!")
            list_resp = [list_resp]
        elif not isinstance(list_resp, list):
            raise ValueError(f"'list_resp' arg must be str or list, not {type(list_resp)}")
        else:
            for resp in list_resp:
                if resp not in sens.responses:
                    raise ValueError(f"{resp} not available in 'Sensitivity' object provided!")
                if sens2 is not None:
                    if resp not in sens2.responses:
                        raise ValueError(f"{resp} not available in 'Sensitivity' object 'sens2' provided!")

        # --- check materials
        if not representativity:
            if list_mat is None:
                list_mat = list(sens.materials.keys())
                if sens2 is not None:
                    list_mat += list(sens2.materials.keys())
                list_mat = list(set(list_mat))
            elif isinstance(list_mat, str):
                if list_mat not in sens.materials.keys():
                    raise ValueError(f"{list_mat} not available in 'Sensitivity' object provided!")
                if sens2 is not None:
                    if list_mat not in sens2.materials.keys():
                        raise ValueError(f"{list_mat} not available in 'Sensitivity' object 'sens2' provided!")
                list_mat = [list_mat]
            elif not isinstance(list_mat, list):
                raise ValueError(f"'list_mat' arg must be str or list, not {type(list_mat)}")
            else:
                for mat in list_mat:
                    if mat not in sens.materials.keys():
                        raise ValueError(f"{mat} not available in 'Sensitivity' object provided!")
                    if sens2 is not None:
                        if mat not in sens2.materials.keys():
                            raise ValueError(f"{mat} not available in 'Sensitivity' object 'sens2' provided!")

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
                raise ValueError(f"{list_za} not available in 'Sensitivity' object provided!")
            if sens2 is not None:
                if list_za not in sens2.zais.keys():
                    raise ValueError(f"{list_za} not available in 'Sensitivity' object 'sens2' provided!")
            list_za = [sens.zais[list_za]]
        elif isinstance(list_za, int):
            if list_za not in sens.zaid.keys():
                raise ValueError(f"{list_za} not available in 'Sensitivity' object provided!")
            if sens2 is not None:
                if list_za not in sens2.zaid.keys():
                    raise ValueError(f"{list_za} not available in 'Sensitivity' object 'sens2' provided!")
            list_za = [sens.zaid[list_za]]
        elif not isinstance(list_za, list):
            raise ValueError(f"'list_za' arg must be str or list, not {type(list_za)}")
        else:
            for zaid in list_za:
                if isinstance(zaid, int):
                    if zaid not in sens.zaid.keys():
                        raise ValueError(f"{zaid} not available in 'Sensitivity' object provided!")
                    if sens2 is not None:
                        if zaid not in sens2.zaid.keys():
                            raise ValueError(f"{zaid} not available in 'Sensitivity' object 'sens2' provided!")
                elif isinstance(zaid, str):
                    if zaid not in sens.zais.keys():
                        raise ValueError(f"{zaid} not available in 'Sensitivity' object provided!")
                    if sens2 is not None:
                        if zaid not in sens2.zais.keys():
                            raise ValueError(f"{zaid} not available in 'Sensitivity' object 'sens2' provided!")

        # --- check MTs
        if list_MTs is None:
            get_MTs = True
        elif isinstance(list_MTs, int):
            list_MTs = [list_MTs]
        elif isinstance(list_MTs, list):
            get_MTs = False
        else:
            raise ValueError(f"list_MTs argument must be of type 'list', not {type(list_MTs)}")

        if not similarity and not representativity:
            if not is_covmat:
                raise sandwhichError("covmat optional argument needed to get the uncertainty propagated with the sandwich formula!")

        # --- enforce consistent set of MTs for any isotope
        sens_MTs = {}
        rcov_MTs = {}
        if is_covmat:
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
                else:
                    map_MF2MT[za] = {}
                    map_MF2MT[za]["errorr33"] = []

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

                        for mt in intersection:
                            mf = f"errorr{MT2MF[mt]}"
                            map_MF2MT[za][mf].append(mt)

                else:

                    intersection = list(set(list_MTs) & set(covMTs) & set(sens_MTs[za]))
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

                    for mt in intersection:
                        mf = f"errorr{MT2MF[mt]}"
                        map_MF2MT[za][mf].append(mt)

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

        if len(za_dict) == 0:
            raise SandwichError("No valid ZAIDs found. Check that there is a non-null intersection between ZAIDs provided in the sensitivity and covariance (if any) objects.")
        else:
            self.za = za_dict

        # --- assign output
        if representativity:
            representativity, dict_map = self.compute_representativity(sens, sens2, covmat, list_resp,
                                                                       map_MF2MT, self.za, self.sens_MC, sigma=self.sigma)
            self.representativity = representativity
            self.dict_map = dict_map
        elif similarity:
            similarity, dict_map = self.compute_similarity(sens, sens2, list_resp, self.MTs, self.za, self.sens_MC, sigma=self.sigma)
            self.similarity = similarity
            self.dict_map = dict_map
        else:
            uncertainty, dict_map = self.compute_uncertainty(sens, covmat, list_resp, list_mat,
                                                        map_MF2MT, self.za, self.sens_MC, sigma=self.sigma)
            self.uncertainty = uncertainty
            self.dict_map = dict_map


    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):

        if self.sens_MC:
            if not isinstance(value, (int, float)):
                raise ValueError(f"'sigma' must be a number, not {type(value)}")
            if value < 0:
                raise ValueError(f"'sigma' must be positive, not {value}")
            self._sigma = value
        else:
            self._sigma = None

    @property
    def za(self):
        return self._za

    @za.setter
    def za(self, value):
        if not isinstance(value, dict):
            raise ValueError(f"'za' must be a dict, not {type(value)}")
        for za, zais in value.items():
            if not isinstance(za, int):
                raise ValueError(f"keys of 'za' dict must be int, not {type(za)}")
            if not isinstance(zais, str):
                raise ValueError(f"values of 'za' dict must be str, not {type(zais)}")
        self._za = value

    @staticmethod
    def compute_similarity(sens, sens2, list_resp, list_MTs, za_dict, sens_MC, sigma=None):

        if sens.reader == 'serpent':
            mat = 'total'
        elif sens.reader == 'eranos':
            mat = 'REACTOR'

        if sens2.reader == 'serpent':
            mat2 = 'total'
        elif sens2.reader == 'eranos':
            mat2 = 'REACTOR'

        # --- apply scalar product (i.e., sandwich rule with unitary covariance)
        output = {}
        dict_map = {}
        norm1 = {}
        norm2 = {}

        for resp in list_resp:

            output[resp] = {}
            dict_map[resp] = {}
            norm1[resp] = 0.0
            norm2[resp] = 0.0

            for key in ['zaid', 'MTs']:
                dict_map[resp][key] = {}

            for iza, za in enumerate(za_dict.keys()):

                dict_map[resp]["zaid"][za] = iza
                output[resp][za] = {}

                # --- diagonal terms 
                for mt in list_MTs[za]:
                    exist = resp in sens.responses and mat in sens.materials.keys() and mt in sens.MTs.keys() and za in sens.zaid.keys()
                    exist2 = resp in sens2.responses and mat in sens2.materials.keys() and mt in sens2.MTs.keys() and za in sens2.zaid.keys()
                    # get group-wise sensitivity vector
                    if sens_MC:
                        if exist:
                            S_avg_r, S_rsd_r = sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za], group_order="ascending")
                        else:
                            S_avg_r = np.zeros((sens.n_groups, ))
                            S_rsd_r = None

                        if exist2:
                            S_avg_l, S_rsd_l = sens2.get(resp=[resp], mat=[mat2], MT=[mt], za=[za], group_order="ascending")
                        else:
                            S_avg_l = np.zeros((sens.n_groups, ))
                            S_rsd_l = None

                        if S_rsd_r is not None:
                            S_r = utils.np2unp(np.squeeze(S_avg_r), sigma * np.squeeze(S_rsd_r))
                            norm1[resp] += S_r.dot(S_r)
                        else:
                            S_r = np.squeeze(S_avg_r)

                        if S_rsd_l is not None:
                            S_l = utils.np2unp(np.squeeze(S_avg_l), sigma * np.squeeze(S_rsd_l))
                            norm2[resp] += S_l.dot(S_l)
                        else:
                            S_l = np.squeeze(S_avg_l)

                    else:
                        if exist:
                            S_r = np.squeeze(sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za], group_order="ascending"))
                            norm1[resp] += S_r.dot(S_r)
                        else:
                            S_r = np.zeros((sens.n_groups, ))

                        if exist2:
                            S_l = np.squeeze(sens2.get(resp=[resp], mat=[mat2], MT=[mt], za=[za], group_order="ascending"))
                            norm2[resp] += S_l.dot(S_l)
                        else:
                            S_l = np.zeros((sens.n_groups, ))

                    # apply sandwich rule
                    if exist and exist2:
                        output[resp][za][(mt, mt)] = np.dot(S_r.T, S_l)
                    else:
                        output[resp][za][(mt, mt)] = 0

        # --- assign normalised output and convert in pandas.DataFrame
        records = []
        for resp, zas in output.items():
            for za, mt_matrix in zas.items():
                for (mt1, mt2), value in mt_matrix.items():
                    records.append((resp, za_dict[za], mt1, mt2, value))

        df = pd.DataFrame(records, columns=["RESPONSE", "ZA", "MT_row", "MT_col", "value"])
        df.set_index(["RESPONSE", "ZA", "MT_row", "MT_col"], inplace=True)
        df_matrix = df["value"].unstack("MT_col").fillna(0)

        for resp in list_resp:
            val1 = norm1[resp]
            val2 = norm2[resp]
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

            idx = pd.IndexSlice
            df_matrix.loc[idx[resp, :], :] /= (val1 * val2)

        similarity = df_matrix
        dict_map = dict_map

        return similarity, dict_map

    @staticmethod
    def compute_uncertainty(sens, covmat, list_resp, list_mat, map_MF2MT,
                            za_dict, sens_MC, sigma=None):
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
                        if mf == "errorr34" or mf == 'errorr35':
                            # SANDY object does not provide MF=35 in the same formats of MF=31 and MF=33
                            continue

                        # --- get covariance matrix
                        cov_df = None
                        if za in covmat.keys():
                            if mf in covmat[za].rcov.keys():
                                cov_df = covmat[za]

                        # --- diagonal terms 
                        for mt in map_MF2MT[za][mf]:
                            exist = resp in sens.responses and mat in sens.materials.keys() and mt in sens.MTs.keys() and za in sens.zaid.keys()
                            # get group-wise sensitivity vector
                            if sens_MC:
                                if exist:
                                    S_avg, S_rsd = sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za], group_order="ascending")
                                else:
                                    S_avg = np.zeros((sens.n_groups, ))
                                    S_rsd = None

                                if S_rsd is not None:
                                    S = utils.np2unp(np.squeeze(S_avg), sigma * np.squeeze(S_rsd))
                                else:
                                    S = np.squeeze(S_avg)

                            else:
                                if exist:
                                    S = np.squeeze(sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za], group_order="ascending"))
                                else:
                                    S = np.zeros((sens.n_groups, ))
                            # get covariance matrix
                            if cov_df is not None:
                                C = cov_df.get((mt, mt), MF=mf, to_numpy=True)

                            # apply sandwich rule
                            if cov_df is not None:
                                if exist:
                                    output[resp][mat][za][(mt, mt)] = np.dot(S.T, np.dot(C, S)) 
                                else:
                                    output[resp][mat][za][(mt, mt)] = 0
                            else:
                                output[resp][mat][za][(mt, mt)] = 0

                        # --- covariances
                        cov_combos = list(permutations(map_MF2MT[za][mf], 2))
                        for nm in cov_combos:
                            exist = resp in sens.responses and mat in sens.materials.keys() and nm[0] in sens.MTs.keys() and za in sens.zaid.keys()
                            exist2 = resp in sens.responses and mat in sens.materials.keys() and nm[1] in sens.MTs.keys() and za in sens.zaid.keys()
                            # get covariance matrix
                            if cov_df is not None:
                                C = cov_df.get(nm, MF=mf, to_numpy=True)

                            # get group-wise sensitivity vector
                            if sens_MC:
                                if exist:
                                   S_avg_r, S_rsd_r = sens.get([resp], [mat], [nm[0]], [za], group_order="ascending")
                                else:
                                    S_avg_r = np.zeros((sens.n_groups, ))
                                    S_rsd_r = None

                                if exist2:
                                    S_avg_l, S_rsd_l = sens.get([resp], [mat], [nm[1]], [za], group_order="ascending")
                                else:
                                    S_avg_l = np.zeros((sens.n_groups, ))
                                    S_rsd_l = None

                                if S_rsd_r is not None:
                                    S_r = utils.np2unp(np.squeeze(S_avg_r), sigma * np.squeeze(S_rsd_r))
                                else:
                                    S_r = np.squeeze(S_avg_r)

                                if S_rsd_l is not None:
                                    S_l = utils.np2unp(np.squeeze(S_avg_l), sigma * np.squeeze(S_rsd_l))
                                else:
                                    S_l = np.squeeze(S_avg_l)
                            else:
                                if exist:
                                    S_r = np.squeeze(sens.get([resp], [mat], [nm[0]], [za], group_order="ascending"))
                                else:
                                    S_r = np.zeros((sens.n_groups, ))

                                if exist2:
                                    S_l = np.squeeze(sens.get([resp], [mat], [nm[1]], [za], group_order="ascending"))
                                else:
                                    S_l = np.zeros((sens.n_groups, ))

                            # apply sandwich rule
                            if cov_df is not None:
                                if exist and exist2:
                                    output[resp][mat][za][nm] = np.dot(S_r.T, np.dot(C, S_l))
                                else:
                                    output[resp][mat][za][nm] = 0
                            else:
                                output[resp][mat][za][nm] = 0

        # --- convert in pandas.DataFrame
        records = []
        for resp, mats in output.items():
            for mat, zas in mats.items():
                for za, mt_matrix in zas.items():
                    for (mt1, mt2), value in mt_matrix.items():
                        records.append((resp, mat, za_dict[za], mt1, mt2, value))

        df = pd.DataFrame(records, columns=["RESPONSE", "MATERIAL", "ZA", "MT_row", "MT_col", "value"])
        df.set_index(["RESPONSE", "MATERIAL", "ZA", "MT_row", "MT_col"], inplace=True)
        df_matrix = df["value"].unstack("MT_col")

        uncertainty = df_matrix

        return uncertainty, dict_map

    @staticmethod
    def compute_representativity(sens, sens2, covmat, list_resp, map_MF2MT,
                                  za_dict, sens_MC, sigma=None):

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
                    if mf == "errorr34" or mf == 'errorr35':
                        # SANDY object does not provide MF=35 in the same formats of MF=31 and MF=33
                        continue

                    # --- get covariance matrix
                    cov_df = None
                    if za in covmat.keys():
                        if mf in covmat[za].rcov.keys():
                            cov_df = covmat[za]

                    # --- diagonal terms 
                    for mt in map_MF2MT[za][mf]:
                        exist = resp in sens.responses and mat in sens.materials.keys() and mt in sens.MTs.keys() and za in sens.zaid.keys()
                        exist2 = resp in sens2.responses and mat in sens2.materials.keys() and mt in sens2.MTs.keys() and za in sens2.zaid.keys()
                        # get group-wise sensitivity vector
                        if sens_MC:
                            if exist:
                                S_avg_r, S_rsd_r = sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za], group_order="ascending")
                            else:
                                S_avg_r = np.zeros((sens.n_groups, ))
                                S_rsd_r = None

                            if exist2:
                                S_avg_l, S_rsd_l = sens2.get(resp=[resp], mat=[mat2], MT=[mt], za=[za], group_order="ascending")
                            else:
                                S_avg_l = np.zeros((sens.n_groups, ))
                                S_rsd_l = None

                            if S_rsd_r is not None:
                                S_r = utils.np2unp(np.squeeze(S_avg_r), sigma * np.squeeze(S_rsd_r))
                            else:
                                S_r = np.squeeze(S_avg_r)

                            if S_rsd_l is not None:
                                S_l = utils.np2unp(np.squeeze(S_avg_l), sigma * np.squeeze(S_rsd_l))
                            else:
                                S_l = np.squeeze(S_avg_l)

                        else:
                            if exist:
                                S_r = np.squeeze(sens.get(resp=[resp], mat=[mat], MT=[mt], za=[za], group_order="ascending"))
                            else:
                                S_r = np.zeros((sens.n_groups, ))

                            if exist2:
                                S_l = np.squeeze(sens2.get(resp=[resp], mat=[mat2], MT=[mt], za=[za], group_order="ascending"))
                            else:
                                S_l = np.zeros((sens.n_groups, ))

                        # get covariance matrix
                        if cov_df is not None:
                            C = cov_df.get((mt, mt), MF=mf, to_numpy=True)

                        # apply sandwich rule
                        if cov_df is not None:
                            if exist and exist2:
                                output[resp][za][(mt, mt)] = np.dot(S_r.T, np.dot(C, S_l)) 
                            else:
                                output[resp][za][(mt, mt)] = 0
                        else:
                            output[resp][za][(mt, mt)] = 0

                    # --- covariances
                    cov_combos = list(permutations(map_MF2MT[za][mf], 2))
                    for nm in cov_combos:
                        exist = resp in sens.responses and mat in sens.materials.keys() and nm[0] in sens.MTs.keys() and za in sens.zaid.keys()
                        exist2 = resp in sens2.responses and mat in sens2.materials.keys() and nm[1] in sens2.MTs.keys() and za in sens2.zaid.keys()
                        # get group-wise sensitivity vector
                        if sens_MC:
                            if exist:
                                S_avg_r, S_rsd_r = sens.get([resp], [mat], [nm[0]], [za], group_order="ascending")
                            else:
                                S_avg_r = np.zeros((sens.n_groups, ))
                                S_rsd_r = None

                            if exist2:
                                S_avg_l, S_rsd_l = sens2.get([resp], [mat2], [nm[1]], [za], group_order="ascending")
                            else:
                                S_avg_l = np.zeros((sens.n_groups, ))
                                S_rsd_l = None

                            if S_rsd_r is not None:
                                S_r = utils.np2unp(np.squeeze(S_avg_r), sigma * np.squeeze(S_rsd_r))
                            else:
                                S_r = np.squeeze(S_avg_r)

                            if S_rsd_l is not None:
                                S_l = utils.np2unp(np.squeeze(S_avg_l), sigma * np.squeeze(S_rsd_l))
                            else:
                                S_l = np.squeeze(S_avg_l)
                        else:
                            if exist:
                                S_r = np.squeeze(sens.get([resp], [mat], [nm[0]], [za], group_order="ascending"))
                            else:
                                S_r = np.zeros((sens.n_groups, ))

                            if exist2:
                                S_l = np.squeeze(sens2.get([resp], [mat2], [nm[1]], [za], group_order="ascending"))
                            else:
                                S_l = np.zeros((sens.n_groups, ))

                        # get covariance matrix
                        if cov_df is not None:
                            C = cov_df.get(nm, MF=mf, to_numpy=True)

                        # apply sandwich rule
                        if cov_df is not None:
                            if exist and exist2:
                                output[resp][za][nm] = np.dot(S_r.T, np.dot(C, S_l))
                            else:
                                output[resp][za][nm] = 0
                        else:
                            output[resp][za][nm] = 0

        # --- get normalisation coefficients
        za_dict_1 = dict(zip(sens.zaid.keys(), sens.zais.keys()))
        unc1, dict_map1 = Sandwich.compute_uncertainty(sens, covmat, list_resp, [mat], map_MF2MT,
                                                        za_dict_1, sens_MC, sigma=sigma)
        za_dict_2 = dict(zip(sens2.zaid.keys(), sens2.zais.keys()))
        unc2, dict_map2 = Sandwich.compute_uncertainty(sens2, covmat, list_resp, [mat2], map_MF2MT,
                                                        za_dict_2, sens_MC, sigma=sigma)

        # --- assign normalised output and convert in pandas.DataFrame
        records = []
        for resp, zas in output.items():
            for za, mt_matrix in zas.items():
                for (mt1, mt2), value in mt_matrix.items():
                    records.append((resp, za_dict[za], mt1, mt2, value))

        df = pd.DataFrame(records, columns=["RESPONSE", "ZA", "MT_row", "MT_col", "value"])
        df.set_index(["RESPONSE", "ZA", "MT_row", "MT_col"], inplace=True)
        df_matrix = df["value"].unstack("MT_col").fillna(0)

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

            idx = pd.IndexSlice
            df_matrix.loc[idx[resp, :], :] /= (val1 * val2)

        representativity = df_matrix
        dict_map = dict_map

        return representativity, dict_map


class SandwhichError(Exception):
    pass

