"""
Author: N. Abrate.

File: GetCovariance.py

Description: Class calling the SANDY and NJOY codes to produce and extract
             multi-group relative covariance matrices
"""
import io
import os
import sys
import sandy
import socket
import numpy as np
import pandas as pd
import shutil as sh
from collections.abc import Iterable
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
try:
    import pyNDUS.utils as utils
except ModuleNotFoundError: # when run as a script
    try:
        from . import utils
    except ImportError:
        import utils

sandy_ver = sandy.__version__

class GetCovariance():
    """
    Class to process and extract multi-group relative covariance matrices for nuclear data
    using NJOY and SANDY codes, respectively.

    This class manages the workflow for:
      - Validating user input for nuclide, temperature, group structure, energy grid, library, and working directory.
      - Creating the necessary directory structure for storing ENDF-6 and ERRORR formatted files (considering different
        libraries and energy group structures).
      - Downloading or reading ENDF-6 formatted files for the requested nuclide and library.
      - Running NJOY (via SANDY) to produce multi-group covariance matrices if not already available.
      - Extracting and storing covariance data in a structured way for further application.
      - Providing properties for accessing key attributes (e.g., ZAID, temperature, MAT, covariance matrices).
      - Plotting covariance or correlation matrices.

    Parameters
    ----------
    zaid : int
        ZAID (Z*1000 + A if Z < 10 or Z*100+A) identifier of the nuclide.
    temperature : float, optional
        Temperature in Kelvin (default: 300).
    group_structure : iterable, optional
        Energy group structure (list, numpy array, tuple etc.).
    egridname : str, optional
        Name of the energy grid (default: None).
    lib : str, optional
        Nuclear data library to use (default: "endfb_80").
    cwd : str or pathlib.Path, optional
        Working directory for storing files (default: current directory).

    Attributes
    ----------
    zaid : int
        ZAID of the nuclide.
    zais : str
        String identifier for the nuclide.
    temperature : float
        Temperature in Kelvin for processing the ENDF-6 file with the BROADR module of NJOY.
    group_structure : iterable
        Energy group structure.
    egridname : str
        Name of the energy grid.
    library : str
        Nuclear data library.
    path : pathlib.Path
        Working directory for storing files.
    MFs2MTs : dict
        Mapping from MF numbers to lists of MT numbers (excluding the header, i.e. MT=451).
    mat : int
        MAT number for the nuclide.
    rcov : dict
        Dictionary with extracted covariance matrices in pandas.DataFrame objects.
        Available MF:
        - MF=31 for neutron multiplicities
        - MF=33 for cross sections
    """
    def __init__(self, zaid, temperature=300, group_structure=None, egridname=None,
                 lib="endfb_80", process_resonances=True, author=None, njoy_ver=None, cwd=None):
        """
        Initialize the GetCovariance object, create necessary directories, and extract covariance matrices.

        Parameters
        ----------
        zaid : int
            ZAID (Z*1000 + A) identifier of the nuclide.
        temperature : float, optional
            Temperature in Kelvin (default: 300).
        group_structure : iterable, optional
            Energy group structure.
        egridname : str, optional
            Name of the energy grid.
        lib : str, optional
            Nuclear data library to use.
        process_resonances: bool, optional
            Flag for processing for resonance parameter covariances
        author : str, optional
            Author name for logging.
        njoy_ver : str, optional
            NJOY version for logging.
        cwd : str or pathlib.Path, optional
            Working directory for storing files.
        """
        # --- input validation
        self.zaid = zaid
        self.zais = utils.zaid2zais(self.zaid)
        self.temperature = temperature
        self.group_structure = group_structure
        self.egridname = egridname
        self.library = lib
        self.path = cwd

        if author is None:
            author = "unknown"
        elif not isinstance(author, str):
            raise ValueError(f"Author arg must be of type str, not of type {type(author)}")

        if njoy_ver is None:
            njoy_ver = "unknown"
        elif not isinstance(njoy_ver, str):
            raise ValueError(f"njoy_ver arg must be of type str, not of type {type(njoy_ver)}")

        # --- create directories
        if not Path.exists(self.path.joinpath(lib)):
            os.mkdir(self.path.joinpath(lib))

        cwd = self.path.joinpath(lib)

        # --- create folder for ENDF-6 formatted files
        subdir = 'endf6'

        if not Path.exists(cwd.joinpath(subdir)):
            os.mkdir(cwd.joinpath(subdir))

        e6dir = cwd.joinpath(subdir)

        if not Path.exists(cwd.joinpath(egridname)):
            os.mkdir(cwd.joinpath(egridname))

        cwd = cwd.joinpath(egridname)

        # --- create folder for ERRORR formatted files
        for subdir in ['errorr']:
            if not Path.exists(cwd.joinpath(subdir)):
                os.mkdir(cwd.joinpath(subdir))

        # --- get ENDF-6 formatted tape
        endf6_name = f'{self.zais}.endf'
        endf6_path = e6dir.joinpath(endf6_name)

        if Path.exists(endf6_path):
            endf6_tape = sandy.Endf6.from_file(endf6_path)
            run_errorr = False
        else:
            endf6_tape = sandy.get_endf6_file(lib, "xs", zaid)
            endf6_tape.to_file(endf6_path)
            run_errorr = True

        # --- get covariance matrix
        errorr_name = f'{self.zais}_{temperature:g}K.errorr'
        errorr_exists = False
        for mf in [31, 33, 34, 35]:
            if Path.exists(cwd.joinpath('errorr', f"{errorr_name}{mf}")):
                errorr_exists = True
                break

        if errorr_exists and not run_errorr:
            errorr_out = {}
            for mf in [31, 33, 34, 35]:
                if cwd.joinpath('errorr', f"{errorr_name}{mf}").exists():
                    errorr_path = cwd.joinpath('errorr', f"{errorr_name}{mf}")
                    errorr_out[f"errorr{mf}"] = sandy.Errorr.from_file(errorr_path)
        else:
            errorr_path = cwd.joinpath('errorr', f"{errorr_name}")
            errorr_out = GetCovariance.sandy_calls_errorr(endf6_tape, zaid, temperature,
                                                          group_structure, egridname,
                                                          errorr_path, process_resonances,
                                                          lib, author, njoy_ver)

        self.MFs2MTs = errorr_out
        self.mat = errorr_out
        self.rcov =  errorr_out

    @property
    def temperature(self):
        """
        Temperature in Kelvin for processing the ENDF-6 file.

        Returns
        -------
        float
            Temperature in Kelvin.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """
        Set the temperature attribute with validation.

        Parameters
        ----------
        value : float or int
            Temperature in Kelvin.

        Raises
        ------
        ValueError
            If value is not a positive number.
        """
        if not isinstance(value, int) and not isinstance(value, float):
            raise ValueError(f"Expected type int or float instead of type {type(value)} for arg 'temperature'")
        elif value <= 0:
            raise ValueError("Temperature must be >0!")
        else:
            self._temperature = value

    @property
    def zaid(self):
        """
        ZAID (Z*1000 + A) identifier of the nuclide.

        Returns
        -------
        int
            ZAID value.
        """
        return self._zaid

    @zaid.setter
    def zaid(self, value):
        """
        Set the ZAID attribute with validation.

        Parameters
        ----------
        value : int
            ZAID value.

        Raises
        ------
        ValueError
            If value is not a positive integer.
        """
        if value <= 0:
            raise ValueError("ZAid must be >0!")
        elif not isinstance(value, int):
            raise ValueError(f"Expected type int instead of type {type(value)} for arg 'ZAid'")

        self._zaid = value

    @property
    def group_structure(self):
        """
        Energy group structure.

        Returns
        -------
        iterable
            Energy group structure.
        """
        return self._group_structure

    @group_structure.setter
    def group_structure(self, value):
        """
        Set the energy group structure with validation.

        Parameters
        ----------
        value : iterable
            Energy group structure.

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
    def egridname(self):
        """
        Name of the energy grid.

        Returns
        -------
        str
            Energy grid name.
        """
        return self._egridname

    @egridname.setter
    def egridname(self, value):
        """
        Set the energy grid name with validation.

        Parameters
        ----------
        value : str
            Energy grid name.

        Raises
        ------
        ValueError
            If value is not a string.
        """
        if value is None:
            self._egridname = "sandy_default_energy_grid"
        elif not isinstance(value, str):
            raise ValueError(f"Expected 'str' instead of type {type(value)} for arg 'egridname'")
        else:
            self._egridname = value

    @property
    def library(self):
        """
        Nuclear data library name.

        Returns
        -------
        str
            Library name.
        """
        return self._library

    @library.setter
    def library(self, value):
        """
        Set the nuclear data library name with validation.

        Parameters
        ----------
        value : str
            Library name.

        Raises
        ------
        ValueError
            If value is not a string.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected 'str' instead of type {type(value)} for arg 'lib'")
        else:
            self._library = value

    @property
    def path(self):
        """
        Working directory for storing files.

        Returns
        -------
        pathlib.Path
            Path to working directory.
        """
        return self._path

    @path.setter
    def path(self, cwd):
        """
        Set the working directory with validation.

        Parameters
        ----------
        cwd : str or pathlib.Path
            Working directory.

        Raises
        ------
        ValueError
            If cwd is not a string or Path.
        """
        if cwd is None:
            self._path = Path(__file__).resolve().parent
        elif isinstance(cwd, str):
            self._path = Path(cwd)
        elif isinstance(cwd, Path):
            self._path = cwd
        else:
            raise ValueError(f"Expected 'str' or 'Path' types instead of type {type(cwd)} for arg 'cwd'")

    @property
    def zais(self):
        """
        String identifier for the nuclide.

        Returns
        -------
        str
            ZAIS string.
        """
        return self._zais

    @zais.setter
    def zais(self, value):
        """
        Set the ZAIS string with validation.

        Parameters
        ----------
        value : str
            ZAIS string.

        Raises
        ------
        ValueError
            If value is not a string.
        """
        if not isinstance(value, str):
            raise ValueError("ZAis must be a string!")
        self._zais = value

    @property
    def MFs2MTs(self):
        """
        Mapping from MF numbers to lists of MT numbers (excluding MT=451).

        Returns
        -------
        dict
            Mapping of MF to MT list.
        """
        return self._MFs2MTs

    @MFs2MTs.setter
    def MFs2MTs(self, rcov):
        """
        Set the MF-to-MT mapping from covariance data.

        Parameters
        ----------
        rcov : dict
            Dictionary of ERRORR objects.

        Sets
        ----
        self._MFs2MTs : dict
            Mapping from MF to MT list.
        """
        map_MF2MT = {}

        for mf in rcov.keys():
            map_MF2MT[mf] = rcov[mf].mt
            # remove the heading MT (no covariance data)
            map_MF2MT[mf].remove(451)

        self._MFs2MTs = map_MF2MT

    @property
    def mat(self):
        """
        MAT number for the nuclide.

        Returns
        -------
        int
            MAT number.
        """
        return self._mat

    @mat.setter
    def mat(self, rcov):
        """
        Set the MAT number from covariance data.

        Parameters
        ----------
        rcov : dict
            Dictionary of ERRORR objects.

        Raises
        ------
        GetCovarianceError
            If multiple MAT numbers are found.
        """
        mat = 0

        for imf, mf in enumerate(rcov.keys()):

            if len(rcov[mf].mat) != 1:
                raise GetCovarianceError(f"Cannot handle an ERRORR file with more MAT numbers!")

            if imf == 0:
                mat = rcov[mf].mat[0]
            elif mat != rcov[mf].mat[0]:
                raise GetCovarianceError(f"MF={mf} contains a different MAT!")

        self._mat = mat

    @property
    def rcov(self):
        """
        Dictionary with extracted covariance matrices.

        Returns
        -------
        dict
            Covariance matrices as pandas.DataFrame objects.
        """
        return self._rcov

    @rcov.setter
    def rcov(self, rcov):
        """
        Set the covariance matrices from ERRORR objects.

        Parameters
        ----------
        rcov : dict
            Dictionary of ERRORR objects.
        """
        out = {}
        for mf in rcov.keys():
            if mf in ['errorr31', 'errorr33']:
                out[mf] = rcov[mf].get_cov().data

        self._rcov = out

    def get(self, MT, MF=None, to_numpy=False):
        """
        Extract covariance matrix or submatrix for specified MT(s) and MF(s).

        Parameters
        ----------
        MT : int, tuple, or list
            MT number(s) or tuple of two MTs for off-diagonal covariance, e.g. (MT1, MT2) or [MT1, MT2].
            If a tuple is provided, it must contain exactly two MTs, and it will return the covariance between them.
            If a list is provided, it will return the full covariance matrix for all specified MTs, e.g. MT1-MT1, 
            MT1-MT2, MT2-MT1, MT2-MT2.
        MF : int or str, optional
            MF number or 'errorr<MF>' string. If None, all available MFs are used.
        to_numpy : bool, optional
            If True, return as numpy array. If False, return as pandas.DataFrame.

        Returns
        -------
        out_cov : pandas.DataFrame or np.ndarray
            Covariance matrix or submatrix.
        """
        # --- input validation
        # TODO ADD tuple arg for sub-matrix
        single_cov = False
        if isinstance(MT, tuple):
            single_cov = True
            MT = list(MT)
        elif isinstance(MT, int):
            MT = [MT]
        elif not isinstance(MT, list):
            raise ValueError(f"MT must be of type int or list, not {type(MT)}")

        list_MF = []
        if MF is None:
            map_MF_2_MT = {}
            for MF, MTs_in_MF in self.MFs2MTs.items():
                # skipping MF=34 and MF=35
                if MF == 'errorr34' or MF == 'errorr35':
                    continue

                for val in MT:
                    if val in MTs_in_MF:
                        if MF not in list_MF:
                            list_MF.append(MF)
                        if MF not in map_MF_2_MT.keys():
                            map_MF_2_MT[MF] = []

                        map_MF_2_MT[MF].append(val)

        else:
            map_MF_2_MT = {}
            if isinstance(MF, int):
                MF = f'errorr{MF}'
            elif not isinstance(MF, str):
                raise ValueError(f"MF must be of type int or str, not {type(MF)}")
            elif 'errorr' not in MF:
                raise ValueError("If MF is str, it must be 'errorr<MF>'")

            list_MF.append(MF)

            map_MF_2_MT[MF] = MT

        # checl existence of requested MT
        for val in MT:
            exists = False
            for MTs in self.MFs2MTs.values():
                if val in MTs:
                    exists = True
                    break
            if not exists:
                raise ValueError(f"MT={val} not available in covariance matrix.")

        if single_cov:
            if len(list_MF) != 1:
                raise ValueError("Cannot get covariance between different MF sections!")

        # --- get covariance
        if not single_cov:
            dict_df = {}

        for iMF, MF in enumerate(list_MF):
            if single_cov:
                out_cov = self.rcov[MF].loc[(self.mat, MT[0]), (self.mat, MT[1])]
            else:
                dict_df[MF] = self.rcov[MF].loc[(self.mat, map_MF_2_MT[MF]), (self.mat, map_MF_2_MT[MF])]

        if not single_cov:
            idx = {}
            col = {}
            for ikey, key in enumerate(dict_df.keys()):
                idx[key] = dict_df[key].index
                col[key] = dict_df[key].columns
                if ikey == 0:
                    index = dict_df[key].index
                    colum = dict_df[key].columns
                else:
                    index = index.union(dict_df[key].index)
                    colum = colum.union(dict_df[key].columns)
            # allocation of the empty dataframe
            out_cov = pd.DataFrame(np.zeros((len(index), len(colum))), 
                                    index=index, columns=colum)
            # merge the dataframes
            for key in dict_df.keys():
                out_cov.loc[idx[key], col[key]] = dict_df[key].values

        if to_numpy:
            return out_cov.to_numpy()
        else:
            return out_cov

    @staticmethod
    def sandy_calls_errorr(endf6_tape, zaid, temperature, group_structure, egridname, errorr_path, 
                           process_resonances, lib, author, njoy_ver):
        """
        Run the ERRORR module of NJOY via SANDY and save output files.

        Parameters
        ----------
        endf6_tape : sandy.Endf6
            ENDF-6 formatted tape.
        zaid : int
            ZAID identifier.
        temperature : float
            Temperature in Kelvin.
        group_structure : iterable
            Energy group structure.
        egridname : str
            Name of the energy grid.
        errorr_path : pathlib.Path
            Path to save ERRORR output.
        lib : str
            Nuclear data library.
        author : str
            Author name for logging.
        njoy_ver : str
            NJOY version for logging.

        Returns
        -------
        errorr : dict
            Dictionary of ERRORR objects.
        """
        if process_resonances:
            irespr = 1
        else:
            irespr = 0

        njoy_inp = endf6_tape.get_errorr(groupr_kws=dict(ek=group_structure), errorr_kws=dict(ek=group_structure, irespr=irespr), 
                                         dryrun=True, temperature=temperature)

        base_path = errorr_path.parent.parent
        if not Path.exists(base_path.joinpath("njoy_input")):
            os.mkdir(base_path.joinpath("njoy_input"))

        with open(base_path.joinpath("njoy_input", errorr_path.name.replace('errorr', 'input')), "w") as f:
            f.write(njoy_inp)

        # redirect stdout to a file
        if not Path.exists(base_path.joinpath("njoy_output")):
            os.mkdir(base_path.joinpath("njoy_output"))
        
        out_file_path = base_path.joinpath("njoy_output", errorr_path.name.replace('.errorr', '.log'))

        GetCovariance.write_log_header(out_file_path, author, njoy_ver)
        with open(out_file_path, "a") as out_file:
            saved_stdout = os.dup(1)  # save current stdout file descriptor
            try:
                # Redirect stdout to the log file
                os.dup2(out_file.fileno(), 1)

                # Call the method that produces output
                errorr = endf6_tape.get_errorr(
                    groupr_kws=dict(ek=group_structure),
                    errorr_kws=dict(ek=group_structure),
                    verbose=False,
                    temperature=temperature
                )
            finally:
                # Restore the original stdout
                os.dup2(saved_stdout, 1)
                os.close(saved_stdout)

        for ik, k in enumerate(errorr.keys()):
            err_file_path = base_path.joinpath("errorr", errorr_path.name.replace('.errorr', f'.{k}'))
            errorr[k].to_file(err_file_path)

        return errorr

    @staticmethod
    def write_log_header(fname, author, njoy_ver):
        """
        Write .log file header with info for reproducibility.

        Parameters
        ----------
        fname : str or Path
            Path to log file.
        author : str
            Author name.
        njoy_ver : str
            NJOY version.

        Returns
        -------
        None
        """
        if fname is None:
            fname = "GetCov.log"

        # datetime object containing current date and time
        sep = "".join(['-']*90)
        now = datetime.now()
        mmddyyhh = now.strftime("%B %d, %Y %H:%M:%S")
        with open(fname, "a") as f:
            f.write(f"Log file generated with python class `GetCov`: \n")
            f.write(f"{sep}\n")
            f.write(f"HOSTNAME: {socket.gethostname()} \n")
            try:
                f.write(f"AUTHOR: {author} \n")
                f.write(f"USERNAME: {os.getlogin()} \n")
            except OSError:
                f.write(f"USERNAME: unknown \n")
            f.write(f"PYTHON VERSION: {sys.version} \n")
            f.write(f"SANDY VERSION: {sandy_ver} \n")
            f.write(f"NJOY VERSION: {njoy_ver} \n")
            f.write(f"NJOY PATH: {os.environ['NJOY']} \n")
            f.write(f"DDYYMMHH: {mmddyyhh} \n")
            f.write(f"{sep}")


class GetCovarianceError(Exception):
    pass


if __name__ == "__main__":

    libs = ['jeff_33']
    eg_ecco33 = sandy.energy_grids.ECCO33

    mynuclides = {
                310710 : "Ga-71",
                922350 : "U-235",
                922380 : "U-238",
                942390 : "Pu-239",
                }

    njoy_ver = 'njoy2016.78'
    author = 'Nicolò Abrate'
    os.environ["NJOY"] = '/usr/local/bin/njoy2016' 
    T = 300 # [K]

    mycov = {}
    for lib in libs:
        mycov[lib] = {}
        print(f"Getting covariances from library {lib}:\n")

        for ZAid, ZAis in mynuclides.items():
            try:
                print(f"Extracting covariance matrix for {ZAis}...")
                mycov[lib][ZAis] = GetCovariance(ZAid, temperature=T, group_structure=eg_ecco33, 
                                                egridname="ECCO-33", lib=lib, njoy_ver=njoy_ver,
                                                author=author)
                rcov_2_102 = mycov[lib][ZAis].get(MT=(2, 102))
                np_cov = mycov[lib][ZAis].get(MT=[2, 18])
                print(f"DONE")
            except KeyError:
                print(f"FAILED")
                continue
            except IndexError:
                print(f"FAILED")
                continue


