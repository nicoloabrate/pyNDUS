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
        ZAID (Z*1000 + A) identifier of the nuclide.
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
    MFs2MTs : dict
        Mapping from MF numbers to lists of MT numbers (excluding the header, i.e. MT=451).
    mat : int
        MAT number for the nuclide.
    rcov : dict
        Dictionary with extracted covariance matrices in ´pandas.DataFrame´ objects.
        Available MF:
        -MF=31 for neutron multiplicities
        -MF=33 for cross sections

    Methods
    -------
    plot(...)
        Static method to plot covariance or correlation matrices.
    sandy_calls_errorr(...)
        Static method to run the ERRORR module of NJOY via SANDY and save output files.
    write_log_header(fname)
        Static method to write a log file header for reproducibility.
    """
    def __init__(self, zaid, temperature=300, group_structure=None, egridname=None,
                 lib="endfb_80", cwd=None):

        # --- input validation
        self.zaid = zaid
        self.zais = utils.zaid2zais(self.zaid)
        self.temperature = temperature
        self.group_structure = group_structure
        self.egridname = egridname
        self.library = lib
        self.path = cwd

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
                                                        errorr_path, lib)

        self.MFs2MTs = errorr_out
        self.mat = errorr_out
        self.rcov =  errorr_out

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):

        if not isinstance(value, int) and not isinstance(value, float):
            raise ValueError(f"Expected type int or float instead of type {type(value)} for arg 'temperature'")
        elif value <= 0:
            raise ValueError("Temperature must be >0!")
        else:
            self._temperature = value

    @property
    def zaid(self):
        return self._zaid

    @zaid.setter
    def zaid(self, value):
        if value <= 0:
            raise ValueError("ZAid must be >0!")
        elif not isinstance(value, int):
            raise ValueError(f"Expected type int instead of type {type(value)} for arg 'ZAid'")

        self._zaid = value

    @property
    def group_structure(self):
        return self._group_structure

    @group_structure.setter
    def group_structure(self, value):
        if value is not None:
            if not isinstance(value, Iterable):
                raise ValueError(f"Expected an iterable instead of type {type(value)} for arg 'group_structure'")

        self._group_structure = value

    @property
    def egridname(self):
        return self._egridname

    @egridname.setter
    def egridname(self, value):
        if value is None:
            self._egridname = "sandy_default_energy_grid"
        elif not isinstance(value, str):
            raise ValueError(f"Expected 'str' instead of type {type(value)} for arg 'egridname'")
        else:
            self._egridname = value

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, value):
        if not isinstance(value, str):
            raise ValueError(f"Expected 'str' instead of type {type(value)} for arg 'lib'")
        else:
            self._library = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, cwd):
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
        return self._zais

    @zais.setter
    def zais(self, value):
        if not isinstance(value, str):
            raise ValueError("ZAis must be a string!")
        self._zais = value

    @property
    def MFs2MTs(self):
        return self._MFs2MTs

    @MFs2MTs.setter
    def MFs2MTs(self, rcov):

        map_MF2MT = {}

        for mf in rcov.keys():
            map_MF2MT[mf] = rcov[mf].mt
            # remove the heading MT (no covariance data)
            map_MF2MT[mf].remove(451)

        self._MFs2MTs = map_MF2MT

    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, rcov):

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
        return self._rcov

    @rcov.setter
    def rcov(self, rcov):
        out = {}
        for mf in rcov.keys():
            if mf in ['errorr31', 'errorr33']:
                out[mf] = rcov[mf].get_cov().data

        self._rcov = out

    @staticmethod
    def sandy_calls_errorr(endf6_tape, zaid, temperature, group_structure, egridname, errorr_path, lib):

        njoy_inp = endf6_tape.get_errorr(groupr_kws=dict(ek=group_structure), errorr_kws=dict(ek=group_structure), 
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

        GetCovariance.write_log_header(out_file_path)
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
    def write_log_header(fname):
        """Write .log file header with info for reproducibility.

        Parameters
        ----------
        ``None``

        Returns
        -------
        ``None``
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
            f.write(f"NJOY PATH: {os.environ["NJOY"]} \n")
            f.write(f"DDYYMMHH: {mmddyyhh} \n")
            f.write(f"{sep}")

    def plot(self, E=None, corr=True, logscale=True, figname=None,
             myax=None, thresh=None, title=None, MT=None, fontsize=None,
             is_pd=True, figsize=None, fmt='pdf'):

        # validate input
        if MT is not None:
            if isinstance(MT, int):
                MT = [MT]
            elif not isinstance(MT, Iterable):
                raise ValueError(f"Input arg. MT must be int or iterable, not type {type(MT)}")

            for mt in MT:
                not_found = True
                for mf, mts in self.MFs2MTs.items():
                    if mt in mts:
                        not_found = False
            
            if not_found:
                raise ValueError(f"Selected MT not available in covariance matrix object, so it cannot be plotted!")
        else:
            MT = []
            for mf, mts in self.MFs2MTs.items():
                MT += mts

        
        # extract 2D arrays

        # call auxiliary plotter


class GetCovarianceError(Exception):
    pass


if __name__ == "__main__":

    libs = ['jeff_33', 'endfb_80', 'jendl_40u']
    eg_ecco33 = sandy.energy_grids.ECCO33

    mynuclides = {
                310690 : "Ga-69",
                310710 : "Ga-71",
                922330 : "U-233",
                922340 : "U-234",
                922350 : "U-235",
                922380 : "U-238",
                942390 : "Pu-239",
                942400 : "Pu-240",
                942410 : "Pu-241",
                942420 : "Pu-242",
                110230 : "Na-23",
                822040 : "Pb-204",
                822060 : "Pb-206",
                822070 : "Pb-207",
                822080 : "Pb-208",
                }

    njoy_ver = 'njoy2016.78'
    author = 'Nicolò Abrate'
    os.environ['NJOY'] = '/usr/local/bin/njoy2016' 
    T = 300 # [K]

    mycov = {}
    for lib in libs:
        mycov[lib] = {}
        print(f"Getting covariances from library {lib}:\n")

        for ZAid, ZAis in mynuclides.items():
            try:
                print(f"Extracting covariance matrix for {ZAis}...")
                mycov[lib][ZAis] = GetCovariance(ZAid, T, eg_ecco33, egridname="ECCO-33", lib=lib)
                print(f"DONE")
            except KeyError:
                print(f"FAILED")
                continue
            except IndexError:
                print(f"FAILED")
                continue


