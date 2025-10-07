import pytest
import numpy as np
from pathlib import Path
from collections import OrderedDict

from pyNDUS.GetSensitivity import Sensitivity, SensitivityError

@pytest.fixture
def dummy_serpent_file(tmp_path):
    # Create a dummy serpent file
    file = tmp_path / "dummy_sens0.m"
    file.write_text("dummy content")
    return file

@pytest.fixture
def dummy_eranos_file(tmp_path):
    # Create a dummy eranos file
    file = tmp_path / "dummy.eranos"
    file.write_text("dummy content")
    return file

def test_filepath_setter(tmp_path):
    file = tmp_path / "test.m"
    file.write_text("test")
    s = Sensitivity(file)
    assert s.filepath == file

    with pytest.raises(SensitivityError):
        Sensitivity(tmp_path / "not_exist.m")

def test_reader_setter(dummy_serpent_file):
    s = Sensitivity(dummy_serpent_file)
    assert s.reader == "serpent"
    with pytest.raises(ValueError):
        s.reader = 123

def test_materials_property():
    s = Sensitivity.__new__(Sensitivity)
    s.reader = "serpent"
    s.materials = ["fuel", "coolant"]
    assert list(s.materials.keys()) == ["fuel", "coolant"]

def test_zaid_property():
    s = Sensitivity.__new__(Sensitivity)
    s.zaid = [922350, 922380]
    assert list(s.zaid.keys()) == [922350, 922380]

def test_zais_property(monkeypatch):
    s = Sensitivity.__new__(Sensitivity)
    monkeypatch.setattr("pyNDUS.GetSensitivity.utils.zaid2zais", lambda x: f"ZAI-{x}")
    s.zais = [922350, 922380]
    assert list(s.zais.keys()) == ["ZAI-922350", "ZAI-922380"]

def test_group_structure_property():
    s = Sensitivity.__new__(Sensitivity)
    s.group_structure = np.arange(5)
    assert (s.group_structure == np.arange(5)).all()
    assert s.n_groups == 4

def test_MTs_setter_serpent():
    s = Sensitivity.__new__(Sensitivity)
    s.reader = "serpent"
    s.MTs = ["xs 2", "xs 18", "nubar total"]
    assert 2 in s.MTs and 18 in s.MTs and 452 in s.MTs

def test_MTs_setter_eranos():
    s = Sensitivity.__new__(Sensitivity)
    s.reader = "eranos"
    s.MTs = [2, 18, 102]
    assert list(s.MTs.values()) == [2, 18, 102]

def test_sens_setter_and_get(monkeypatch):
    s = Sensitivity.__new__(Sensitivity)
    s.reader = "serpent"
    s.responses = ["keff"]
    s.materials = OrderedDict([("fuel", 0)])
    s.zais = OrderedDict([("U-235", 0)])
    s.MTs = OrderedDict([(2, 0)])
    s.group_structure = np.arange(3)
    arr = np.zeros((1, 1, 1, 1, 2))
    s.sens = {"keff": arr}
    result = s.get(resp="keff", mat="fuel", MT=2, za="U-235", g=1)
    assert isinstance(result, np.ndarray) or isinstance(result, tuple)

def test_get_invalid_inputs():
    s = Sensitivity.__new__(Sensitivity)
    s.reader = "serpent"
    s.responses = ["keff"]
    s.materials = OrderedDict([("fuel", 0)])
    s.zais = OrderedDict([("U-235", 0)])
    s.MTs = OrderedDict([(2, 0)])
    s.group_structure = np.arange(3)
    arr = np.zeros((1, 1, 1, 1, 2))
    s.sens = {"keff": arr}
    with pytest.raises(ValueError):
        s.get(resp="not_a_response")
    with pytest.raises(ValueError):
        s.get(mat="not_a_material")
    with pytest.raises(ValueError):
        s.get(MT=999)
    with pytest.raises(ValueError):
        s.get(za="not_a_za")