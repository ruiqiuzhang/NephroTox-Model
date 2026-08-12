from pathlib import Path

from rdkit import Chem

from nephrotox.config import load_config, project_path
from nephrotox.features import atompair1024, maccs166, morgan2048, rdkit166, torsion1024


def test_fingerprint_dimensions() -> None:
    molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    assert morgan2048(molecules).shape == (2, 2048)
    assert maccs166(molecules).shape == (2, 166)
    assert atompair1024(molecules).shape == (2, 1024)
    assert torsion1024(molecules).shape == (2, 1024)


def test_frozen_descriptor_definition() -> None:
    config = load_config()
    descriptor_path = project_path(config, "configs/rdkit_descriptors_166.txt")
    names = [line for line in descriptor_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(names) == len(set(names)) == 166
    matrix = rdkit166([Chem.MolFromSmiles("CCO")], descriptor_path)
    assert matrix.shape == (1, 166)

