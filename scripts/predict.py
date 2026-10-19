"""Predict with one saved final model and its recorded preprocessing.

CSV input needs compound_id and standardized_smiles. Chemistry standardization
must precede inference; this script rejects mixtures and never completes stereo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from nephrotox.features import fingerprint_matrix,compute_descriptors,load_descriptor_names


def inference_features(smiles,representation):
    mols=[Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None or len(Chem.GetMolFrags(m))!=1 or any(a.GetAtomicNum()==0 for a in m.GetAtoms()) for m in mols):
        raise ValueError("Inference requires complete single-component structures with defined atoms")
    if representation=="graph":
        return np.asarray(smiles)
    if representation=="rdkit166":
        names=load_descriptor_names(ROOT/"configs/rdkit_descriptors_166.txt")
        X,failures=compute_descriptors(mols,[str(i) for i in range(len(mols))],names)
        if failures:
            raise ValueError(f"Descriptor calculation failed: {failures}")
        return X
    if "count" not in representation:
        return fingerprint_matrix(mols,representation)
    generators={
        "morgan_count2048":lambda:rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True),
        "atompair_count1024":lambda:rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024,includeChirality=True),
        "torsion_count1024":lambda:rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024,includeChirality=True),
    }
    generator=generators[representation]()
    X=np.zeros((len(mols),2048 if "2048" in representation else 1024),dtype=np.int32)
    for i,mol in enumerate(mols):
        DataStructs.ConvertToNumpyArray(generator.GetCountFingerprint(mol),X[i])
    return X


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smiles")
    inputs.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    metadata=json.loads((args.model_dir/"metadata.json").read_text(encoding="utf-8"))
    path=args.model_dir/metadata["model_file"]
    if not metadata.get("complete") or hashlib.sha256(path.read_bytes()).hexdigest()!=metadata["model_sha256"]:
        raise ValueError("Model is incomplete or does not match its metadata")
    if args.input:
        with args.input.open(encoding="utf-8-sig",newline="") as stream:
            records=list(csv.DictReader(stream))
    else:
        records=[{"compound_id":"QUERY001","standardized_smiles":args.smiles}]
    if not records or any(not r.get("compound_id") or not r.get("standardized_smiles") for r in records):
        raise ValueError("Input requires compound_id and standardized_smiles for every query")
    if len({r["compound_id"] for r in records})!=len(records):
        raise ValueError("Duplicate query compound_id")
    X=inference_features([r["standardized_smiles"] for r in records],metadata["representation"])
    estimator=joblib.load(path)
    probabilities=estimator.predict_proba(X)[:,1]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["compound_id","standardized_smiles","model_id","probability","predicted_label","threshold"])
        writer.writeheader()
        for row,p in zip(records,probabilities):
            writer.writerow({"compound_id":row["compound_id"],"standardized_smiles":row["standardized_smiles"],
                "model_id":metadata["model_id"],"probability":float(p),"predicted_label":int(p>=.5),"threshold":.5})
    print(args.output)


if __name__=="__main__":
    main()
