"""Development-only exploratory enrichment of native RDKit alert catalogs.

Catalog predicates are evaluated by RDKit, including their counts, logical
combinations and exclusions. Catalog entry names are never parsed as SMARTS.
The serialized one-entry catalogs preserve the complete native definitions;
matched leaf SMARTS are explanatory metadata, not replacement predicates.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import FilterCatalog
from scipy.stats import fisher_exact

from .data import ROOT, rows_fingerprint
from .input_provenance import source_file_fields, verify_input_identity

CATALOG_NAMES = ("BRENK", "NIH", "ZINC", "PAINS_A", "PAINS_B", "PAINS_C")
SCHEMA_VERSION = 1
DOCUMENTATION = "https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html"
SOURCE_README = "https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/FilterCatalog/README"
SMARTS_MATCHER_DOCUMENTATION = "https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html#rdkit.Chem.rdfiltercatalog.SmartsMatcher"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _csv(path, records, fields=None):
    fields = list(fields or records[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjustment over the complete supplied family."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all() or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("BH requires a finite one-dimensional P-value family")
    if not len(pvalues):
        return pvalues.copy()
    order = np.argsort(pvalues, kind="stable")
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def enrichment_statistics(positive_present, negative_present, n_positive, n_negative):
    """Two-sided Fisher P and Haldane-Anscombe corrected OR/Wald interval."""
    values = (positive_present, negative_present, n_positive, n_negative)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Contingency counts must be nonnegative integers")
    a, b = positive_present, negative_present
    c, d = n_positive - a, n_negative - b
    if min(c, d) < 0 or min(n_positive, n_negative) < 1:
        raise ValueError("Invalid class totals or present counts")
    statistic, pvalue = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    corrected = np.asarray([a, b, c, d], dtype=float) + .5
    odds = float(corrected[0] * corrected[3] / (corrected[1] * corrected[2]))
    standard_error = float(np.sqrt((1 / corrected).sum()))
    return {"positive_present": a, "reference_negative_present": b,
            "positive_absent": c, "reference_negative_absent": d,
            "support": a + b, "n_positive": n_positive, "n_reference_negative": n_negative,
            "raw_odds_ratio": float(statistic) if math.isfinite(statistic) else None,
            "raw_odds_ratio_status": "finite" if math.isfinite(statistic) else "undefined_or_infinite",
            "odds_ratio_corrected": odds,
            "or_ci95_lower": math.exp(math.log(odds) - 1.96 * standard_error),
            "or_ci95_upper": math.exp(math.log(odds) + 1.96 * standard_error),
            "fisher_p": float(pvalue)}


def serialize_entry(entry):
    """Export a reloadable complete native predicate, not an inferred SMARTS."""
    catalog = FilterCatalog.FilterCatalog()
    catalog.AddEntry(entry)
    blob = bytes(catalog.Serialize())
    return {"native_one_entry_catalog_base64": base64.b64encode(blob).decode("ascii"),
            "native_definition_sha256": _hash_bytes(blob)}


def restore_entry(definition):
    blob = base64.b64decode(definition["native_one_entry_catalog_base64"], validate=True)
    if _hash_bytes(blob) != definition["native_definition_sha256"]:
        raise ValueError("Serialized native catalog predicate changed")
    catalog = FilterCatalog.FilterCatalog(blob)
    if catalog.GetNumEntries() != 1:
        raise ValueError("A native entry definition must contain exactly one predicate")
    return catalog.GetEntryWithIdx(0)


def native_match_detail(entry, molecule):
    """Keep RDKit's predicate decision and its actual leaf/atom match data."""
    if not entry.HasFilterMatch(molecule):
        return None
    parts, atoms = [], set()
    for match in entry.GetFilterMatches(molecule):
        matcher = match.filterMatch
        pairs = [[int(pair.query), int(pair.target)] for pair in match.atomPairs]
        atoms.update(target for _, target in pairs)
        part = {"matcher_name": matcher.GetName(), "matcher_type": type(matcher).__name__, "atom_pairs": pairs}
        if isinstance(matcher, FilterCatalog.SmartsMatcher):
            part.update(smarts=Chem.MolToSmarts(matcher.GetPattern()),
                        min_count=int(matcher.GetMinCount()), max_count=int(matcher.GetMaxCount()))
        else:
            part.update(smarts=None, min_count=None, max_count=None)
        parts.append(part)
    return {"atom_indices": sorted(atoms), "match_parts": parts,
            "highlight_scope": "Native matching atoms only; logical absence/exclusion predicates may have no atom highlight."}


def load_catalogs():
    if not FilterCatalog.FilterCatalogCanSerialize():
        raise RuntimeError("RDKit native catalog serialization is required to preserve predicate definitions")
    catalogs, definitions, lookup, catalog_records = [], [], {}, []
    for name in CATALOG_NAMES:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name))
        catalog = FilterCatalog.FilterCatalog(params)
        blob = bytes(catalog.Serialize())
        catalog_records.append({"catalog": name, "entry_count": catalog.GetNumEntries(),
                                "native_catalog_sha256": _hash_bytes(blob),
                                "native_catalog_base64": base64.b64encode(blob).decode("ascii")})
        catalogs.append((name, catalog))
        for index in range(catalog.GetNumEntries()):
            entry = catalog.GetEntryWithIdx(index)
            if not entry.IsValid():
                raise ValueError(f"Invalid native catalog entry: {name}/{index}")
            entry_sha = _hash_bytes(bytes(entry.Serialize()))
            key = (name, entry_sha)
            alert_id = f"{name}:{index:04d}"
            lookup.setdefault(key, []).append(alert_id)
            properties = {key: entry.GetProp(key) for key in entry.GetPropList()}
            definitions.append({"alert_id": alert_id, "catalog": name, "catalog_index_zero_based": index,
                                "alert_name": entry.GetDescription(), "native_properties": properties,
                                "native_entry_sha256": entry_sha,
                                "reference": properties.get("Reference", ""), "scope": properties.get("Scope", ""),
                                "definition_semantics": "Complete RDKit catalog predicate; do not substitute the entry name or observed leaf SMARTS.",
                                **serialize_entry(entry), "observed_matching_leaf_definitions": []})
    for definition in definitions:
        definition["identical_native_entry_ids_within_catalog"] = lookup[(definition["catalog"], definition["native_entry_sha256"])]
    return catalogs, definitions, lookup, catalog_records


def analyze_development(rows, assignments, *, split_mode="structure_group"):
    """Use development labels only; preserve source SMILES and atom ordering."""
    if len(rows) != len(assignments) or [r["compound_id"] for r in rows] != [r["compound_id"] for r in assignments]:
        raise ValueError("Assignments must match every current DATA row in order")
    if split_mode not in {"structure_group", "legacy_random"} or len({r["compound_id"] for r in rows}) != len(rows):
        raise ValueError("Unknown split mode or duplicated compound identity")
    if any(item.get("dataset_partition") not in {"development", "test"} for item in assignments):
        raise ValueError("Unrecognized partition")
    development = [row for row, item in zip(rows, assignments) if item["dataset_partition"] == "development"]
    if not development or len({r["compound_id"] for r in development}) != len(development):
        raise ValueError("Development records must be nonempty and unique")
    groups = {part: {item["structure_group_id"] for item in assignments if item["dataset_partition"] == part}
              for part in ("development", "test")}
    shared_groups = len(groups["development"] & groups["test"])
    if split_mode == "structure_group" and shared_groups:
        raise ValueError("Development/test structure-group leakage")
    if any(type(row["label"]) is not int or row["label"] not in (0, 1) for row in development):
        raise ValueError("Invalid development binary label")
    n_positive = sum(row["label"] for row in development)
    n_negative = len(development) - n_positive
    catalogs, definitions, lookup, catalog_records = load_catalogs()
    hits, grouped, observed = [], {r["alert_id"]: [] for r in definitions}, {r["alert_id"]: {} for r in definitions}
    for row in development:
        molecule = Chem.MolFromSmiles(row["standardized_smiles"])
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(f"Invalid development source structure: {row['compound_id']}")
        matched_alerts = set()
        for catalog_name, catalog in catalogs:
            for entry in catalog.GetMatches(molecule):
                alert_ids = lookup[(catalog_name, _hash_bytes(bytes(entry.Serialize())))]
                detail = native_match_detail(entry, molecule)
                if detail is None:
                    raise ValueError("Native catalog and native entry predicate disagree")
                for alert_id in alert_ids:
                    if alert_id in matched_alerts:
                        continue
                    matched_alerts.add(alert_id)
                    record = {"alert_id": alert_id, "catalog": catalog_name, "alert_name": entry.GetDescription(),
                          "compound_id": row["compound_id"], "name": row["name"], "PubChem_CID": row["PubChem_CID"],
                          "standardized_smiles": row["standardized_smiles"], "label": row["label"],
                          "heavy_atom_count": molecule.GetNumHeavyAtoms(),
                          "atom_indices_json": json.dumps(detail["atom_indices"]),
                          "match_parts_json": json.dumps(detail["match_parts"], ensure_ascii=False, sort_keys=True),
                          "highlight_scope": detail["highlight_scope"]}
                    hits.append(record)
                    grouped[alert_id].append(record)
                    for part in detail["match_parts"]:
                        definition = {key: value for key, value in part.items() if key != "atom_pairs"}
                        observed[alert_id][json.dumps(definition, sort_keys=True)] = definition
    enrichment = []
    for definition in definitions:
        alert_hits = grouped[definition["alert_id"]]
        positive = sum(hit["label"] for hit in alert_hits)
        statistics = enrichment_statistics(positive, len(alert_hits) - positive, n_positive, n_negative)
        preferred = 1 if statistics["odds_ratio_corrected"] > 1 else 0
        candidates = [hit for hit in alert_hits if hit["label"] == preferred] or alert_hits
        example = min(candidates, key=lambda hit: (hit["heavy_atom_count"], hit["compound_id"])) if candidates else None
        record = {key: definition[key] for key in ("alert_id", "catalog", "alert_name", "reference", "scope", "native_definition_sha256")}
        record.update(statistics, display_support_eligible=statistics["support"] >= 5,
                      example_compound_id=example["compound_id"] if example else None,
                      example_name=example["name"] if example else None,
                      example_PubChem_CID=example["PubChem_CID"] if example else None,
                      example_standardized_smiles=example["standardized_smiles"] if example else None,
                      example_label=example["label"] if example else None,
                      example_atom_indices_json=example["atom_indices_json"] if example else None,
                      example_match_parts_json=example["match_parts_json"] if example else None)
        enrichment.append(record)
        definition["observed_matching_leaf_definitions"] = list(observed[definition["alert_id"]].values())
    qvalues = bh_adjust([item["fisher_p"] for item in enrichment])
    for item, qvalue in zip(enrichment, qvalues):
        item.update(fdr_bh_q=float(qvalue), significant_bh_005=bool(qvalue < .05),
                    bh_family_size=len(enrichment), direction="positive" if item["odds_ratio_corrected"] > 1 else "reference-negative")
    return {"development": development, "definitions": definitions, "catalogs": catalog_records,
            "hits": hits, "enrichment": enrichment, "n_positive": n_positive, "n_reference_negative": n_negative,
            "split_mode": split_mode, "holdout_shared_groups": shared_groups}


def export_structural_alerts(rows, assignments, source, split_path, output, *, protocol_path=None):
    output = Path(output)
    if source["rows_fingerprint"] != rows_fingerprint(rows):
        raise ValueError("DATA rows differ from their supplied fingerprint")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    split_mode = split.get("manifest", {}).get("split_mode", "structure_group")
    if split["assignments"] != assignments:
        raise ValueError("Assignments differ from their source file")
    protocol_sha = None
    if protocol_path is not None:
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        verify_input_identity(protocol, source, ROOT)
        if (protocol.get("source_rows_fingerprint") != source["rows_fingerprint"]
                or protocol.get("split_payload_sha256") != sha256(split_path)):
            raise ValueError("Independent exploration does not match the frozen DATA/split")
        if protocol.get("split_mode", "structure_group") != split_mode:
            raise ValueError("Catalog exploration split mode differs from the frozen protocol")
        protocol_sha = sha256(protocol_path)
    result = analyze_development(rows, assignments, split_mode=split_mode)
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "catalog_definitions.json", {"schema_version": SCHEMA_VERSION, "rdkit_version": rdBase.rdkitVersion,
          "catalogs": result["catalogs"], "entries": result["definitions"],
          "native_definition_authority": "Serialized one-entry catalogs retain complete SMARTS/count/logical predicates and exceptions.",
          "observed_leaves_limit": "Observed matching leaf SMARTS do not replace a full predicate. Zero-hit entries may have no observed leaves."})
    _csv(output / "alert_enrichment.csv", result["enrichment"])
    _csv(output / "development_alert_hits.csv", result["hits"], fields=(list(result["hits"][0]) if result["hits"] else
         ["alert_id", "catalog", "alert_name", "compound_id", "name", "PubChem_CID", "standardized_smiles", "label", "heavy_atom_count", "atom_indices_json", "match_parts_json", "highlight_scope"]))
    _csv(output / "development_identity.csv", result["development"])
    candidates = [item for item in result["enrichment"] if item["display_support_eligible"] and item["significant_bh_005"]]
    representatives = sorted([item for item in candidates if item["direction"] == "positive"],
                             key=lambda item: (item["fdr_bh_q"], item["fisher_p"], item["alert_id"]))[:5]
    _json(output / "figure_s5_selection.json", {"forest": sorted(candidates, key=lambda item: (item["odds_ratio_corrected"], item["alert_id"])),
          "positive_representatives": representatives,
          "selection_rule": "All BH-significant support>=5 entries in the forest; up to five positive-direction entries by q/P/ID for structures. Per-entry example: direction-matching class, then smallest heavy-atom count, then compound ID.",
          "no_legacy_cases_or_statistics_reused": True})
    manifest = {"schema_version": SCHEMA_VERSION, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_name": "independent_development_catalog_alert_exploration", "rdkit_version": rdBase.rdkitVersion,
        "split_mode": split_mode, "holdout_shared_groups": result["holdout_shared_groups"],
        **source_file_fields(protocol if protocol_path is not None else source), "source_rows_fingerprint": source["rows_fingerprint"],
        "split_payload_sha256": sha256(split_path), "frozen_model_protocol_sha256_reference_only": protocol_sha,
        "analysis_source_sha256": sha256(__file__), "catalogs": list(CATALOG_NAMES),
        "catalog_entry_counts": {item["catalog"]: item["entry_count"] for item in result["catalogs"]},
        "n_development": len(result["development"]), "positive": result["n_positive"],
        "reference_negative": result["n_reference_negative"], "test_labels_used": False,
        "model_predictions_used": False, "frozen_fitting_protocol_modified": False,
        "bh_family_size": len(result["enrichment"]), "bh_family": "Every native entry in the six prespecified catalogs, including zero-support entries; no support prefilter before BH.",
        "native_duplicate_policy": "Distinct catalog indices remain separate hypotheses, including native-identical ZINC entries; each compound is counted once per entry. Entry hashes and equivalence IDs expose duplicates.",
        "display_support_minimum": 5, "n_display_significant": len(candidates), "n_representative_structures": len(representatives),
        "fisher_alternative": "two-sided", "effect": "Haldane-Anscombe +0.5 to all four cells; log-OR Wald 95% interval",
        "historical_difference": "Historical enrichment CSV tested 163 observed alerts, including support=1; support>=5 was a display restriction. This independent analysis corrects the complete catalog family, including zero-support entries, before applying that display restriction.",
        "limits": "General medicinal-chemistry catalog associations are exploratory, not kidney-specific toxicophores or causal mechanisms. Fisher/Wald calculations do not account for within-scaffold dependence.",
        "documentation": DOCUMENTATION, "smarts_matcher_documentation": SMARTS_MATCHER_DOCUMENTATION,
        "catalog_reference_source": SOURCE_README,
        "files": {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path.name != "structural_alert_manifest.json"}}
    _json(output / "structural_alert_manifest.json", manifest)
    return manifest
