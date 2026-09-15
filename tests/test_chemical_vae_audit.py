"""Regression tests for the step-31.1/31.2 Chemical VAE audit tool."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import audit_chemical_vae


class ChemicalVaeAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chem = audit_chemical_vae._rdkit_parser()
        self.charset = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "reference-proejct"
                / "chemical_vae"
                / "models"
                / "zinc"
                / "zinc.json"
            ).read_text(encoding="utf-8")
        )

    def test_identity_preflight_keeps_boundary_failures_distinct(self) -> None:
        cases = {
            "": (True, ["missing"]),
            "CCO": (False, []),
            "C1(": (False, ["rdkit_invalid"]),
            "[Ni]": (False, ["unsupported_character"]),
            "[Pd]": (False, ["unsupported_character"]),
            "CCO.CC": (False, ["unsupported_character"]),
            "CCO,CC": (False, ["rdkit_invalid", "unsupported_character"]),
            "CCO;CC": (False, ["rdkit_invalid", "unsupported_character"]),
            "C" * 121: (False, ["too_long"]),
        }
        for value, (missing, expected_reasons) in cases.items():
            with self.subTest(value=value[:12]):
                result = audit_chemical_vae.assess_smiles(
                    value, charset=self.charset, max_len=120, chem=self.chem
                )
                self.assertEqual(result["processed_input"], value)
                self.assertEqual(result["is_missing"], missing)
                self.assertEqual(result["failure_reasons"], expected_reasons)
                self.assertEqual(result["preflight_encodable"], not expected_reasons)

    def test_hdf5_contract_matches_zinc_metadata(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        audited = audit_chemical_vae.asset_audit(
            repository_root / "reference-proejct" / "chemical_vae", repository_root
        )
        contract = audited["variants"]["zinc"]["encoder_contract"]
        self.assertEqual(contract["status"], "verified_from_hdf5_metadata")
        self.assertEqual(contract["input_shape"], [None, 120, 35])
        self.assertEqual(contract["z_mean_layer"], "z_mean_sample")
        self.assertEqual(contract["z_mean_units"], 196)

    def test_boundary_fixture_writes_full_row_and_reaction_diagnostics(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        dataset = audit_chemical_vae.CoverageDataset(
            name="boundary",
            path=repository_root / "tests" / "fixtures" / "chemical_vae_boundary_smiles.csv",
            sample_id_column="case_id",
            columns=(("contract_test", "smiles"),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = audit_chemical_vae.coverage_audit(
                (dataset,),
                charset=self.charset,
                max_len=120,
                repository_root=repository_root,
                output_dir=output,
            )
            self.assertFalse(manifest["inference_performed"])
            self.assertTrue((output / "sample_role_diagnostics.csv").is_file())
            self.assertTrue((output / "reaction_coverage.csv").is_file())
            summary = manifest["reaction_coverage"][0]
            self.assertEqual(summary["reaction_count"], 9)
            self.assertEqual(summary["concat_kept_count"], 1)
            self.assertEqual(summary["strict_all_roles_encodable_count"], 1)


if __name__ == "__main__":
    unittest.main()
