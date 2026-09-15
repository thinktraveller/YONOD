"""Non-modelling consistency checks for the step-31 public documentation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class ChemicalVaeDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.readme = (cls.root / "README.md").read_text(encoding="utf-8")
        cls.example = (cls.root / "example.md").read_text(encoding="utf-8")
        cls.reference = (cls.root / "project-docs" / "chemical-vae-reference.md").read_text(encoding="utf-8")

    def test_public_docs_describe_the_frozen_runtime_and_limitations(self) -> None:
        for text in (self.readme, self.example):
            for expected in (
                "chemical_vae",
                "model_manifest",
                "backend: pytorch",
                "device: cpu",
                "input_preprocessing: identity",
                "output: z_mean_sample",
                "yonod.py",
            ):
                self.assertIn(expected, text)
        for expected in (
            "不 trim",
            "不 canonicalize",
            "不拆盐",
            "不截断",
            "TensorFlow/Keras",
            "Windows CPU 仍没有实测",
            "cuda:0",
            "zinc_properties",
        ):
            self.assertIn(expected, self.readme)
        self.assertIn("YONOD 当前接入状态", self.reference)
        self.assertIn("not_verified", self.reference)
        self.assertIn("性质头", self.reference)

    def test_documented_chemical_vae_examples_and_dependencies_exist(self) -> None:
        for relative in (
            "configs/chemical_vae/step31_6_features.yaml",
            "configs/chemical_vae/step31_6_train.yaml",
            "configs/chemical_vae/step31_6_all.yaml",
            "configs/chemical_vae/step31_7_comparison_all.yaml",
            "configs/chemical_vae/step31_9_zinc_v5_gpu_all.yaml",
            "configs/chemical_vae/step31_9_zinc_properties_gpu_all.yaml",
            "configs/chemical_vae/step31_9_zinc_v5_windows_cpu_all.yaml",
            "WEIGHTS/chemical_vae/zinc-37e96cd3bc8f9680/v5/conversion_manifest.json",
            "WEIGHTS/chemical_vae/zinc_properties-9f923d03c5ce558d/v5/conversion_manifest.json",
            "requirements-chemical-vae-parity.txt",
            "project-docs/chemical-vae-platform-handoff.md",
        ):
            self.assertTrue((self.root / relative).is_file(), relative)
        base_requirements = (self.root / "requirements.txt").read_text(encoding="utf-8")
        parity_requirements = (self.root / "requirements-chemical-vae-parity.txt").read_text(encoding="utf-8")
        self.assertIn("h5py==3.14.0", base_requirements)
        self.assertNotIn("tensorflow-cpu==", base_requirements)
        self.assertIn("tensorflow-cpu==2.15.1", parity_requirements)

    def test_acceptance_index_records_user_authorized_completion_and_deferred_windows_validation(self) -> None:
        index_path = self.root / "derived" / "chemical_vae" / "step31_acceptance" / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(index["schema_version"], "chemical_vae_step31_acceptance_index/v4")
        self.assertEqual(index["overall_status"], "complete")
        self.assertEqual(
            index["completion_authorization"]["status"],
            "user_authorized_deferred_validation",
        )
        self.assertEqual(
            index["completion_authorization"]["deferred_acceptance"],
            ["C11.windows_cpu"],
        )
        self.assertEqual(set(index["acceptance"]), {f"C{number:02d}" for number in range(1, 13)})
        self.assertEqual(index["acceptance"]["C11"]["status"], "deferred_by_user")
        self.assertEqual(index["acceptance"]["C11"]["windows_cpu"], "not_verified")
        self.assertEqual(index["acceptance"]["C11"]["gpu"], "passed")
        self.assertEqual(index["acceptance"]["C11"]["zinc_properties"], "passed_encoder_scope")
        for entry in index["acceptance"].values():
            for relative in entry["evidence"]:
                self.assertTrue((self.root / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
