"""Unit tests for RDKit 2D descriptor."""

import numpy as np
import pytest

from yonod.descriptors.rdkit2d import RDKit2DDescriptor, RDKIT_2D_DESCRIPTORS


class TestRDKit2DDescriptor:
    """Test cases for RDKit2DDescriptor."""

    def test_output_dim(self):
        """Test output dimension matches descriptor list."""
        desc = RDKit2DDescriptor()
        assert desc.output_dim == len(RDKIT_2D_DESCRIPTORS)

    def test_valid_smiles(self):
        """Test featurization of valid SMILES."""
        desc = RDKit2DDescriptor()
        smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        features, mask = desc.featurize(smiles)

        assert features.shape == (3, desc.output_dim)
        assert mask.all()
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        desc = RDKit2DDescriptor()
        smiles = ["CCO", "INVALID_SMILES", ""]
        features, mask = desc.featurize(smiles)

        assert features.shape == (3, desc.output_dim)
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == False

    def test_empty_marker(self):
        """Test handling of (无) marker."""
        desc = RDKit2DDescriptor()
        smiles = ["(无)", "CCO"]
        features, mask = desc.featurize(smiles)

        assert mask[0] == False
        assert mask[1] == True

    def test_multi_component_smiles(self):
        """Test that multi-component SMILES uses first component."""
        desc = RDKit2DDescriptor()
        # 多分子 SMILES，只取第一个
        smiles = ["CCO.CC(=O)O"]
        features, mask = desc.featurize(smiles)

        # 应该只计算 CCO 的描述符
        features_single, _ = desc.featurize(["CCO"])
        assert np.allclose(features[0], features_single[0])

    def test_descriptor_names(self):
        """Test get_descriptor_names method."""
        desc = RDKit2DDescriptor()
        names = desc.get_descriptor_names()
        assert len(names) == len(RDKIT_2D_DESCRIPTORS)
        assert names[0] == 'MaxAbsEStateIndex'
