"""Unit tests for DRFP descriptor."""

import numpy as np
import pandas as pd
import pytest

from yonod.descriptors.drfp_desc import (
    DRFPDescriptor,
    build_reaction_smarts,
    build_reaction_smarts_from_df,
    _ensure_drfp,
)


class TestBuildReactionSmarts:
    """Test cases for build_reaction_smarts function."""

    def test_two_substrates(self):
        """Test with two substrates."""
        rxn = build_reaction_smarts("CCO", "CC(=O)O", "CCOC(C)=O")
        assert rxn == "CCO.CC(=O)O>>CCOC(C)=O"

    def test_one_substrate(self):
        """Test with one substrate (sub2 empty)."""
        rxn = build_reaction_smarts("CCO", "", "CCO")
        assert rxn == "CCO>>CCO"

    def test_no_marker(self):
        """Test with (无) marker."""
        rxn = build_reaction_smarts("CCO", "(无)", "CCO")
        assert rxn == "CCO>>CCO"

    def test_empty_product(self):
        """Test with empty product."""
        rxn = build_reaction_smarts("CCO", "CC", "")
        assert rxn == ""


class TestBuildReactionSmartsFromDF:
    """Test cases for build_reaction_smarts_from_df function."""

    def test_basic_construction(self):
        """Test basic reaction SMARTS construction from DataFrame."""
        df = pd.DataFrame({
            'sub_1': ['CCO', 'c1ccccc1'],
            'sub_2': ['CC(=O)O', 'CC'],
            'prod': ['CCOC(C)=O', 'CCc1ccccc1'],
        })

        smarts = build_reaction_smarts_from_df(
            df,
            reactant_cols=['sub_1', 'sub_2'],
            product_cols=['prod'],
        )

        assert smarts[0] == "CCO.CC(=O)O>>CCOC(C)=O"
        assert smarts[1] == "c1ccccc1.CC>>CCc1ccccc1"

    def test_empty_handling(self):
        """Test handling of (无) marker and NaN."""
        df = pd.DataFrame({
            'sub_1': ['CCO', 'c1ccccc1', 'CC'],
            'sub_2': ['(无)', np.nan, 'CC(=O)O'],
            'prod': ['CCO', 'c1ccccc1', 'CCOC(C)=O'],
        })

        smarts = build_reaction_smarts_from_df(
            df,
            reactant_cols=['sub_1', 'sub_2'],
            product_cols=['prod'],
        )

        assert smarts[0] == "CCO>>CCO"
        assert smarts[1] == "c1ccccc1>>c1ccccc1"
        assert smarts[2] == "CC.CC(=O)O>>CCOC(C)=O"


@pytest.mark.skipif(not _ensure_drfp(), reason="drfp not installed")
class TestDRFPDescriptor:
    """Test cases for DRFPDescriptor (requires drfp installed)."""

    def test_output_dim(self):
        """Test default output dimension."""
        desc = DRFPDescriptor()
        assert desc.output_dim == 2048

    def test_custom_n_bits_error(self):
        """Test that custom n_bits raises error (drfp 0.3.x limitation)."""
        with pytest.raises(ValueError, match="仅支持 2048 维"):
            desc = DRFPDescriptor(n_bits=1024)

    def test_valid_reaction(self):
        """Test featurization of valid reaction SMARTS."""
        desc = DRFPDescriptor(n_bits=2048)
        rxn = ["CCO.CC(=O)O>>CCOC(C)=O"]
        features, mask = desc.featurize(rxn)

        assert features.shape == (1, 2048)
        assert mask[0] == True

    def test_invalid_format(self):
        """Test handling of invalid reaction format (no >>)."""
        desc = DRFPDescriptor()
        rxn = ["CCO.CC(=O)O"]  # 缺少 >>
        features, mask = desc.featurize(rxn)

        assert mask[0] == False

    def test_empty_smarts(self):
        """Test handling of empty SMARTS."""
        desc = DRFPDescriptor()
        rxn = ["", "CCO>>CCO"]
        features, mask = desc.featurize(rxn)

        assert mask[0] == False
        assert mask[1] == True
