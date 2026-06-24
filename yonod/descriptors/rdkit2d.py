"""RDKit 2D physicochemical descriptors (~200 descriptors).

Computes a comprehensive set of 2D molecular properties without
requiring 3D conformer generation.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from .base import BaseDescriptor, split_multi_smiles


# 硬编码的 2D 描述符列表（避免 RDKit 版本兼容问题）
# 排除需要 3D 构象的描述符（PMI, NPR, Asphericity 等）
RDKIT_2D_DESCRIPTORS = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex',
    'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge',
    'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
    'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
    'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0',
    'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
    'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2',
    'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
    'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
    'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
    'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
    'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
    'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
    'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
    'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
    'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
    'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
    'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
    'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount',
    'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
    'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
    'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO',
    'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N',
    'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO',
    'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
    'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
    'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
    'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide',
    'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic',
    'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
    'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
    'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
    'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
    'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
    'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
    'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
    'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
    'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea',
]


class RDKit2DDescriptor(BaseDescriptor):
    """RDKit 2D physicochemical descriptors.

    Computes ~200 2D molecular properties for each molecule.
    Failed SMILES parsing results in zero vectors with mask=False.
    """

    name = "rdkit2d"
    output_dim = len(RDKIT_2D_DESCRIPTORS)

    def __init__(self, descriptor_names: Optional[List[str]] = None) -> None:
        """Initialize the RDKit 2D descriptor calculator.

        Args:
            descriptor_names: Optional custom list of descriptor names.
                If None, uses RDKIT_2D_DESCRIPTORS (~200 descriptors).
        """
        if descriptor_names is not None:
            self.descriptor_names = descriptor_names
        else:
            self.descriptor_names = RDKIT_2D_DESCRIPTORS

        self.output_dim = len(self.descriptor_names)

        # Create the descriptor calculator
        self._calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self.descriptor_names
        )

    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RDKit 2D descriptors for a batch of SMILES.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            features: ndarray of shape (n, output_dim), dtype=float32.
            mask: bool array of shape (n,), True where parsing succeeded.
        """
        n = len(smiles_list)
        features, mask = self._empty_outputs(n, dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            if not smi or not smi.strip():
                continue

            # Handle multi-component SMILES (take first component)
            components = split_multi_smiles(smi)
            smi_clean = components[0] if components else ""
            if not smi_clean or smi_clean == "(无)":
                continue

            mol = Chem.MolFromSmiles(smi_clean)
            if mol is None:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    desc_values = self._calculator.CalcDescriptors(mol)

                arr = np.array(desc_values, dtype=np.float32)
                # Replace NaN and Inf with 0
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

                features[i] = arr
                mask[i] = True

            except Exception:
                continue

        return features, mask

    def get_descriptor_names(self) -> List[str]:
        """Return the list of descriptor names in order."""
        return list(self.descriptor_names)
