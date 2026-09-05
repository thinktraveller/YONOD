"""Molecular descriptors package.

使用延迟导入避免预先加载重依赖（如 torch_geometric）。
"""

from .base import BaseDescriptor

# 延迟导入重依赖描述符，避免 import 时触发模块加载
# 用户代码应按需导入：from yonod.descriptors.fisd import FISDDescriptor

__all__ = [
    "BaseDescriptor",
    "MorganDescriptor",
    "MFPDescriptor",
    "OHEFeature",
    "FeatureSpec",
    "FeatureProvider",
    "get_feature_provider",
    "normalise_feature_specs",
    "ATMOMACCSDescriptor",
    "FISDDescriptor",
    "MolMetaLMDescriptor",
    "MAFDescriptor",
    "RDKit2DDescriptor",
    "DRFPDescriptor",
]


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    if name == "MorganDescriptor":
        from .morgan import MorganDescriptor
        return MorganDescriptor
    elif name == "MFPDescriptor":
        from .mfp import MFPDescriptor
        return MFPDescriptor
    elif name == "OHEFeature":
        from .ohe import OHEFeature
        return OHEFeature
    elif name in {"FeatureSpec", "FeatureProvider", "get_feature_provider", "normalise_feature_specs"}:
        from .registry import FeatureProvider, FeatureSpec, get_feature_provider, normalise_feature_specs
        return {
            "FeatureSpec": FeatureSpec,
            "FeatureProvider": FeatureProvider,
            "get_feature_provider": get_feature_provider,
            "normalise_feature_specs": normalise_feature_specs,
        }[name]
    elif name == "ATMOMACCSDescriptor":
        from .atmomaccs import ATMOMACCSDescriptor
        return ATMOMACCSDescriptor
    elif name == "FISDDescriptor":
        from .fisd import FISDDescriptor
        return FISDDescriptor
    elif name == "MolMetaLMDescriptor":
        from .molmetalm import MolMetaLMDescriptor
        return MolMetaLMDescriptor
    elif name == "MAFDescriptor":
        from .maf import MAFDescriptor
        return MAFDescriptor
    elif name == "RDKit2DDescriptor":
        from .rdkit2d import RDKit2DDescriptor
        return RDKit2DDescriptor
    elif name == "DRFPDescriptor":
        from .drfp_desc import DRFPDescriptor
        return DRFPDescriptor
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
