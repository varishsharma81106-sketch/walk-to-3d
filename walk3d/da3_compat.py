# Patches DA3 imports to defer optional export backends until export is requested.
"""Keep DA3's optional export backends out of the inference import path."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _stub_heavy_optional_imports() -> None:
    """Keep unused optional backends unavailable without installing them."""
    for name in ("e3nn", "gsplat", "open3d", "pycolmap", "xformers"):
        module = types.ModuleType(name)
        module.__path__ = []

        def missing_attribute(attribute: str, *, _name=name):
            if attribute.startswith("__"):
                raise AttributeError(attribute)
            raise ModuleNotFoundError(
                f"{_name} is an optional DA3 backend dependency and is not installed "
                f"(requested attribute: {_name}.{attribute})",
                name=_name,
            )

        module.__getattr__ = missing_attribute
        sys.modules.setdefault(name, module)

    # DA3 imports evo's trajectory type even when pose alignment is unused.
    evo = types.ModuleType("evo")
    evo.__path__ = []
    core = types.ModuleType("evo.core")
    core.__path__ = []
    trajectory = types.ModuleType("evo.core.trajectory")

    class PosePath3D:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "evo is required only for DA3 input-pose alignment; this run does not use it",
                name="evo",
            )

    trajectory.PosePath3D = PosePath3D
    evo.core = core
    core.trajectory = trajectory
    sys.modules.setdefault("evo", evo)
    sys.modules.setdefault("evo.core", core)
    sys.modules.setdefault("evo.core.trajectory", trajectory)


def load_depth_anything3():
    """Import DA3 without eagerly importing moviepy/open3d/pycolmap exporters."""
    _stub_heavy_optional_imports()
    module_name = "depth_anything_3.utils.export"
    if module_name not in sys.modules:
        export_package = types.ModuleType(module_name)
        export_package.__package__ = module_name
        export_package.__path__ = [
            str(
                Path(__file__).resolve().parents[1]
                / "third_party"
                / "da3"
                / "src"
                / "depth_anything_3"
                / "utils"
                / "export"
            )
        ]

        def export(prediction, export_format: str, export_dir: str, **kwargs):
            if "-" in export_format:
                for item in export_format.split("-"):
                    export(prediction, item, export_dir, **kwargs)
                return
            backends = {
                "glb": ("glb", "export_to_glb"),
                "mini_npz": ("npz", "export_to_mini_npz"),
                "npz": ("npz", "export_to_npz"),
                "feat_vis": ("feat_vis", "export_to_feat_vis"),
                "depth_vis": ("depth_vis", "export_to_depth_vis"),
                "gs_ply": ("gs", "export_to_gs_ply"),
                "gs_video": ("gs", "export_to_gs_video"),
                "colmap": ("colmap", "export_to_colmap"),
            }
            try:
                module, function = backends[export_format]
            except KeyError as error:
                raise ValueError(f"Unsupported export format: {export_format}") from error
            backend = importlib.import_module(f"{module_name}.{module}")
            return getattr(backend, function)(
                prediction, export_dir, **kwargs.get(export_format, {})
            )

        export_package.export = export
        export_package.__all__ = ["export"]
        sys.modules[module_name] = export_package

    from depth_anything_3.api import DepthAnything3

    return DepthAnything3
