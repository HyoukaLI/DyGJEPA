"""SG-JEPA reproduction and RCPS-JEPA comparison package."""

from .data import DynamicGraph, Snapshot
from .jodie_baseline import JODIELinkBaseline
from .sg_jepa import SGJEPA
from .rcps_jepa import RCPSJEPA
from .snapshot_ssl_baselines import (
    CLDGLinkBaseline,
    DVGMAELinkBaseline,
    MaskDGNNLinkBaseline,
)

__all__ = [
    "DynamicGraph",
    "Snapshot",
    "SGJEPA",
    "JODIELinkBaseline",
    "RCPSJEPA",
    "CLDGLinkBaseline",
    "MaskDGNNLinkBaseline",
    "DVGMAELinkBaseline",
]
