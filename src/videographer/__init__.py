"""VideoGrapher - Object tracking with lens distortion correction."""

from .calibration import MeterStickCalibrator
from .selectors import ObjectSelector, ReferencePointSelector
from .tracker import ObjectTracker

__version__ = "0.1.0"

__all__ = [
    "ObjectTracker",
    "MeterStickCalibrator",
    "ObjectSelector",
    "ReferencePointSelector",
]
