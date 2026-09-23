from __future__ import annotations

from .base import BaseDetector, assign_levels, iter_groups
try:
    from .autoencoder import DenseAutoencoderDetector, LSTMAutoencoderDetector
except ImportError:
    DenseAutoencoderDetector = None
    LSTMAutoencoderDetector = None

from .isolation_forest import IsolationForestDetector
from .statistical import IQRTukeyDetector, RollingIQRDetector, ZScoreDetector

__all__ = [
    "BaseDetector",
    "DenseAutoencoderDetector",
    "IQRTukeyDetector",
    "IsolationForestDetector",
    "LSTMAutoencoderDetector",
    "RollingIQRDetector",
    "ZScoreDetector",
    "assign_levels",
    "iter_groups",
]
