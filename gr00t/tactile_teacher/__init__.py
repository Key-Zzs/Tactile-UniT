"""Wrench-history contact-state teacher utilities.

S1 is deliberately isolated from the UniT tokenizer.  The canonical input is
the 60-D per-frame ``observation.tactile_force`` vector from the public T-Rex
LeRobot v3 dataset.  Image/deformation branches are outside the current
milestone.
"""

from .dataset import EpisodeData, TactileEpisodeStore
from .normalization import RobustFeatureStats
from .schema import FEATURE_ORDER, TactileDataContract
from .split import EpisodeSplit, build_episode_split
from .window import TemporalWindow, resample_window

__all__ = [
    "EpisodeData",
    "EpisodeSplit",
    "FEATURE_ORDER",
    "RobustFeatureStats",
    "TactileDataContract",
    "TactileEpisodeStore",
    "TemporalWindow",
    "build_episode_split",
    "resample_window",
]
