"""ECG heartbeat classification with a compact 1-D CNN: public interface
(preview).

The implementation will be released later. Until then, the classifier methods
below raise NotImplementedError.

Copyright (c) 2026 Md Monzurul Islam Rabu. All rights reserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["CLASSES", "ECGRecord", "CNNBeatClassifier"]

CLASSES = ("N", "S", "V")

RELEASE_MESSAGE = (
    "This is a preview of the interface. The implementation will be released "
    "later. See README.md."
)


@dataclass
class ECGRecord:
    """One single-lead ECG record and its beats.

    Attributes
    ----------
    signal : np.ndarray
        1-D float array: single-lead ECG.
    fs : float
        Sampling rate in Hz.
    beats : np.ndarray
        Beat locations as sample indices, for example from a QRS detector.
    labels : sequence of str, optional
        Class of each beat ("N", "S" or "V"). Needed for training only.
    """

    signal: np.ndarray
    fs: float
    beats: np.ndarray
    labels: Optional[Sequence[str]] = None


class CNNBeatClassifier:
    """Compact 1-D CNN that labels each beat as N, S or V.

    Input per beat: the beat waveform resampled to 128 samples plus four
    RR-interval context channels (128 x 5).
    """

    classes_ = CLASSES
    input_shape = (128, 5)

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def fit(self, records: Sequence[ECGRecord]) -> "CNNBeatClassifier":
        """Train on labeled records. Returns the fitted classifier."""
        raise NotImplementedError(RELEASE_MESSAGE)

    def predict_proba(self, record: ECGRecord) -> np.ndarray:
        """Class probabilities for each beat, shape (n_beats, 3), columns in
        the order of ``classes_``."""
        raise NotImplementedError(RELEASE_MESSAGE)

    def predict(self, record: ECGRecord) -> np.ndarray:
        """Predicted class ("N", "S" or "V") for each beat."""
        raise NotImplementedError(RELEASE_MESSAGE)
