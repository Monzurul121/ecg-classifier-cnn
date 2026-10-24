# ECG Heartbeat Classification with a Compact 1-D CNN

A compact one-dimensional convolutional neural network that labels each detected heartbeat in a single-lead ECG as normal, supraventricular ectopic or ventricular ectopic, using the beat waveform and its RR-interval context.

> **Status: preview.** This repository documents the method and its interface. The implementation will be released later.

## Overview

Beat-by-beat classification is a core task in ambulatory ECG analysis. Reports from Holter and patch monitors count premature beats, and those counts help clinicians assess arrhythmia burden.

A convolutional network can learn useful waveform features directly from the beat, without hand-designed features. A single beat window, however, says little about timing. Premature beats are often recognized as much by *when* they occur as by their shape. This model therefore adds RR-interval context to the waveform as extra input channels. The network is deliberately small, which keeps it fast to train and run.

Each detected beat is assigned one of three classes derived from the AAMI grouping:

- **N**: normal
- **S**: supraventricular ectopic
- **V**: ventricular ectopic

## How it works

1. **Input.** A single-lead ECG and the beat locations from any QRS detector, for example XQRS from the WFDB Python package.
2. **Preprocessing.** The signal is bandpass filtered, resampled to a common rate and normalized using statistics from the start of the record.
3. **Beat window.** A fixed window around each beat location is extracted and resampled to 128 samples.
4. **RR-context channels.** Four values describe how early or late the beat arrives compared with the local rhythm and its neighboring beats. Each value is repeated along the 128 samples as its own channel, which gives an input of size 128 × 5.
5. **Classification.** The CNN below outputs a probability for each of N, S and V.
6. **Training.** The network is trained with class weights, so the rare S and V beats are not ignored.

## Architecture

| Stage | Layers | Output size |
|---|---|---|
| Input | beat waveform + 4 RR-context channels | 128 × 5 |
| Block 1 | Conv1D (32 filters, kernel 5) → BatchNorm → ReLU → MaxPool (2) | 64 × 32 |
| Block 2 | Conv1D (64 filters, kernel 3) → BatchNorm → ReLU → MaxPool (2) | 32 × 64 |
| Block 3 | Conv1D (128 filters, kernel 3) → BatchNorm → ReLU | 32 × 128 |
| Head | Global average pooling → Dropout (0.3) → Dense (3) → Softmax | 3 |

The network has about 33,000 parameters.

## Key properties

- **Compact:** fast to train and to run, also on a CPU.
- **Waveform plus rhythm:** combines learned shape features with explicit RR-interval timing.
- **Probabilistic output:** class probabilities for every beat.
- **Detector-agnostic:** works with beat locations from any QRS detector.

## Measuring performance

Results are reported per class, because overall accuracy is dominated by the many normal beats:

- **F1 score** for each of N, S and V;
- **macro-F1**, the unweighted mean of the three F1 scores.

## Limitations

- Results depend on accurate beat locations from the QRS detector.
- Only three classes are covered. Fusion and unclassifiable beats are not handled.
- Like any trained model, it works best on recordings similar to its training data.

## Typical uses

Automated beat-level analysis of ambulatory (Holter) ECG, for example counting premature ventricular and supraventricular beats. The small model also suits resource-limited settings.

## Interface (preview)

```python
from cnn_classifier import ECGRecord, CNNBeatClassifier

train = [ECGRecord(signal, fs=360, beats=r_peaks, labels=beat_labels), ...]
clf = CNNBeatClassifier(seed=0).fit(train)

test = ECGRecord(signal, fs=360, beats=detected_beats)
labels = clf.predict(test)          # "N", "S" or "V" for each beat
proba = clf.predict_proba(test)     # shape (n_beats, 3)
```

In this preview, the methods raise `NotImplementedError`. They show the interface that the released code will provide.

## Roadmap

- Release the implementation.
- Add usage examples and documentation.

## Copyright

Copyright © 2026 Md Monzurul Islam Rabu. All rights reserved.

No permission is granted to use, copy, modify or redistribute the contents of this repository. An open-source license will be added when the code is released.
