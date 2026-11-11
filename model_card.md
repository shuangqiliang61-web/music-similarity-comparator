# Model card

- Model: `StandardScaler + LogisticRegression (multinomial, 3 classes)`
- Features: `landmark, sim_timbre, sim_melody`
- Training data: 100 labelled pairs ({'copy': 35, 'cover': 33, 'unrelated': 32})
- Audio: mono, 22050 Hz, first 60 s
- **Cross-validated accuracy: 66.8%** (macro-F1 0.644, 5-fold x 4 repeats)
- In-sample accuracy 68.0% is shown only as a sanity check, **not** a performance claim

## Feature meanings

- `landmark`: spectral-peak fingerprint match rate (same recording)
- `sim_timbre`: MFCC sequence sub-sequence DTW similarity (timbre)
- `sim_melody`: key-invariant Chroma DTW similarity (melody)

## Limitations

- Only 100 labelled pairs from a single source; accuracy varies 0.52-0.88 across 20 different train/test splits (mean ~0.67).
- Cover detection is the weakest class (see README / 评测报告.md).
- Not suitable for production copyright decisions.

## Privacy

Contains model coefficients only. No audio, filenames or IDs.

## Reproduce

```bash
conda activate music
python train_model.py
python predict.py song1.mp3 song2.mp3
```
