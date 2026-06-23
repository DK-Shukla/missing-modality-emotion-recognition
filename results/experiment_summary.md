# Missing Modality Emotion Recognition

## Dataset
CMU-MOSEI

## Baseline Results

| Experiment | Accuracy |
|------------|-----------|
| Full Modalities | 83.30% |
| Audio Missing | 74.05% |
| Vision Missing | 82.97% |
| Audio + Vision Missing | 78.23% |

---

## Proposed Method

Components:
- Text Encoder
- Audio Encoder
- Vision Encoder
- Semantic Transformer
- Audio Recovery Network
- Transformer Fusion
- Sentiment Classifier

Loss:

Total Loss =
Classification Loss +
0.1 × Reconstruction Loss

---

## Proposed Results

| Experiment | Accuracy |
|------------|-----------|
| Audio Missing | 78.23% |
| Audio + Vision Missing | 81.32% |

---

## Improvement

Audio Missing:
74.05% → 78.23%
(+4.18%)

Audio + Vision Missing:
78.23% → 81.32%
(+3.09%)

---

## Conclusion

The proposed Semantic Transformer and Audio Recovery Network improve robustness under missing modality conditions by recovering semantic information before classification.