Text Input
     │
Text Encoder
     │
     ▼

Audio Input ─► Audio Encoder ─┐
                              │
Vision Input ─► Vision Encoder│
                              ▼
                   Cross Modal Attention
                              │
                              ▼
                    Semantic Transformer
                              │
                              ▼
                     Audio Recovery Module
                              │
                    Residual Fusion
                              │
                           Dropout
                              │
                              ▼
                    Sentiment Classifier
                              │
                              ▼
                   Positive / Negative






# Missing Modality Emotion Recognition

Internship project focused on multimodal sentiment analysis using the CMU-MOSEI dataset.

## Dataset

- CMU-MOSEI
- aligned_50.pkl

## Modalities

- Text
- Audio
- Vision

## Current Baseline

Text Encoder
→ Audio Encoder
→ Vision Encoder
→ Transformer Fusion
→ Sentiment Classifier

## Task

Binary Sentiment Classification

- Negative
- Positive

Neutral samples are excluded.
