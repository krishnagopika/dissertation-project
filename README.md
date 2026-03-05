# Multimodal Fusion for Sentiment and Emotion Extraction
### A Study of Robustness Across Australian and UK English Dialects
**MSc Dissertation — Department of Computer Science, University of Warwick**

---

## Overview

This dissertation proposes a novel multimodal fusion pipeline for sentiment and emotion extraction from speech. The pipeline fuses **Voxtral-based transcription** and **internal acoustic feature extraction** with an **ASR-aware fine-tuned XLM-RoBERTa** text classifier via a learned fusion layer.

The key contribution is an **ASR-aware fine-tuning strategy** — training the text classifier on Voxtral-transcribed text rather than clean gold-standard text, making it robust to transcription noise introduced by accented speech. The pipeline is evaluated on MELD, CMU-MOSI, RAVDESS, and GoEmotions, and stress-tested for dialect robustness across Australian and UK English regional dialects.

---

## Pipeline Architecture

```
Raw Audio
    │
    ├──── Voxtral (ASR) ──────────► Transcript ──► XLM-RoBERTa (ASR-aware fine-tuned) ──► Text Representation ──┐
    │                                                                                                              ├──► Fusion Layer ──► Sentiment + Emotion Labels
    └──── Voxtral (Whisper Large-v3 Encoder) ──► Internal Acoustic Embeddings ───────────────────────────────────┘
```

### Branches
- **Text Branch**: Voxtral transcribes raw audio → XLM-RoBERTa fine-tuned on ASR-output text classifies sentiment/emotion
- **Acoustic Branch**: Internal Whisper Large-v3 encoder embeddings extracted directly from Voxtral capture prosodic features (tone, pitch, rhythm)
- **Fusion Layer**: Concatenation of text and acoustic representations → classification head produces simultaneous sentiment and emotion labels

---

## Key Contributions

1. A novel multimodal fusion pipeline combining Voxtral acoustic embeddings with ASR-aware XLM-RoBERTa text classification
2. An ASR-aware fine-tuning strategy that improves robustness to dialect transcription noise (extending [Taghavi et al., 2023])
3. Systematic dialect robustness evaluation across Australian English and UK regional dialects
4. Benchmarking study comparing Voxtral Mini (3B) and Voxtral Small (24B) as the ASR backbone

---

## Datasets

### Training & Evaluation
| Dataset | Description |
|---|---|
| [MELD](https://aclanthology.org/P19-1050/) | ~13,000 utterances from Friends TV series; 7 emotion + 3 sentiment classes |
| [CMU-MOSI](https://arxiv.org/abs/1606.06259) | Opinion video segments with continuous sentiment scores |
| [RAVDESS](https://doi.org/10.1371/journal.pone.0196391) | Acted speech recordings with emotion labels |
| [GoEmotions](https://aclanthology.org/2020.acl-main.372/) | 58,000 Reddit comments across 27 fine-grained emotion categories |

### Dialect Robustness Evaluation (inference only)
| Dataset | Description |
|---|---|
| [Common Voice en-AU](https://arxiv.org/abs/1912.06670) | Australian English speech |
| [English Dialects](https://aclanthology.org/2020.lrec-1.804) | Southern, Midlands, Northern, Welsh, and Scottish English |

---

## Baselines

| # | Baseline | Description |
|---|---|---|
| 1 | Text-only (clean) | XLM-RoBERTa on gold-standard transcripts |
| 2 | Text-only (ASR) | XLM-RoBERTa on Voxtral transcripts |
| 3 | Off-the-shelf | CardiffNLP XLM-T |
| 4 | Acoustic-only | Voxtral internal embeddings, no text branch |
| 5 | Fusion (Mini) | Full pipeline with Voxtral Mini (3B) |
| 6 | Fusion (Small) | Full pipeline with Voxtral Small (24B) |

---

## Ablation Study

| ID | Ablation | Purpose |
|---|---|---|
| A1 | Remove acoustic branch | Tests whether text alone is sufficient |
| A2 | Remove text branch | Tests whether acoustics alone is sufficient |
| A3 | Clean-text vs ASR-text fine-tuning | Validates ASR-aware training strategy |
| A4 | Voxtral vs Whisper transcription | Validates Voxtral's specific contribution |
| A5 | Concatenation vs sum fusion | Validates fusion architecture choice |

---

## Evaluation Metrics

- **Weighted F1** — primary metric, accounts for class imbalance
- **Accuracy** — secondary metric
- **Per-class F1** — identifies hardest emotion categories
- **Per-accent breakdown** — dialect robustness across AU and UK subsets

---

## Hardware

All experiments run on the `wmlg-ada` partition:
- 4x Nvidia L40S GPUs (48GB each) — 192GB total VRAM
- Voxtral Small (24B) runs across two GPUs
- XLM-RoBERTa fine-tuning on a single GPU

---

## Timeline

| Month | Milestone |
|---|---|
| March 2026 | Environment setup, GPU config, dataset preprocessing, basic Voxtral inference pipeline, embedding extraction proof-of-concept |
| April 2026 | XLM-RoBERTa ASR-aware fine-tuning, acoustic embedding extraction, baseline experiments |
| May 2026 | Fusion layer implementation, full pipeline training, all ablation experiments |
| June 2026 | Error analysis, dialect robustness evaluation, figures and tables |
| July 2026 | Dissertation writing (all results finalised) |

### Submission Deadlines
| Assessment | Weight | Deadline |
|---|---|---|
| Presentation | 5% | End of April – mid May 2026 (in-person, 20 min + 5 min discussion) |
| Interim Report | 15% | 16 July 2026, noon |
| Dissertation Report | 80% | 8 September 2026, noon |

---

## References

### Foundational Models
| Paper | Link |
|---|---|
| CTC — Graves et al. (2006) | https://www.cs.toronto.edu/~graves/icml_2006.pdf |
| Attention Is All You Need — Vaswani et al. (2017) | https://arxiv.org/abs/1706.03762 |
| BERT — Devlin et al. (2018) | https://aclanthology.org/N19-1423/ |

### Self-Supervised Speech
| Paper | Link |
|---|---|
| wav2vec — Schneider et al. (2019) | https://arxiv.org/abs/1904.05862 |
| vq-wav2vec — Baevski et al. (2019) | https://arxiv.org/abs/1910.05453 |
| wav2vec 2.0 — Baevski et al. (2020) | https://arxiv.org/abs/2006.11477 |

### Multilingual Speech Transcription
| Paper | Link |
|---|---|
| Whisper — Radford et al. (2022) | https://arxiv.org/abs/2212.04356 |
| WhisperX — Bain et al. (2023) | https://arxiv.org/abs/2303.00747 |
| SeamlessM4T — Barrault et al. (2023) | https://arxiv.org/abs/2308.11596 |
| Voxtral — Mistral AI (2025) | https://arxiv.org/abs/2507.13264 |

### Text Representations
| Paper | Link |
|---|---|
| XLM-RoBERTa — Conneau et al. (2020) | https://aclanthology.org/2020.acl-main.747/ |
| XLM-T — Barbieri et al. (2022) | https://aclanthology.org/2022.lrec-1.27/ |

### Emotion & Sentiment
| Paper | Link |
|---|---|
| SER Survey — Schuller (2018) | https://doi.org/10.1145/3129340 |
| A Change of Heart — Taghavi et al. (2023) | https://arxiv.org/abs/2307.11584 |
| Tensor Fusion Network — Zadeh et al. (2017) | https://aclanthology.org/D17-1115/ |

### Models & Repositories
| Resource | Link |
|---|---|
| Voxtral Mini | https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 |
| Voxtral Small | https://huggingface.co/mistralai/Voxtral-Small-24B-2507 |
| XLM-RoBERTa | https://huggingface.co/FacebookAI/xlm-roberta-base |
| CardiffNLP XLM-T | https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment |
| NVIDIA NeMo | https://github.com/NVIDIA-NeMo/NeMo |
| Kimi | https://arxiv.org/pdf/2504.18425v1 |
