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

---

## Repository Structure

For a full walkthrough of what each approach does and every result table, see **`PROGRESS.md`**. This section is a code-map: what lives where and why.

### Top-level layout

```
dissertation-project/
├── CLAUDE.md                              # Coding constitution (device handling, config,
│                                          #   checkpointing, Slurm rules, MELD label schema)
├── PROGRESS.md                            # Full ablation report — every approach + results
├── README.md                              # This file
├── Requirements.txt                       # Python 3.12 deps (torch, transformers, vllm, ...)
├── voxtral_architecture.md                # Notes on Voxtral's internal architecture
├── Research_Proposal_...pdf               # MSc research proposal
├── src/                                   # All source code (see below)
├── results/                               # All metric JSONs, confusion matrices, loss curves
├── checkpoints/                           # Local checkpoint dir (large weights are on scratch)
├── logs/                                  # Slurm stdout/stderr + Python logs
└── notebooks/                             # Reserved for exploratory analysis (empty)
```

### `src/` — all source code

```
src/
├── configs/                               # YAML config files (all paths, hyperparams here)
│   ├── mini.yaml                          # ★ primary — XLM-R-base + Voxtral-Mini-3B
│   ├── mini_asr.yaml                      # variant that swaps in ASR-transcript training
│   └── small.yaml                         # Voxtral-Small-24B (2 GPUs) — larger backbone
│
├── data/                                  # Dataset loaders (Dataset subclasses)
│   ├── meld.py                            # MELD utterance loader (7-emo + 3-sent)
│   ├── ravdess.py                         # RAVDESS (Ryerson emotion audio) loader
│   ├── cmu_mosi.py                        # CMU-MOSI opinion segments
│   ├── goemotions.py                      # GoEmotions Reddit corpus
│   └── dialect.py                         # Common Voice AU + English Dialects (eval only)
│
├── models/                                # nn.Module classes
│   ├── voxtral.py                         # VoxtralWrapper — audio encoder + ASR
│   ├── xlmr.py                            # XLMRobertaClassifier — joint emo + sent heads
│   ├── fusion.py                          # 4 fusion heads: FusionModel (concat), SumFusion,
│   │                                      #   GatedFusion, CrossModalGating + FusionPipeline
│   ├── acoustic_classifier.py             # AcousticEmotionClassifier for RAVDESS pretrain —
│   │                                      #   acoustic_proj submodule transferable to fusion
│   ├── context_lstm.py                    # LSTM over dialogue-context embeddings
│   └── voxtral_encoder_classifier.py      # Direct emotion head on frozen Voxtral encoder
│
├── preprocessing/                         # Cache generation (run once, results live on disk)
│   ├── transcribe_all.py                  # Pass 1: Voxtral ASR (vllm) → transcripts.json
│   │                                      # Pass 2: Voxtral encoder → embeddings.pt (1280-d)
│   ├── transcribe_whisper.py              # Whisper Large-v3 ASR baseline
│   ├── extract_text_embeddings.py         # XLM-R [CLS] over gold text → text_embeddings.pt
│   ├── extract_text_embeddings_asr.py     # Same but over ASR transcripts (ablation A3)
│   ├── extract_ravdess_embeddings.py      # Voxtral encoder over RAVDESS 1440 clips
│   └── augment_transcripts.py             # Voxtral paraphrase generation for minority classes
│
├── training/                              # Training scripts (one per pipeline phase)
│   ├── finetune.py                        # ★ Phase 1: XLM-R fine-tune on ASR transcripts
│   │                                      #   Supports focal loss, WeightedRandomSampler,
│   │                                      #   paraphrase augmentation via config flags
│   ├── train_fusion.py                    # ★ Phase 2: fusion training on cached embeddings
│   │                                      #   Supports RAVDESS-pretrained backbone warm-start
│   ├── train_classical.py                 # SVM + XGBoost on 2048-d fused features
│   ├── train_context.py                   # Context-aware LSTM over dialogue-level embeddings
│   ├── finetune_encoder.py                # Direct Voxtral encoder fine-tune (alternative)
│   ├── pretrain_ravdess.py                # Phase A: 8-class acoustic classifier on RAVDESS
│   └── pretrain_combined.py               # Combined RAVDESS+MELD 7-class acoustic training
│
├── evaluation/                            # Metrics + evaluation scripts
│   ├── metrics.py                         # Weighted/macro F1, per-class F1, confusion matrix
│   ├── evaluate.py                        # Fusion eval on test set (per fusion_type)
│   ├── evaluate_text_only.py              # XLM-R text-only baseline eval
│   ├── voxtral_zeroshot.py                # Zero-shot ERC via direct Voxtral prompting
│   ├── dialect_eval.py                    # Cross-dialect robustness eval (AU + UK regional)
│   ├── late_fusion.py                     # 5 late-fusion strategies + oracle upper bound
│   ├── extract_acoustic_probs.py          # Cache acoustic-only softmax outputs
│   ├── extract_text_probs.py              # Cache text-only softmax outputs
│   ├── compute_wer.py                     # WER / CER / MER via jiwer
│   └── export_wer_pairs.py                # Export (gold, asr) pairs for qualitative review
│
├── ablations/                             # A1-A5 ablation drivers from the research proposal
│   ├── a1_text_only.py                    # A1: drop acoustic branch
│   ├── a2_acoustic_only.py                # A2: drop text branch
│   ├── a3_clean_vs_asr.py                 # A3: clean gold vs ASR fine-tuning
│   ├── a4_voxtral_vs_whisper.py           # A4: Voxtral vs Whisper transcription
│   └── a5_concat_vs_sum.py                # A5: concat vs sum fusion
│
├── scripts/                               # Slurm sbatch files + shell wrappers
│   ├── transcribe.sbatch                  # Pass 1+2 Voxtral (Mini) on all MELD splits
│   ├── transcribe_small.sbatch            # Same but with Voxtral-Small (2 GPUs)
│   ├── transcribe_test.sbatch             # MELD test-split only (smoke test)
│   ├── transcribe_whisper.sbatch          # Whisper Large-v3 baseline transcription
│   ├── extract_text_embeddings.sbatch     # Re-extract text embeddings per Phase 1 checkpoint
│   ├── extract_ravdess_embeddings.sbatch  # Voxtral encoder over RAVDESS
│   ├── augment.sbatch                     # Voxtral paraphrase generation
│   ├── finetune.sbatch                    # ★ Phase 1: XLM-R fine-tune
│   ├── finetune_encoder.sbatch            # Direct Voxtral encoder fine-tune
│   ├── train_fusion.sbatch                # ★ Phase 2: all 4 fusion types back-to-back
│   ├── train_classical.sbatch             # SVM + XGBoost sweep
│   ├── train_context.sbatch               # Context-LSTM training
│   ├── evaluate.sbatch                    # Text-only + all fusion variants on test
│   ├── evaluate_ablation.sbatch           # ★ One-shot: extract embs → train fusion → eval
│   │                                      #   Takes <TAG> <PHASE1_CKPT> as positional args
│   ├── pretrain_ravdess.sbatch            # RAVDESS Phase A backbone training
│   ├── pretrain_combined.sbatch           # Joint RAVDESS+MELD acoustic training
│   ├── compute_wer.sbatch                 # WER analysis job
│   ├── context_asr.sbatch                 # Context model on ASR transcripts
│   ├── fusion_prep.sbatch                 # Pre-fusion cache prep (embeddings)
│   ├── voxtral_zeroshot.sbatch            # Voxtral zero-shot ERC eval job
│   ├── run_baselines.sh                   # Shell wrapper — submit all baselines
│   ├── run_ablations.sh                   # Shell wrapper — submit A1-A5 ablations
│   └── run_dialect_eval.sh                # Shell wrapper — dialect robustness sweep
│
└── utils/                                 # Small shared helpers
    ├── __init__.py                        # Re-exports get_device, set_seed, load_config, ...
    ├── config.py                          # YAML loader
    ├── logging.py                         # setup_logging (console + file)
    └── seed.py                            # set_seed for full reproducibility
```

### `results/mini/` — everything measured lives here

```
results/mini/
├── test_results_text_only*.json                  # XLM-R text-only per Phase 1 variant
├── test_results_{concat,sum,gated,crossmodal}*.json  # Fusion per (arch × ablation)
├── classical_{svm,xgboost}_{emotion,sentiment}*.json # Classical baselines (dev)
├── combined_acoustic_results.json                # Combined RAVDESS+MELD acoustic-only
├── ravdess_pretrain_results.json                 # RAVDESS 8-class Phase A test
├── late_fusion_results.json                      # 5 late-fusion strategies + oracle upper
├── voxtral_zeroshot_{dev,test}_metrics.json      # Zero-shot ERC metrics
├── voxtral_zeroshot_{dev,test}_predictions.json  # Zero-shot predictions per utterance
├── wer_analysis.json                             # Voxtral WER on MELD
├── wer_analysis_whisper.json                     # Whisper baseline WER
├── acoustic_probs.json                           # Cached acoustic-only softmax outputs
├── xlmr_text_probs.json                          # Cached text-only softmax outputs
├── confusion_{emotion,sentiment}_*.png           # Confusion matrices per config
├── phase1_loss_curve*.png                        # XLM-R fine-tune loss curves
├── phase2_loss_curve_*.png                       # Fusion loss curves per architecture
├── ravdess_pretrain_loss_curve.png               # RAVDESS Phase A curve
└── combined_acoustic_loss_curve.png              # Combined training curve
```

**Tag convention on filenames**: base name = fusion architecture or `text_only`; suffix = Phase 1 recipe. Examples:
- `test_results_gated.json` = baseline gated fusion
- `test_results_gated_focal_sampler.json` = gated fusion using focal+sampler Phase 1
- `test_results_gated_aug.json` = gated fusion using paraphrase-aug Phase 1
- `test_results_gated_ravdesspre.json` = gated fusion warm-started from RAVDESS backbone
- `test_results_gated_focal_sampler_ravdesspre.json` = both
- `test_results_gated_no_pretrain.json` = control (ravdess warm-start disabled)

---

## Data & checkpoints on HPC scratch (`/dcs/large/u5734759/`)

Cached embeddings, checkpoints, HF model weights, and the venv are all too large for the home directory quota. They live under the shared HPC scratch. **Everything under `/dcs/large/u5734759/` should be treated as regenerable** — nothing here is version-controlled.

**Current disk usage** (as of writing):

| Path | Size | What lives here |
|---|---:|---|
| `venv/` | 11 GB | Python 3.12 virtualenv (all sbatch jobs `source /dcs/large/u5734759/venv/bin/activate`) |
| `hf_cache/` | 22 GB | HuggingFace model download cache (Voxtral-Mini, XLM-R, Whisper) |
| `data/meld_raw/` | 21 GB | Extracted MELD .tar.gz archives (train/dev/test .mp4 clips + CSVs) |
| `data/ravdess/` | 566 MB | RAVDESS `Actor_01/` … `Actor_24/` .wav files |
| `data/meld_embeddings/` | 71 MB | Voxtral encoder outputs, one dict per split (1280-d per utt) |
| `data/meld_text_embeddings/` | 42 MB | XLM-R [CLS] from **gold** text, per Phase 1 checkpoint (re-extracted per ablation) |
| `data/meld_text_embeddings_asr/` | 42 MB | XLM-R [CLS] from **ASR** text (used for A3 clean-vs-ASR ablation) |
| `data/ravdess_embeddings/` | 7.5 MB | Voxtral encoder over 1440 RAVDESS clips |
| `data/meld_transcripts/` | 2.4 MB | ASR transcripts + paraphrases (JSON) |
| `checkpoints/mini/` | 8.8 GB | Current Phase 1 XLM-R + all 4 Phase 2 fusion best_model.pt |
| `checkpoints/mini_baseline/` | 3.2 GB | Archived original baseline Phase 1 (before ablation runs) |
| `checkpoints/mini_focal_sampler/` | 6.3 GB | Archived focal+sampler Phase 1 |
| `checkpoints/mini_aug/` | 3.2 GB | Archived Voxtral-paraphrase-aug Phase 1 |
| `checkpoints/mini_asr_transcripts/` | 7.4 GB | Archived ASR-transcript-trained Phase 1 (A3) |
| `checkpoints/ravdess/` | 2.6 MB | RAVDESS Phase A acoustic backbone |
| `checkpoints/combined/` | 5.1 MB | Combined RAVDESS+MELD Phase A backbone + full model |

### Full scratch layout

```
/dcs/large/u5734759/
├── venv/                                  # Python 3.12 virtualenv — activated by every sbatch
├── hf_cache/                              # HF_HOME and TRANSFORMERS_CACHE point here
│
├── data/
│   ├── meld_raw/                          # MELD raw audio + CSVs
│   │   ├── train_sent_emo.csv             # 9989 rows: dialogue_id, utterance_id, utterance,
│   │   │                                  #    speaker, emotion, sentiment
│   │   ├── dev_sent_emo.csv               # 1109 rows
│   │   ├── test_sent_emo.csv              # 2610 rows
│   │   ├── train/                         # .mp4 utterance clips (dia{d}_utt{u}.mp4)
│   │   ├── dev/                           # (also .mp4 clips)
│   │   ├── test/                          # (also .mp4 clips)
│   │   ├── {train,dev,test}.tar.gz        # Original archives (kept for re-extraction if needed)
│   │   ├── train_splits/                  # Alt structure from MELD distribution
│   │   ├── dev_splits_complete/           # Alt structure from MELD distribution
│   │   ├── output_repeated_splits_test/   # Alt structure from MELD distribution
│   │   └── README.txt                     # MELD's own README
│   │
│   ├── meld_transcripts/                  # ASR outputs + augmentation
│   │   ├── train_transcripts.json         # Voxtral ASR — {"dia0_utt0": "verbatim text", ...}
│   │   ├── dev_transcripts.json           # (~9989 / 1109 / 2610 keys respectively)
│   │   ├── test_transcripts.json
│   │   ├── train_transcripts_whisper.json # Whisper Large-v3 ASR (WER baseline)
│   │   ├── dev_transcripts_whisper.json
│   │   ├── test_transcripts_whisper.json
│   │   └── train_paraphrases.json         # Voxtral paraphrase augmentation — list of
│   │                                      #   {"key", "emotion", "sentiment", "original", "text"}
│   │
│   ├── meld_embeddings/                   # Voxtral encoder → 1280-d acoustic vectors
│   │   ├── train_embeddings.pt            # dict {"dia0_utt0": Tensor(1280,), ...}
│   │   ├── dev_embeddings.pt
│   │   └── test_embeddings.pt
│   │
│   ├── meld_text_embeddings/              # XLM-R [CLS] over gold text (768-d, cached)
│   │   ├── train_text_embeddings.pt       # ★ overwritten by extract_text_embeddings.py
│   │   │                                  #    every time we run a new Phase 1 ablation
│   │   ├── dev_text_embeddings.pt
│   │   └── test_text_embeddings.pt
│   │
│   ├── meld_text_embeddings_asr/          # XLM-R [CLS] over ASR transcripts (A3 ablation)
│   │   ├── train_text_embeddings.pt
│   │   ├── dev_text_embeddings.pt
│   │   └── test_text_embeddings.pt
│   │
│   ├── ravdess/                           # RAVDESS Audio_Speech_Actors_01-24.zip extracted
│   │   ├── Actor_01/  ...  Actor_24/      # 60 .wav each (2 sentences × 8 emo × 2 intensity × 2 rep)
│   │   │                                  #    Filename: 03-01-06-01-02-01-12.wav
│   │   │                                  #    field 3 = emotion (01 neutral..08 surprised)
│   │   │                                  #    field 5 = statement (01=kids by the door,
│   │   │                                  #                          02=dogs by the door)
│   │
│   └── ravdess_embeddings/                # Voxtral encoder over all 1440 RAVDESS clips
│       └── embeddings.pt                  # {"03-01-06-01-02-01-12": Tensor(1280,), ...}
│
└── checkpoints/
    ├── mini/                              # Current Phase 1 + Phase 2 checkpoints (mini config)
    │   ├── best_model.pt                  # ★ Current XLM-R checkpoint (whichever run was last)
    │   ├── checkpoint_epoch009_*.pt       # Latest-epoch checkpoint (older ones auto-deleted)
    │   └── fusion/                        # Phase 2 fusion checkpoints per architecture
    │       ├── concat/best_model.pt
    │       ├── sum/best_model.pt
    │       ├── gated/best_model.pt
    │       └── crossmodal/best_model.pt
    │
    ├── mini_baseline/                     # Archived original baseline Phase 1
    │   └── checkpoint_epoch009_baseline.pt
    ├── mini_focal_sampler/                # Archived focal+sampler Phase 1
    │   ├── best_model.pt
    │   └── checkpoint_epoch009_f10.5410.pt
    ├── mini_aug/                          # Archived Voxtral-aug Phase 1
    │   └── best_model.pt
    ├── mini_asr_transcripts/              # Archived ASR-transcript-trained Phase 1 (A3)
    │
    ├── ravdess/                           # RAVDESS Phase A acoustic backbone
    │   └── best_acoustic_backbone.pt      # Contains full state_dict + backbone-only state_dict
    │                                      #   (backbone_state_dict is what fusion warm-starts from)
    │
    └── combined/                          # Combined RAVDESS+MELD Phase A (7-class MELD head)
        ├── best_acoustic_backbone.pt      # Backbone-only for fusion transfer
        └── best_full_model.pt             # Full model — used for the acoustic-only ablation row
```

### How large files are managed

1. **Nothing under `/dcs/large/u5734759/` is git-tracked.** Only source code lives in the git repo.
2. **All paths are read from `mini.yaml`** — never hardcoded. So switching to `small.yaml` (or a fresh scratch dir) works without code changes.
3. **Every heavy artefact is regenerable** by re-running its preprocessing sbatch. Rough regeneration times:
   - MELD transcription (Pass 1) — ~2 h on 1 GPU
   - MELD acoustic embeddings (Pass 2) — ~1 h on 1 GPU
   - Whisper transcription — ~1 h on 1 GPU
   - RAVDESS acoustic embeddings — ~15 min on 1 GPU
   - Voxtral paraphrases (~2839 lines) — ~1 h on 1 GPU
4. **Checkpoints are per-ablation, archived by hand.** After each Phase 1 recipe run, the resulting `checkpoints/mini/best_model.pt` is renamed into `checkpoints/mini_<recipe>/` so it's not overwritten by the next run.
5. **`meld_text_embeddings/` is *not* per-ablation — it's a rolling cache.** `extract_text_embeddings.py` overwrites it every time. `evaluate_ablation.sbatch` handles this transparently by re-extracting text embeddings before re-training fusion, so each ablation run uses the correct text embeddings for its Phase 1 checkpoint.
6. **HF model cache is shared across all runs** — Voxtral-Mini (~7 GB) and Whisper Large-v3 (~3 GB) are only downloaded once.
7. **Slurm jobs export `HF_HOME` and `TRANSFORMERS_CACHE` to `/dcs/large/u5734759/hf_cache`** at the top of every sbatch script.

---

## End-to-end pipeline (how to reproduce)

1. **Preprocessing** (one-off, ~4 h total across parallel jobs)
   ```bash
   sbatch src/scripts/transcribe.sbatch                    # Voxtral ASR + acoustic embeddings for MELD
   sbatch src/scripts/transcribe_whisper.sbatch            # Whisper baseline for WER comparison
   sbatch src/scripts/compute_wer.sbatch                   # WER analysis
   sbatch src/scripts/extract_ravdess_embeddings.sbatch    # RAVDESS acoustic embeddings
   ```

2. **Phase 1** — XLM-R text-only fine-tune (~30 min per recipe)
   ```bash
   # Edit src/configs/mini.yaml:
   #   training.use_focal_loss / use_weighted_sampler / use_paraphrases
   # then:
   sbatch src/scripts/augment.sbatch     # (only if use_paraphrases=true)
   sbatch src/scripts/finetune.sbatch
   ```

3. **Phase A** — RAVDESS acoustic backbone (optional, ~15-30 min)
   ```bash
   sbatch src/scripts/pretrain_ravdess.sbatch    # RAVDESS-only 8-class
   sbatch src/scripts/pretrain_combined.sbatch   # RAVDESS + MELD 7-class
   ```

4. **Full evaluation for one Phase 1 checkpoint** (~1 h — extracts embs, trains all 4 fusion types, evaluates everything)
   ```bash
   sbatch src/scripts/evaluate_ablation.sbatch <TAG> <PATH_TO_PHASE1_CKPT>
   # Example:
   sbatch src/scripts/evaluate_ablation.sbatch \
       _focal_sampler \
       /dcs/large/u5734759/checkpoints/mini_focal_sampler/best_model.pt
   ```

5. **Late fusion + zero-shot** (auxiliary)
   ```bash
   sbatch src/scripts/voxtral_zeroshot.sbatch    # Voxtral zero-shot ERC
   python src/evaluation/late_fusion.py --config src/configs/mini.yaml   # after prob caches exist
   ```

### Where to look for what

| Question | File |
|---|---|
| What ablations were run and what did they achieve? | `PROGRESS.md` |
| What are the coding rules I must follow? | `CLAUDE.md` |
| Where's the primary hyperparameter config? | `src/configs/mini.yaml` |
| How does focal loss / weighted sampler work in code? | `src/training/finetune.py` (see `FocalLoss` + `build_class_weighted_sampler`) |
| How does Voxtral paraphrase augmentation work? | `src/preprocessing/augment_transcripts.py` + `TranscriptDataset` in `finetune.py` |
| Where is the Voxtral encoder → acoustic embedding? | `src/models/voxtral.py` + `src/preprocessing/transcribe_all.py` Pass 2 |
| Where's the RAVDESS→MELD label map? | `src/training/pretrain_combined.py` (`RAVDESS_TO_MELD`) |
| Where is fusion warm-started from the RAVDESS backbone? | `src/training/train_fusion.py` (searches for `use_ravdess_pretrain`) |
| Where are the metrics computed? | `src/evaluation/metrics.py` |
| Where are cached embeddings on disk? | `/dcs/large/u5734759/data/meld_*` and `ravdess_embeddings/` |
| Where are Phase 1 checkpoints archived? | `/dcs/large/u5734759/checkpoints/mini_<recipe>/best_model.pt` |
