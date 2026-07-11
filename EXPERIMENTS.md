# Experiments Log — Multimodal Emotion & Sentiment on MELD

**Project:** Multimodal fusion for emotion/sentiment, end goal = **robustness across unseen
UK / Australian English accents.**
**Backbones:** Voxtral-Mini-3B (= Whisper-large-v3 encoder + a 3B LLM), XLM-RoBERTa-base.
**Dataset (in-domain):** MELD (American English, *Friends*). 7 emotions, 3 sentiments.
**Primary metric:** weighted-F1 (WF1); macro-F1 reported too (severe imbalance: neutral ~47%,
disgust/fear ~3% each).
**Last updated:** 2026-06-18.

> All numbers are **MELD test** unless noted. The dialect (UK/AU) evaluation — the actual
> research target — is **not yet built** (see §7).

---

## 1. Results summary (MELD test)

| # | Approach | Emotion WF1 | Emotion macro | Sentiment WF1 | Sentiment macro |
|---|----------|:-----------:|:-------------:|:-------------:|:---------------:|
| 1 | Text only — XLM-R | 0.449 | 0.292 | 0.544 | 0.502 |
| 2 | Zero-shot Voxtral LLM (prompted) | 0.504 | 0.320 | 0.552 | 0.516 |
| 3 | Acoustic encoder fine-tune (+focal) | 0.547 | 0.357 | 0.615 | 0.587 |
| 4 | Late fusion of 1+2+3 (meta) | 0.534 | 0.368 | 0.633 | 0.609 |
| 5 | bc-LSTM context — **gold text** ⊕ acoustic | **0.624** | **0.440** | **0.704** | **0.679** |
| **5-ASR** | **bc-LSTM context — Voxtral ASR text ⊕ acoustic** | **0.492** | **0.324** | **0.575** | **0.539** |
| 5b | bc-LSTM bigger (512×2) — *ablation, worse* | 0.612 | 0.437 | 0.686 | 0.666 |
| — | *Oracle upper bound (1+2+3)* | *0.728* | *0.543* | *0.806* | *0.789* |

**Two headline numbers:**
- **#5 (gold text) = 0.624 / 0.704** — the *upper bound*, assuming perfect transcripts.
- **#5-ASR (Voxtral ASR text) = 0.492 / 0.575** — the *realistic* number, using actual ASR.
- The **~13-pt gold→ASR gap is itself a finding**: the text modality is **highly dependent on
  ASR quality**. **#5-ASR is the correct in-domain baseline for the dialect comparison** (UK/AU
  will also use ASR — baselining against gold would inflate the apparent dialect drop).

---

## 2. Each approach — result and why preferred / dropped

1. **Text-only XLM-R** (`finetune.py`): test 0.449. Weakest single modality; kept as the text leg.
2. **Zero-shot Voxtral LLM** (`voxtral_zeroshot.py`): no training, test 0.504. Baseline; **not**
   in the final model. *Potential value = dialect robustness (broadly-pretrained, never overfit
   US English) — to test in §7.*
3. **Acoustic encoder fine-tune** (`finetune_encoder.py`): test 0.547 (best utterance-level).
   Overfit after epoch 4; disgust stayed 0.00. An optional feature upgrade for #5 (§6).
4. **Late fusion** (`late_fusion.py`): **sentiment win** (meta 0.633 > 0.615); **emotion no gain**
   (0.534 < 0.547). Oracle 0.728/0.806 → complementary, but combiner can't reach it (one-hot
   zero-shot; disgust/fear have no signal in any model). Reusable component.
5. **★ bc-LSTM dialogue context** (`context_lstm.py`, `train_context.py`): group utterances by
   `Dialogue_ID`; per utterance = concat(text, acoustic); BiLSTM over the conversation → heads.
   - **gold text → 0.624 / 0.704 (best, upper bound).**
   - **Voxtral ASR text → 0.492 / 0.575 (realistic).**
   Tiny (4.7M), trains ~1 min. **Confirms context is the dominant lever.** Text source decides
   ~13 pts.

---

## 3. Ablations & findings (negative / informative)
- **Gold vs ASR text (the big one).** Swapping gold transcripts for Voxtral ASR drops the
  bc-LSTM **0.624 → 0.492** (emotion). ASR word-error ~42% / char-error ~34% on MELD test
  (§3a), *mostly cosmetic*
  (curly-quote artifacts, paraphrase) — but it **destroys short emotional utterances**
  (`"Push!"` → `"Oh shit"`), which carry the emotion. So the text modality's contribution is
  fragile to ASR quality. → Realistic baseline = 0.492; gold = ceiling.
- **Bigger context model hurt.** bc-LSTM 512×2 (16.8M) → 0.612, below 256×1 (0.624). Val loss
  never improved past epoch 2 → overfitting. Extra capacity (and attention, by extension) does
  not help on 1,038 dialogues.
- **Imbalance tricks underwhelmed.** Focal ≈ cross-entropy on text; augmentation gained on dev
  but not test. The thing that moved rare classes was **dialogue context**, not loss/sampling.

---

## 3a. Voxtral ASR quality (WER and related measures)

Voxtral ASR transcripts vs MELD gold (`Utterance` column), normalised lowercase +
punctuation-stripped before scoring (`src/evaluation/compute_wer.py`,
`results/mini/wer_analysis.json`). WIP = 1 − WIL.

| Split | n | WER ↓ | MER ↓ | WIL ↓ | WIP ↑ | CER ↓ |
|-------|---|:-----:|:-----:|:-----:|:-----:|:-----:|
| train | 9988 | 0.383 | 0.326 | 0.424 | 0.576 | 0.296 |
| dev   | 1108 | 0.340 | 0.295 | 0.394 | 0.606 | 0.250 |
| **test** | 2610 | **0.424** | 0.348 | 0.447 | 0.553 | 0.336 |

- **WER** (word error rate), **MER** (match error rate), **WIL** (word information lost),
  **WIP** (word information preserved), **CER** (character error rate). All corpus-level
  (micro-averaged over words/chars), not per-utterance means.
- **Test WER ≈ 0.42** is the headline ASR-quality number; **CER ≈ 0.34** is lower, confirming
  many errors are sub-word (curly-quote/paraphrase artifacts) rather than whole-word misses.
- This quantifies the noise behind the **gold→ASR fusion gap** in §3: the text leg degrades
  because it consumes transcripts at this error rate.

---

## 4. Considered and rejected (with reasons)
- **Qwen2-Audio (7B):** redundant with Voxtral, heavier, unverified on UK/AU. Rejected.
- **LoRA the 3B LLM end-to-end:** backprop through 3B, breaks cached pipeline, weeks. Deferred.
- **Plug fine-tuned encoder into the frozen LLM:** distribution mismatch — degrades, not helps.
- **MLP fusion modules (concat/sum/gated/cross):** original flat ~0.55; superseded by context.
- **Inject Voxtral emotion+confidence into the ASR text:** redundant with acoustic fusion +
  risks the model collapsing to "copy Voxtral's guess" (capped ~0.50). Worth only as an ablation.

---

## 5. SOTA context (honest positioning)
- MELD **7-class emotion WF1 SOTA is mid-to-high 60s** (MMGCN ~66, MM-DFN ~67, M3Net ~67–69),
  all context/graph models. Figures like 93/96% are **not** standard 7-class emotion WF1.
- **bc-LSTM gold 0.624 sits in the lower published band**; ASR 0.492 is the realistic figure.
- **Architecture is not novel** — bc-LSTM is a named MELD baseline (Poria et al.). **The
  contribution is the dialect-robustness study (§6), not the architecture.**

---

## 6. Final pipeline (realistic — Voxtral, ASR text)

```
                ┌─ transcribe (ASR) ──► XLM-R ──► text_emb (768) ─┐ per utterance ⊕
  audio ─► VOXTRAL ┤                                               ┤
                └─ encoder (mean-pool) ─────────► acou_emb (1280) ─┘
                                                                   ▼
                                       BiLSTM (256×1) over the dialogue ─► emotion + sentiment heads
                                       (focal loss · early stopping)
```
- **Voxtral does both jobs**: ASR (→ text for XLM-R) and the acoustic encoder. The **3B LLM is
  used only for ASR in preprocessing**, never in the bc-LSTM (which runs on cached features).
- **Realistic result: 0.492 / 0.575** (ASR text). **Gold-text 0.624 / 0.704 = upper bound.**

### Deployment simplification (optional)
- **Whisper-large-v3 can replace Voxtral** for both ASR and the acoustic encoder, dropping the
  3B LLM → lighter/real-time. (Voxtral's encoder *is* Whisper-large-v3.)

### Optional in-domain upgrade (not required)
- Feed **fine-tuned-encoder** acoustic features (#3) into the *same 256×1* bc-LSTM. (Do **not**
  scale the BiLSTM — that hurt.)

### Robustness study (the thesis)
- Compare the in-domain→UK/AU drop, **baselining against the ASR pipeline (0.492), not gold.**
- **Voxtral-zero-shot robustness hypothesis:** a broadly-pretrained audio-LLM (never trained on
  MELD) may transfer to UK/AU *better* than the US-fine-tuned supervised model — if the ranking
  flips out-of-domain, that's the headline finding.
- Note: accent hits **both** modalities — acoustic directly, **and text via ASR errors** (ASR is
  typically worse on non-US accents). Report dialect numbers with **ASR transcripts**; optionally
  also with gold UK/AU transcripts to separate the acoustic-accent effect from the ASR effect.

---

## 7. Open items / next steps (priority order)
1. **Build `dialect_eval.py` + obtain UK/AU emotion-speech data.** Empty stub; no UK/AU data on
   disk. **The actual research evaluation — highest priority.**
2. Run the full pipeline (and Voxtral-zero-shot) on UK/AU; report drop vs the **ASR** baseline.
3. Optional in-domain: fine-tuned-encoder features into the 256×1 bc-LSTM.
4. Optional: standalone-Whisper refactor; speaker info → DialogueRNN; `--balanced` + RAVDESS.
5. **Prompt tuning of the Voxtral zero-shot classifier** (later): better instructions / few-shot /
   constrained-label output / dialect-aware prompts to lift the zero-shot baseline (0.504). Most
   relevant as the audio-LLM *robustness* contender on UK/AU, not as an in-domain SOTA push.
6. **Voxtral-generated commonsense (à la COSMIC, but audio-grounded)** (later): prompt Voxtral
   per utterance for commonsense (intent/cause/reaction — NOT the emotion label, to avoid the
   circular/leaky shortcut), cache, encode with XLM-R, add as a 3rd stream
   `[text ⊕ acoustic ⊕ commonsense]` into the bc-LSTM. COSMIC's commonsense gave ~+2-3 on MELD
   via text-only COMET; Voxtral can ground it in *prosody* (novel). Could push gold 0.624 toward
   ~0.65, and "does LLM commonsense transfer to UK/AU?" is a second robustness question.

---

## 8. File map
| Component | File |
|---|---|
| Zero-shot LLM classification | `src/evaluation/voxtral_zeroshot.py` |
| Acoustic encoder fine-tune | `src/training/finetune_encoder.py`, `src/models/voxtral_encoder_classifier.py` |
| Text embeddings — gold / ASR | `src/preprocessing/extract_text_embeddings.py` / `extract_text_embeddings_asr.py` |
| Probability extractors (fusion) | `src/evaluation/extract_text_probs.py`, `extract_acoustic_probs.py`, `voxtral_probs.py` (WIP) |
| Late fusion / weighted scoring | `src/evaluation/late_fusion.py` |
| **bc-LSTM dialogue context (best)** | `src/models/context_lstm.py`, `src/training/train_context.py` |
| Metrics (per-class, absent-class safe) | `src/evaluation/metrics.py` |
| Configs | `src/configs/mini.yaml` (gold text), `mini_asr.yaml` (ASR text) |
| Slurm | `src/scripts/{voxtral_zeroshot,finetune_encoder,fusion_prep,train_context,context_asr}.sbatch` |
