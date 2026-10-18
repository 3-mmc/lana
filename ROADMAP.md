# Analysis roadmap

Model, accuracy and analysis work after 0.1.6, as agreed on 2026-10-18. Each item is
ticked when it is merged to main and has been checked on real audio. The reasoning
behind the list is in `.agent-memory/` (see `lana-analysis-roadmap`).

## Now

### Frame probabilities (the base for the next four)
- [ ] Run ZIPA, Omnilingual and MMS through `ort` against the onnxruntime that
      sherpa-onnx already links (no second runtime), keeping per-frame log-probabilities
- [ ] Kaldi fbank in Rust for ZIPA, checked against sherpa-onnx's output on the demo clips
- [ ] Same tiers as before on the demo clips (no regression in symbols or times)

### Accuracy fixes
- [x] **Confidence**: each phone and word carries the model's probability; low ones are
      coloured, and a menu item steps through them (Whisper too, from its token p)
- [x] **Alternatives**: an uncertain phone carries the model's runner-ups with their
      probabilities (n 0.55 · m 0.25 · ŋ 0.12), shown on the label and one click to accept;
      a word gets alternative spellings from its weakest characters
- [x] **Triage pass**: one list of everything worth a look — low-confidence labels,
      dropped Whisper loops, pieces with no output — stepped through with one key
- [ ] **Alphabet**: Omnilingual is held to the language's alphabet (from MMS vocabularies),
      so Dargwa stops coming out in Arabic, Latin or Georgian script
- [x] **Re-timing**: after labels are corrected, boundaries are refitted to the corrected
      labels by CTC forced alignment over the stretch
- [ ] **Loudness**: audio is normalised before the models only (never for display or
      measurement), so a quiet recording like FLEURS ka is not missed by pyannote
- [ ] **Whisper loops**: a piece whose text repeats (compression ratio / repeated n-grams)
      is dropped and reported, rather than shown as words

### New models
- [ ] **Silero VAD**: pieces cut at real pauses; silence skipped
- [ ] **Parakeet TDT 0.6B v3** (25 European languages, sherpa-onnx transducer)
- [ ] **Audio tagging** (sherpa-onnx AudioSet tagger): laughter, music, noise, coughs as a tier

### Romanisation tiers (asked 2026-10-18)
- [ ] Derived tiers like the ARPABET one, worked backwards from what the models write:
      Han → pinyin (per-character readings; characters with several readings marked, the
      alternatives one click away), pinyin → Wade-Giles, Zhuyin (bopomofo), Gwoyeu Romatzyh
      and IPA by syllable table; POJ ↔ Tâi-lô for Min Nan (MMS `nan` writes POJ/Tâi-lô)

### Analysis aids
- [x] **Search by example**: select a stretch, find similar ones (DTW over ZIPA posteriors)
- [x] **Voice quality**: H1–H2, CPP, HNR from Praat, with an optional creaky/breathy/modal
      tier by thresholds the user sets
- [ ] **Neural pitch** beside Praat's (PENN or RMVPE, licence to check), as a second f0 track

## Later (noted, not started)

- **Training bundle**: export corrected tiers + audio as a training package (MMS adapter,
  ZIPA fine-tune); train adapters inside Lana after that
- **ZIPA fine-tune for Georgian** with grapheme-to-IPA targets that keep ejectives
- **Spoken language ID** per window, routing code-switched stretches to the right model
  (MMS-LID, NC; or VoxLingua107 ECAPA, Apache) — ONNX exports to verify
- **Diarization**: pyannote community-1 (licence to verify) or 3D-Speaker ERes2Net
  embeddings; expected-speaker-count field; overlap tier from the segmentation output
- **Qwen3-ASR** (52 languages; timestamps need its forced aligner)
- **Ejective / glottalisation detector** on ZIPA encoder features
- **Evaluation**: DoReCo phone-aligned corpus as reference; boundary-timing metric
  (share within 20 ms) in lana-bench
