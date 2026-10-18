# Analysis roadmap

Model, accuracy and analysis work after 0.1.6, as agreed on 2026-10-18. Each item is
ticked when it is merged to main and has been checked on real audio. The reasoning
behind the list is in `.agent-memory/` (see `lana-analysis-roadmap`).

## Now

### Frame probabilities (the base for the next four)
- [x] Run ZIPA, Omnilingual and MMS through `ort` against the onnxruntime that
      sherpa-onnx already links (no second runtime), keeping per-frame log-probabilities
- [x] Kaldi fbank in Rust for ZIPA, checked against sherpa-onnx's output on the demo clips
- [x] Same tiers as before on the demo clips (no regression in symbols or times)

### Accuracy fixes
- [x] **Confidence**: each phone and word carries the model's probability; low ones are
      coloured, and a menu item steps through them (Whisper too, from its token p)
- [x] **Alternatives**: an uncertain phone carries the model's runner-ups with their
      probabilities (n 0.55 · m 0.25 · ŋ 0.12), shown on the label and one click to accept;
      a word gets alternative spellings from its weakest characters
- [x] **Triage pass**: one list of everything worth a look — low-confidence labels,
      dropped Whisper loops, pieces with no output — stepped through with one key
- [x] **Alphabet**: Omnilingual is held to the language's alphabet (from MMS vocabularies),
      so Dargwa stops coming out in Arabic, Latin or Georgian script
- [x] **Re-timing**: after labels are corrected, boundaries are refitted to the corrected
      labels by CTC forced alignment over the stretch
- [x] ~~**Loudness**~~ tested 2026-10-18 and dropped: raising FLEURS ka (−50 dBFS) by 40 dB
      before the models changed nothing (ZIPA small 43 vs 42 phones below 0.6, large 28 vs 29;
      the speaker model found the same 13.6 s of speech either way)
- [x] **Whisper loops**: a piece whose text repeats (compression ratio / repeated n-grams)
      is dropped and reported, rather than shown as words

### New models
- [x] **Silero VAD**: pieces cut at real pauses; silence skipped
- [x] **Parakeet TDT 0.6B v3** (25 European languages, sherpa-onnx transducer)
- [x] **Audio tagging** (sherpa-onnx AudioSet tagger): laughter, music, noise, coughs as a tier

### Romanisation tiers (asked 2026-10-18)
- [x] Derived tiers like the ARPABET one, worked backwards from what the models write:
      Han → pinyin (per-character readings; characters with several readings marked, the
      alternatives one click away), pinyin → Wade-Giles, Zhuyin (bopomofo), Gwoyeu Romatzyh
      and IPA by syllable table; POJ ↔ Tâi-lô for Min Nan (MMS `nan` writes POJ/Tâi-lô)

### Analysis aids
- [x] **Search by example**: select a stretch, find similar ones (DTW over ZIPA posteriors)
- [x] **Voice quality**: H1–H2, CPP, HNR from Praat, with an optional creaky/breathy/modal
      tier by thresholds the user sets
- [x] **Neural pitch** beside Praat's: CREPE (MIT; PENN and FCPE publish no ONNX), tiny and small,
      downloaded from onnxcrepe's GitHub release with pinned checksums

### Evaluation
- [x] **DoReCo as reference**, boundary timing in lana-bench (`--eval`): 17 min of CC BY speech
      in Sanzhi Dargwa, Svan, Nǁng and Southern English. Re-timing with ZIPA small puts 56% of
      phone onsets within 20 ms of DoReCo's (86% within 50 ms; spread evenly: 13%), ZIPA large
      47%; free reading F1 at 20 ms 0.60 (small) / 0.51 (large); phone error rate 56% / 51%.
      Omnilingual's word onsets run about 40 ms late (21–25% within 20 ms)

## Later (noted, not started)

- **Readings by use in speech**: the romanisation tiers rank a character's readings by how many
  CC-CEDICT words use each, not by how often each is said, so 的 (de) and 还 (hái) are marked more
  often than they need to be; a spoken-corpus frequency would settle most of them
- **Gwoyeu Romatzyh** and Jyutping (Cantonese: MMS `yue` writes Han)

- **Training bundle**: export corrected tiers + audio as a training package (MMS adapter,
  ZIPA fine-tune); train adapters inside Lana after that
- **ZIPA fine-tune for Georgian** with grapheme-to-IPA targets that keep ejectives
- **Spoken language ID** per window, routing code-switched stretches to the right model
  (MMS-LID, NC; or VoxLingua107 ECAPA, Apache) — ONNX exports to verify
- **Diarization**: pyannote community-1 (licence to verify) or 3D-Speaker ERes2Net
  embeddings; expected-speaker-count field; overlap tier from the segmentation output
- **Qwen3-ASR** (52 languages; timestamps need its forced aligner)
- **Ejective / glottalisation detector** on ZIPA encoder features
