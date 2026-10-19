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

### Codeswitching and speakers (asked 2026-10-18)
- [x] **Per-recording toggles**: "contains multiple speakers" (with the number, if known) drives the
      speaker tier; "contains codeswitching" (with the languages, if known) adds a language tier
- [x] **Spoken language ID**: MMS-LID 1024 (NC, behind the MMS acceptance), 3 s windows every 1 s,
      each classified on its own, Viterbi with a switch cost; candidates restricted to the languages
      given. Measured on spliced FLEURS/Common Voice: ka→de switch within 0.38 s, 98.6% of time right;
      oss/dar told the pair 63–82%; 1.3–1.6× real time, 1.6 GB peak
- [x] **Routing**: a word run on a codeswitching recording is read stretch by stretch in each stretch's
      language (Whisper code, Omnilingual alphabet, MMS adapter when installed)

### Analysis aids
- [x] **Search by example**: select a stretch, find similar ones (DTW over ZIPA posteriors)
- [x] **Voice quality**: H1–H2, CPP, HNR from Praat, with an optional creaky/breathy/modal
      tier by thresholds the user sets
- [x] **Neural pitch** beside Praat's: CREPE (MIT; PENN and FCPE publish no ONNX), tiny and small,
      downloaded from onnxcrepe's GitHub release with pinned checksums

## Later (noted, not started)

- **Readings by use in speech**: the romanisation tiers rank a character's readings by how many
  CC-CEDICT words use each, not by how often each is said, so 的 (de) and 还 (hái) are marked more
  often than they need to be; a spoken-corpus frequency would settle most of them
- **Gwoyeu Romatzyh** and Jyutping (Cantonese: MMS `yue` writes Han)

- **Training bundle**: export corrected tiers + audio as a training package (MMS adapter,
  ZIPA fine-tune); train adapters inside Lana after that
- **ZIPA fine-tune for Georgian** with grapheme-to-IPA targets that keep ejectives
- **Spoken language ID, later steps**: VoxLingua107 ECAPA (Apache) as a non-NC alternative; language
  ID inside lana-bench; context across stretch boundaries when routing (each stretch is now read alone)
- **Diarization**: pyannote community-1 (licence to verify) or 3D-Speaker ERes2Net
  embeddings; overlap tier from the segmentation output
- **Qwen3-ASR** (52 languages; timestamps need its forced aligner)
- **Ejective / glottalisation detector** on ZIPA encoder features
- **Evaluation**: DoReCo phone-aligned corpus as reference; boundary-timing metric
  (share within 20 ms) in lana-bench
