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

### Romanisation, second pass (2026-10-18)
- [x] **Readings by use in speech**: a character no word settles takes the reading said most often
      in that position in AISHELL-3's pinyin transcripts (Apache 2.0), with CC-CEDICT words weighted
      by OpenSubtitles counts (FrequencyWords, CC BY-SA 4.0) for characters AISHELL-3 seldom has.
      Marked characters 308 → 263; 的 还 了 着 都 no longer marked (还 now hái, was huán). On the
      AISHELL-3 test half (held out): wrong readings 2.93% → 1.22%, marks 15,252 → 2,282 (right
      anyway 13,961 → 1,626). Also fixed: CC-CEDICT pinyin split wrongly without apostrophes
      (可能 kěn éng), and 28 words respelt / 11 dropped where speech says they are seldom that word
      (生活中的 is not zhòngdì)
- [x] **Gwoyeu Romatzyh** by rule, all 416 syllables × 5 tones checked against Wiktionary's converter
- [x] **Jyutping** (rime-cantonese, CC BY 4.0, weighted by HKCanCor, CC BY 4.0) with Yale and
      Cantonese IPA by rule (703 syllables × 6 tones checked); offered when the language is Cantonese,
      or beside Mandarin when the tier holds Cantonese-only characters. 69 characters marked; held-out
      HKCanCor half 2.45% wrong

## Later (noted, not started)

- **Romanisations still to add**, by what Lana's models write:
  - Hakka: Pha̍k-fa-sṳ (MMS `hak`) → Hakka Romanization System (Taiwan MOE), and Hakka from Han
  - Min Nan from Han (MMS `nan` sometimes writes Han): Taiwanese MOE dictionary readings → POJ/Tâi-lô;
    Teochew Peng'im
  - Mandarin: Tongyong pinyin, MPS II, Yale Mandarin, Cyrillic Palladius (Wiktionary has rules for all
    four, testable the same way as Gwoyeu Romatzyh); GR's neutral tone with its original tone
  - Cantonese: Sidney Lau, Cantonese Pinyin, Guangdong romanisation (rules in Wiktionary yue-pron);
    changed tones (變調) offered as alternatives
  - Wu (Shanghainese, MMS `wuu`): Wugniu
  - Japanese (Whisper, MMS `jpn` write kana/kanji): kana → Hepburn by rule; kanji needs a dictionary
    (UniDic/JMdict) and is the same reading problem as Han
  - Korean (Whisper, MMS `kor` write Hangul): Revised Romanization and Yale by rule, IPA by rule with
    the usual assimilations
  - Scripts Omnilingual and MMS write for languages linguists usually cite in Latin: Georgian
    (national system and ISO 9984), Cyrillic (ISO 9 / scholarly, per language: Russian, Ossetian,
    Dargwa, Avar with their palochka and digraphs), Armenian (Hübschmann-Meillet), Arabic and Persian
    (abjad, needs vowels the script omits: only consonantal transliteration is honest), Hebrew,
    Devanagari and other Brahmic scripts (ISO 15919 / IAST), Thai (RTGS), Greek (ELOT)

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
