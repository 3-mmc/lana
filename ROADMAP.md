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
- [x] **Names for every language ID code**: the 500 MMS-LID codes the language list lacked come from
      SIL's ISO 639-3 table (and its retirements table for mrd, Western Magar, merged into mgp in 2026),
      so they show by name and can be typed by name; 1,218 → 1,718 languages, existing rows unchanged

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

### Evaluation
- [x] **DoReCo as reference**, boundary timing in lana-bench (`--eval`): 17 min of CC BY speech
      in Sanzhi Dargwa, Svan, Nǁng and Southern English. Re-timing with ZIPA small puts 56% of
      phone onsets within 20 ms of DoReCo's (86% within 50 ms; spread evenly: 13%), ZIPA large
      47%; free reading F1 at 20 ms 0.60 (small) / 0.51 (large); phone error rate 56% / 51%.
      Omnilingual's word onsets ran 40–47 ms late in every language (21–25% within 20 ms); the lag is
      the model's, so Lana now takes 40 ms off Omnilingual and MMS times: Omnilingual 300M re-timing
      21% → 50% within 20 ms, free F1 0.15 → 0.34; MMS eng 17% → 45%

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
- **Spoken language ID, later steps**: language ID inside lana-bench; context across stretch
  boundaries when routing (each stretch is now read alone, and MMS reloads per stretch)
- **VoxLingua107 ECAPA** (Apache, 85 MB, 107 languages; no official ONNX, export it ourselves):
  sliding 2–3 s windows to find where the language changes, with MMS-LID or Whisper's
  `lang_detect` deciding each stretch between change points
- **Diarization, community-1**: pyannote community-1 (CC BY 4.0) = the segmentation-3.0 Lana
  ships + WeSpeaker ResNet34 + VBx/PLDA clustering, which sherpa-onnx lacks. Try the `speakrs`
  crate (Apache, full pipeline in Rust) before porting VBx ourselves; overlap tier from the
  segmentation's powerset classes; 3D-Speaker ERes2Net as an embedding to bench
- **Qwen3-ASR** 0.6B/1.7B (Apache, 30 languages + 22 Chinese dialects; already in sherpa-onnx
  1.13.8, no timestamps). Qwen3-ForcedAligner-0.6B gives 80 ms timestamps for 11 languages only,
  so word timing needs our CTC re-timing or the aligner
- **Ejective / glottalisation detector**, no training to start: ZIPA's per-frame ʼ and ʔ mass plus
  acoustic cues (burst, glottal lag, creak at vowel onset); later a small probe on ZIPA log-probs
  with Georgian orthography as weak labels, validated on DoReCo Svan / Sanzhi / Tabasaran
