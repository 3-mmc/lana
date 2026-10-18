# lana-bench

Times Lana's built-in speech models on one computer: how fast each runs, how much memory it takes, and how accurate it is on three short recordings with known transcripts. The results are for deciding which models Lana should recommend on which hardware.

It runs the app's own recognition code (`src-tauri/src/local`, included by path), so the timings are the ones Lana itself would get. Each model runs in a separate process, which keeps its peak memory separate from the others and records a model that runs out of memory as a failure instead of ending the test. Nothing is uploaded: the result is one JSON file, saved next to the program, that holds the timings and the computer's operating system, processor and memory.

## Getting it

The [build workflow](https://github.com/3-mmc/lana/actions/workflows/build.yml) produces `lana-bench-<runner>` artefacts for macOS, Windows and Linux on every run. Downloading them requires a GitHub account.

- **Windows.** Unzip and double-click `lana-bench.exe`. If SmartScreen intervenes, choose **More info → Run anyway**.
- **macOS.** Unzip, then in Terminal: `tar xzf lana-bench-macOS.tar.gz && xattr -d com.apple.quarantine lana-bench && ./lana-bench`. The `xattr` step removes the download flag that would otherwise make macOS refuse an unsigned program.
- **Linux.** `tar xzf lana-bench-Linux.tar.gz && ./lana-bench`

## Running it

Started without arguments, it asks which test to run:

- **Quick**: ZIPA small, Whisper base and Speakers. About 250 MB to download and a few minutes to run.
- **Full**: every model in Lana's catalogue, plus MMS for Dargwa and Georgian if its non-commercial licence is accepted. About 4 GB to download and 10 to 30 minutes to run.

Models that Lana has already downloaded are used where they are. Anything missing goes to the user's cache folder (`lana-bench-models`), and the program offers to delete it afterwards.

Options for scripted runs: `--quick`, `--full`, `--mms` (accept the MMS licence without asking), `--keep` (do not offer to delete downloads), `--no-pause` (do not wait for Enter at the end).  `--eval <folder>` measures boundary timing instead; see below.

## What is measured

Each model reads its first recording twice, so that loading from disk (`cold_seconds`) can be told apart from steady speed (`warm_seconds`). `x_realtime` is seconds of audio per second of computing on the warm run of the first recording, which is always one the model can read, so 5.0 means a ten-minute recording takes two minutes. A model given a language it does not know can loop on its own output and take far longer (Whisper small spent 42 seconds on 15 seconds of Georgian); such runs are kept in `runs` but do not set the headline speed. `peak_mb` is the model process's peak resident memory. `cer` is the character error rate against the reference transcript, after lowercasing and removing punctuation. Digits are kept, so a model that spells out 4892 is charged for it, in the same way on every computer.

The recordings are FLEURS sentence 1618, read in Georgian and German (CC BY 4.0, Conneau et al. 2022, shared with the demo in `web/demo/`), and a Dargwa sentence from Mozilla Common Voice 27.0 (CC0).

On macOS, Whisper runs on the GPU through Metal, and the result file records this as `whisper_backend`.

## Boundary timing (`--eval`)

`lana-bench --eval <folder>` measures where the phone and word models put their boundaries, against recordings whose phones have been aligned already. It reads every long-format TextGrid under the folder that has an audio file of the same name somewhere under it too, and expects DoReCo's tier names: `ref@`, `wd@` and `ph@` followed by the speaker. `--models zipa-small,omnilingual-300m` chooses the models (ZIPA small and large and Omnilingual 300M are the default; MMS works as `mms-<code>` once Lana has that language), and `--seconds 300` reads only the first five minutes of each recording. The result is saved next to the program as `lana-bench-eval-<system>-<time>.json`.

The reference is [DoReCo](https://doreco.huma-num.fr) 2.0 (Seifart, Paschen & Stave 2024), which has narrative recordings in 53 languages with phone-aligned transcriptions. Its word boundaries were checked and corrected by hand; the phone boundaries inside each word come from WebMAUS's language-independent aligner (Paschen et al. 2020). So the phone figures compare one aligner with another, and the word-onset figures are the ones with a person behind them. DoReCo's phones are written in X-SAMPA; `data/xsampa.tsv` turns them into IPA, and any symbol it does not list is reported rather than guessed, as is any IPA character a model has no symbol for.

Scoring happens in stretches: runs of ordinary words within one of DoReCo's annotation units. A labelled word (foreign material, a false start, unclear speech) has no phones and ends a stretch, and a stretch that overlaps another speaker's is left out. A boundary is where a phone (or, for Omnilingual and MMS, a word) begins.

- **Free reading.** The model reads the recording as Lana does, and its onsets are matched one to one with the reference's: precision, recall and F1 within 20 ms, the R-value (which, unlike F1, punishes extra boundaries), recall within 50 ms, and the median signed offset of each reference onset's nearest prediction. A mark such as ʼ, ʰ or ː that ZIPA writes as a symbol of its own is joined to the letter before it and is not a boundary. Affricates and diphthongs are one DoReCo phone but two letters for ZIPA, so their inner boundary counts against precision.
- **Forced alignment.** The reference phones (or words) of each stretch go to Lana's re-timing, the same code as Re-time in the app, and each placed onset except the stretch's first, which the request gives away, is compared with the reference: the share within 20 and 50 ms and the median error. For comparison, `even_spread_within_20ms` is what spreading the same onsets evenly over the stretch would score.
- **Phone error rate**, for ZIPA only: its free reading against the reference phones, stretch by stretch, counting a letter with its marks as one phone (`strict`), and once more by letters alone (`letters_only`).

DoReCo's annotations and audio are separate downloads for each language; the language's page links both, and audio files can be fetched one at a time. Licences differ by language and between annotation and audio: of the 53, 35 have CC BY annotations and 29 have CC BY audio as well; the rest are non-commercial (CC BY-NC, -NC-SA or -NC-ND), or keep their audio in an outside archive. The set measured below is CC BY 4.0 throughout, annotations and audio:

- Sanzhi Dargwa (Forker & Schiborr 2024): `mc_sanzhi_bazhuk`, `mc_sanzhi_tape`; ejectives and pharyngealised vowels
- Svan (Gippert 2024): `svv35_02`, `svv35_03`; ejectives and aspirated stops
- Nǁng (Güldemann, Ernszt, Siegmund & Witzlack-Makarevich 2024): `NC090930-01_A`, `NE080825-01_A`, `NE091006-01_A`; clicks
- Southern English (Schiborr 2024): `mc_english_kent02_b_part1`

With `--seconds 300` that is 17 minutes of scored speech and 11,098 reference phones. The full references for each dataset are in its `dataset-info.txt`, and DoReCo asks that each one used be cited.

Results on an i7-9700K under WSL (October 2026):

| Model | Free P / R / F1 (20 ms) | R-value | Forced within 20 ms | Forced median error | Phone error rate |
| --- | --- | --- | --- | --- | --- |
| ZIPA small | 0.57 / 0.62 / 0.60 | 0.64 | 56% (word onsets 52%) | 20 ms | 56% |
| ZIPA large | 0.55 / 0.47 / 0.51 | 0.59 | 47% (word onsets 46%) | 23 ms | 51% |
| Omnilingual 300M (words) | 0.19 / 0.13 / 0.15 | 0.34 | 21% | 44 ms | |
| Omnilingual 1B (words) | 0.19 / 0.12 / 0.15 | 0.34 | 25% | 40 ms | |

Spreading the phones evenly would put 13% of them within 20 ms. Within 50 ms, ZIPA small's re-timing places 86% of onsets and its free reading finds 85%. ZIPA small times boundaries better than ZIPA large, which writes about a fifth fewer phones than the reference has (9,645 against 11,662), though large makes fewer phone errors. Omnilingual's word onsets come about 40 ms late, both read freely and aligned, so a word model is the wrong tool for re-timing at this tolerance. Per language, ZIPA small's re-timing ranged from 52% within 20 ms (English) to 61% (Svan), and its phone error rate from 42% (Svan) to 79% (Nǁng, whose clicks it mostly does not write).
