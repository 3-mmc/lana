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

Options for scripted runs: `--quick`, `--full`, `--mms` (accept the MMS licence without asking), `--keep` (do not offer to delete downloads), `--no-pause` (do not wait for Enter at the end).

## What is measured

Each model reads its first recording twice, so that loading from disk (`cold_seconds`) can be told apart from steady speed (`warm_seconds`). `x_realtime` is seconds of audio per second of computing on the warm run of the first recording, which is always one the model can read, so 5.0 means a ten-minute recording takes two minutes. A model given a language it does not know can loop on its own output and take far longer (Whisper small spent 42 seconds on 15 seconds of Georgian); such runs are kept in `runs` but do not set the headline speed. `peak_mb` is the model process's peak resident memory. `cer` is the character error rate against the reference transcript, after lowercasing and removing punctuation. Digits are kept, so a model that spells out 4892 is charged for it, in the same way on every computer.

The recordings are FLEURS sentence 1618, read in Georgian and German (CC BY 4.0, Conneau et al. 2022, shared with the demo in `web/demo/`), and a Dargwa sentence from Mozilla Common Voice 27.0 (CC0).

On macOS, Whisper runs on the GPU through Metal, and the result file records this as `whisper_backend`.
