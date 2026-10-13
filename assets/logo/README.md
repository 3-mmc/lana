# The Lana icon

The icon is drawn from data, not by hand: the waveform and the F1/F2/F3 tracks are Praat's
analysis of the word *Lana* [ˈlɑːnə], laid out like Praat's own sound editor (waveform above,
formant track below), in Lana's formant colours. F1 rises and F2 falls into the long [ɑː], and
both move back towards the centre for the final [ə].

The recording is synthesised (`espeak-ng -v en-gb -s 110 -p 45 -w lana-en-gb.wav Lana`), so
the file can be regenerated exactly. To rebuild the icon:

```sh
node analyse-lana.mjs ../../vendor lana-en-gb.wav lana-en-gb.json   # praat-wasm: Pitch, Formant (burg), Intensity
python3 draw-logo.py lana-en-gb.json lana-icon.svg final
rsvg-convert -w 1024 lana-icon.svg -o ../../src-tauri/icons/source.png
cd ../../src-tauri && cargo tauri icon icons/source.png
```
