"""Record the answers the demo replays.

Runs every demo model on every demo clip through a running serve.py and saves
the broker's responses in results/, plus the diarizer's re-clustering to 1-3
speakers. Usage: python3 demo/record.py [http://127.0.0.1:8710] [model ...]
"""
import json, sys, time, uuid, urllib.request
from pathlib import Path
DEMO = Path(__file__).resolve().parent
BASE = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else "http://127.0.0.1:8710") + "/api/transcribe"
# (model, language, whisper size). A size goes into the file name, so the demo
# can answer for whichever size the visitor picks; the first size listed for a
# model is also saved without one, as the fallback.
RUNS = {
    'ka-vinson': [('owsm', 'kat', None), ('ipa', 'ka', 'large-v3'), ('ipa', 'ka', 'medium'), ('mms', 'kat', None),
                  ('whisper', 'ka', 'large-v3'), ('whisper', 'ka', 'medium'), ('diarize', '', None)],
    'de-vinson': [('owsm', 'deu', None), ('ipa', 'de', 'medium'), ('mms', 'deu', None),
                  ('whisper', 'de', 'medium'), ('whisper', 'de', 'large-v3'), ('diarize', '', None)],
}
only = [a for a in sys.argv[1:] if not a.startswith("http")]
for clip, runs in RUNS.items():
    blob = (DEMO / f'{clip}.wav').read_bytes()
    seen = set()
    for model, lang, size in runs:
        if only and model not in only: continue
        fields = {'provider': 'local', 'model': model, 'response_format': 'verbose_json'}
        if lang: fields['language'] = lang
        if size: fields['asr_model'] = size
        b = uuid.uuid4().hex
        body = b''.join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
        body += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{clip}.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + blob + f'\r\n--{b}--\r\n'.encode()
        req = urllib.request.Request(BASE, data=body, headers={'Content-Type': f'multipart/form-data; boundary={b}'})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r: out = json.load(r)
        except urllib.error.HTTPError as e:
            print(clip, model, 'HTTP', e.code, e.read()[:300]); continue
        secs = round(time.time() - t, 2)
        out['_recorded'] = {'seconds': secs, 'language_sent': lang, 'asr_model': fields.get('asr_model'),
                            'date': time.strftime('%Y-%m-%d')}
        body = json.dumps(out, ensure_ascii=False, indent=1) + '\n'
        if size:
            (DEMO / 'results' / f'{clip}.{model}.{size}.json').write_text(body)
        if model not in seen:
            (DEMO / 'results' / f'{clip}.{model}.json').write_text(body)
        seen.add(model)
        print(clip, model, size or '', secs, 's', len(out.get('cues', [])), 'cues |', (out.get('text') or '')[:90])
        if model == 'diarize' and out.get('hash'):
            for n in (1, 2, 3):
                rq = urllib.request.Request(BASE.replace('/transcribe', '/recluster'), headers={'Content-Type': 'application/json'},
                    data=json.dumps({'provider': 'local', 'model': 'diarize', 'hash': out['hash'], 'speakers': n}).encode())
                with urllib.request.urlopen(rq, timeout=120) as r:
                    (DEMO / 'results' / f'{clip}.diarize.n{n}.json').write_text(json.dumps(json.load(r), ensure_ascii=False, indent=1) + '\n')
