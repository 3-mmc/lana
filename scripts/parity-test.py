#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Send the same requests through serve.py and the Rust broker; demand the same answers.

    scripts/parity-test.py [path/to/lana-desktop binary]

Starts a mock recogniser, then web/serve.py and `lana-desktop` (LANA_SERVE_ONLY,
no window) on one shared providers.json pointing at it. Every case is sent to
both brokers and the status + JSON compared. `wall_seconds` is dropped, and for
a provider that cannot be reached only the status is compared, since the two
HTTP stacks word a refused connection differently.

The mock echoes what it received (form fields, Authorization) in the response, so
a field the broker drops or a key it fails to send shows up as a mismatch.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Differences that are deliberate. Anything else that differs is a bug in one or the other.
KNOWN = {
    # serde_json writes {"a":1}, Python's json.dumps writes {"a": 1}; this is the
    # upstream's error object flattened into a message nobody parses.
    'upstream error object, no message',
    # serve.py's message tells you to edit serve.py, which the app has no copy of.
    'unknown kind',
}

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / '.cache/lana-desktop-target/debug/lana-desktop'


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ mock

class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.endswith('/progress'):
            return self.send(200, {'active': True, 'stage': 'words', 'path': self.path})
        if '/partial' in self.path:
            return self.send(200, {'cues': [], 'path': self.path})
        if self.path.endswith('/broken/progress'):
            return self.send(500, b'boom', 'text/plain')
        self.send(404, {'error': 'mock: no such GET ' + self.path})

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n)
        if self.path.endswith(('/cancel', '/recluster')):
            return self.send(200, {'ok': True, 'path': self.path,
                                   'got': json.loads(body) if body else None})
        if not self.path.endswith('/v1/audio/transcriptions'):
            return self.send(404, {'error': 'mock: no such POST ' + self.path})
        fields = parse_fields(self.headers['Content-Type'], body)
        auth = self.headers.get('Authorization')
        echo = json.dumps({'fields': fields, 'auth': auth}, sort_keys=True)
        model = fields.get('model', '')
        if model == 'seg-words':
            return self.send(200, {
                'text': echo, 'language': 'de', 'duration': 9.5,
                'segments': [{'start': 0, 'end': 3, 'text': ' eins zwei'},
                             {'start': 4, 'end': 6, 'text': ' drei'},
                             {'start': 7, 'end': 9.5, 'text': '  leer  '}],
                'words': [{'word': 'eins', 'start': 0.1, 'end': 0.9},
                          {'word': 'zwei', 'start': '1.2', 'end': 2.0},
                          {'word': 'gap', 'start': 3.2, 'end': 3.4},     # between segs
                          {'word': 'drei', 'start': 4.5, 'end': 5.0},
                          {'word': 'notime', 'start': None, 'end': 1}]})
        if model == 'words-only':
            return self.send(200, {'text': echo, 'words': [
                {'text': 'a', 'start': 1, 'end': 2}, {'word': 'b', 'start': 2, 'end': 3}]})
        if model == 'cues':
            return self.send(200, {'text': echo, 'cues': [{'start': 0, 'end': 1, 'words': []}]})
        if model == 'plain':
            return self.send(200, echo.encode(), 'text/plain')
        if model == 'not-object':
            return self.send(200, [1, 2, 3])
        if model == 'err-obj':
            return self.send(400, {'error': {'message': 'bad model', 'code': 'x'}})
        if model == 'err-obj-nomsg':
            return self.send(422, {'error': {'code': 'x'}})
        if model == 'err-str':
            return self.send(429, {'error': 'busy: ipa is running'})
        if model == 'err-text':
            return self.send(500, b'Internal meltdown', 'text/plain')
        self.send(200, {'text': echo, 'segments': []})


def parse_fields(ctype, body):
    boundary = ctype.split('boundary=')[1].encode()
    out = {}
    for part in body.split(b'--' + boundary):
        head, _, val = part.partition(b'\r\n\r\n')
        if b'name="' not in head:
            continue
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        val = val[:-2] if val.endswith(b'\r\n') else val
        if name == 'file':
            fn = head.split(b'filename="')[1].split(b'"')[0].decode()
            ct = head.split(b'Content-Type: ')[1].split(b'\r\n')[0].decode() \
                if b'Content-Type: ' in head else None
            out['file'] = f'{fn} {ct} {len(val)}B'
        else:
            out[name] = val.decode()
    return out


# ---------------------------------------------------------------- client

def call(base, method, path, form=None, body=None, file=('a.wav', 'audio/wav', b'RIFF....')):
    data, headers = None, {}
    if form is not None:
        b = uuid.uuid4().hex
        chunks = []
        for k, v in form.items():
            chunks.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        if file:
            fn, ct, blob = file
            chunks.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{fn}"\r\n'
                          f'Content-Type: {ct}\r\n\r\n'.encode() + blob + b'\r\n')
        chunks.append(f'--{b}--\r\n'.encode())
        data = b''.join(chunks)
        headers['Content-Type'] = f'multipart/form-data; boundary={b}'
    elif body is not None:
        data = body
        headers['Content-Type'] = 'application/json'
    elif method == 'POST':
        data = b''
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        v = {'_raw': raw.decode('utf-8', 'replace')[:200]}
    if isinstance(v, dict):
        v.pop('wall_seconds', None)
        v.pop('configured', None)   # the file path; identical by construction
        if v.get('service') in ('lana-web', 'lana-desktop'):
            v['service'] = '*'
    return code, v


def main():
    mock_port, py_port, rs_port, dead_port = (free_port() for _ in range(4))
    mock = ThreadingHTTPServer(('127.0.0.1', mock_port), Mock)
    threading.Thread(target=mock.serve_forever, daemon=True).start()
    m = f'http://127.0.0.1:{mock_port}'

    providers = {'providers': [
        {'id': 'mock', 'label': 'Mock', 'base': m, 'control': '/upstream/asr',
         'models': [{'id': 'seg-words', 'role': 'words'},
                    {'id': 'diar', 'role': 'speakers', 'control': '/upstream/diarize'}]},
        {'id': 'keyed', 'base': m + '/', 'api_key_env': 'LANA_PARITY_KEY',
         'headers': {'X-Extra': 'yes'}},
        {'id': 'keyless', 'base': m, 'api_key_env': 'LANA_PARITY_MISSING'},
        {'id': 'inline', 'base': m, 'api_key': 'sk-inline', 'pass_extras': False,
         'fields': {'timestamp_granularities[]': 'word', 'response_format': 'json'}},
        {'id': 'dead', 'base': f'http://127.0.0.1:{dead_port}', 'control': '/c'},
        {'id': 'broken', 'base': m, 'control': '/broken'},
        {'id': 'other-kind', 'kind': 'soap', 'base': m},
        {'label': 'no id, dropped'},
    ]}
    tmp = Path(tempfile.mkdtemp())
    pfile = tmp / 'providers.json'
    pfile.write_text(json.dumps(providers))
    env = dict(os.environ, PRAAT_WEB_PROVIDERS=str(pfile), LANA_PROVIDERS=str(pfile),
               PRAAT_WASM_DIR=str(ROOT / 'vendor'), LANA_PARITY_KEY='sk-env',
               LANA_SERVE_ONLY=str(rs_port))
    env.pop('LANA_PARITY_MISSING', None)
    procs = [
        subprocess.Popen([sys.executable, str(ROOT / 'web/serve.py'), '--port', str(py_port)],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([str(BIN)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    try:
        for port in (py_port, rs_port):
            for _ in range(100):
                try:
                    socket.create_connection(('127.0.0.1', port), 0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                sys.exit(f'nothing listening on {port}')

        T = '/api/transcribe'
        cases = [
            ('health', 'GET', '/api/health', {}),
            ('providers', 'GET', '/api/providers', {}),
            ('segments+words, extras, lang', 'POST', T, {'form': {
                'provider': 'mock', 'model': 'seg-words', 'language': 'de',
                'asr_model': 'medium', 'speakers': '2', 'unlisted': 'dropped', 'translate_to': ''}}),
            ('words only', 'POST', T, {'form': {'provider': 'mock', 'model': 'words-only'}}),
            ('cues pass through', 'POST', T, {'form': {'provider': 'mock', 'model': 'cues'}}),
            ('non-JSON reply', 'POST', T, {'form': {'provider': 'mock', 'model': 'plain'}}),
            ('JSON but not an object', 'POST', T, {'form': {'provider': 'mock', 'model': 'not-object'}}),
            ('no model field', 'POST', T, {'form': {'provider': 'mock'}}),
            ('response_format given', 'POST', T, {'form': {'provider': 'mock', 'response_format': 'json'}}),
            ('upstream error object', 'POST', T, {'form': {'provider': 'mock', 'model': 'err-obj'}}),
            ('upstream error object, no message', 'POST', T, {'form': {'provider': 'mock', 'model': 'err-obj-nomsg'}}),
            ('upstream error string', 'POST', T, {'form': {'provider': 'mock', 'model': 'err-str'}}),
            ('upstream error text', 'POST', T, {'form': {'provider': 'mock', 'model': 'err-text'}}),
            ('key from env + headers', 'POST', T, {'form': {'provider': 'keyed', 'model': 'x'}}),
            ('key missing', 'POST', T, {'form': {'provider': 'keyless', 'model': 'x'}}),
            ('inline key, fixed fields, no extras', 'POST', T, {'form': {
                'provider': 'inline', 'model': 'x', 'asr_model': 'big'}}),
            ('unknown kind', 'POST', T, {'form': {'provider': 'other-kind'}}),
            ('unknown provider', 'POST', T, {'form': {'provider': 'nope'}}),
            ('no file', 'POST', T, {'form': {'provider': 'mock'}, 'file': None}),
            ('empty file', 'POST', T, {'form': {'provider': 'mock'}, 'file': ('e.wav', 'audio/wav', b'')}),
            ('file type guessed', 'POST', T, {'form': {'provider': 'mock'},
                                              'file': ('x.mp3', '', b'ID3')}),
            ('unicode filename', 'POST', T, {'form': {'provider': 'mock'},
                                             'file': ('ქართ.wav', 'audio/wav', b'RIFF')}),
            ('dead upstream', 'POST', T, {'form': {'provider': 'dead', 'model': 'x'}, 'status_only': True}),
            ('progress', 'GET', '/api/progress?provider=mock', {}),
            ('partial', 'GET', '/api/partial?provider=mock&since=17', {}),
            ('progress, no control', 'GET', '/api/progress?provider=keyed', {}),
            ('progress, broken upstream', 'GET', '/api/progress?provider=broken', {}),
            ('progress, dead upstream', 'GET', '/api/progress?provider=dead', {}),
            ('progress, unknown provider', 'GET', '/api/progress?provider=zzz', {}),
            ('progress, no provider', 'GET', '/api/progress', {}),
            ('cancel', 'POST', '/api/cancel?provider=mock', {}),
            ('cancel, no control', 'POST', '/api/cancel?provider=keyed', {}),
            ('cancel, dead', 'POST', '/api/cancel?provider=dead', {'status_only': True}),
            ('recluster, model override', 'POST', '/api/recluster', {'body': json.dumps(
                {'provider': 'mock', 'model': 'diar', 'hash': 'abc', 'speakers': 3}).encode()}),
            ('recluster, provider control', 'POST', '/api/recluster', {'body': json.dumps(
                {'provider': 'mock', 'hash': 'abc'}).encode()}),
            ('recluster, bad JSON', 'POST', '/api/recluster', {'body': b'{nope', 'status_only': True}),
            ('recluster, empty', 'POST', '/api/recluster', {'body': b''}),
        ]
        bad = known = 0
        for name, method, path, kw in cases:
            status_only = kw.pop('status_only', False)
            a = call(f'http://127.0.0.1:{py_port}', method, path, **kw)
            b = call(f'http://127.0.0.1:{rs_port}', method, path, **kw)
            same = a[0] == b[0] if status_only else a == b
            if not same and name in KNOWN:
                print(f'known {name}  ({a[0]} / {b[0]})')
                known += 1
                continue
            print(f"{'ok  ' if same else 'DIFF'} {name}")
            if not same:
                bad += 1
                print(f'     serve.py: {a}\n     rust:     {b}')
        print(f'\n{len(cases) - bad - known}/{len(cases)} identical, {known} known differences, {bad} failures')
        sys.exit(1 if bad else 0)
    finally:
        for p in procs:
            p.terminate()
        mock.shutdown()


if __name__ == '__main__':
    main()
