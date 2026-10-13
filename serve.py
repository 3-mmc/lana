#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Serve Lana (formerly praat-web), and optionally broker its recognition requests.

    python3 serve.py                 # http://127.0.0.1:8710
    python3 serve.py --port 9000
    python3 serve.py --host 0.0.0.0  # reachable from the LAN

Standard library only — no pip install, no build step. The one external thing is
the praat-wasm npm tarball (~24 MB), downloaded once into ./vendor/ and cached.
If you would rather fetch it yourself:

    npm pack praat-wasm@6.4.6200 && tar xzf praat-wasm-6.4.6200.tgz
    mv package vendor

praat-wasm is GPL-3.0-or-later, as is this project. It is not vendored into git
because 24 MB of binary does not belong in a source repository.

THE BROKER
----------
Copy providers.example.json to providers.json and the server grows a /api/
surface that stands between the page and one or more recognisers:

    GET  /api/providers              what is configured (never any key material)
    POST /api/transcribe             multipart upload -> normalised cues
    GET  /api/progress?provider=id   proxied, if the provider declares a control path
    GET  /api/partial?provider=id&since=n     cues decoded so far
    POST /api/cancel?provider=id

Three reasons this is worth a server rather than fetching from the page:

  1. An API key stays on this machine. The browser is told that a provider has
     a key, never what it is.
  2. No CORS. Everything the page talks to is same-origin, which is what makes
     a hosted service usable by someone who did not configure the recogniser.
  3. One response shape. Providers disagree about where word timings live; the
     normaliser here means the page only ever parses one contract.

With no providers.json the /api/ surface reports an empty list and the page
falls back to talking to an endpoint you type in yourself. Everything except
recognition — analysis, tiers, TextGrid export — needs no backend at all.
"""
import argparse
import io
import json
import mimetypes
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from email.parser import BytesParser
import email.policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get('PRAAT_WASM_DIR', HERE / 'vendor'))
PROVIDERS_FILE = Path(os.environ.get('PRAAT_WEB_PROVIDERS', HERE / 'providers.json'))
PKG = 'praat-wasm'
VERSION = '6.4.6200'
TARBALL = f'https://registry.npmjs.org/{PKG}/-/{PKG}-{VERSION}.tgz'

MAX_UPLOAD = 512 * 1024 * 1024
ASR_TIMEOUT = 3600          # a film is an hour of GPU work; do not cut it short
CONTROL_TIMEOUT = 10

TYPES = {
    '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript',
    '.wasm': 'application/wasm', '.json': 'application/json', '.css': 'text/css',
    '.map': 'application/json', '.ts': 'text/plain', '.svg': 'image/svg+xml',
}


def ensure_vendor() -> None:
    """Download and unpack praat-wasm unless it is already there."""
    if (VENDOR / 'js' / 'worker-client.mjs').is_file():
        return
    print(f'praat-wasm not found at {VENDOR}', file=sys.stderr)
    print(f'downloading {TARBALL} (~24 MB, once)…', file=sys.stderr)
    VENDOR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TARBALL, timeout=300) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tf:
        for member in tf.getmembers():
            # npm tarballs put everything under "package/"; strip it, and refuse
            # any path that would escape the vendor directory.
            name = member.name.split('/', 1)[-1] if member.name.startswith('package/') else None
            if not name or member.isdev() or member.issym() or member.islnk():
                continue
            dest = (VENDOR / name).resolve()
            if not str(dest).startswith(str(VENDOR.resolve())):
                continue
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src:
                dest.write_bytes(src.read())
    if not (VENDOR / 'js' / 'worker-client.mjs').is_file():
        sys.exit('praat-wasm did not unpack as expected')
    print(f'praat-wasm {VERSION} ready in {VENDOR}', file=sys.stderr)


# --------------------------------------------------------------- providers

_providers_cache = {'mtime': None, 'value': []}


def load_providers() -> list:
    """Read providers.json, re-reading it whenever the file changes on disk.

    Re-reading rather than caching for the process lifetime is deliberate: the
    common case is someone adding a key and wondering why the page has not
    noticed, and a restart is a worse answer than a stat() per request.
    """
    try:
        mtime = PROVIDERS_FILE.stat().st_mtime
    except OSError:
        _providers_cache.update(mtime=None, value=[])
        return []
    if _providers_cache['mtime'] == mtime:
        return _providers_cache['value']
    try:
        doc = json.loads(PROVIDERS_FILE.read_text(encoding='utf-8'))
        value = [p for p in doc.get('providers', []) if p.get('id')]
    except Exception as exc:
        print(f'providers.json is not usable: {exc}', file=sys.stderr)
        value = []
    _providers_cache.update(mtime=mtime, value=value)
    return value


def find_provider(pid: str):
    return next((p for p in load_providers() if p.get('id') == pid), None)


def provider_key(p: dict):
    """The provider's API key, or None. Checked in the order a user expects:
    an explicit key in the file beats the environment."""
    return p.get('api_key') or (os.environ.get(p['api_key_env']) if p.get('api_key_env') else None)


def public_provider(p: dict) -> dict:
    """What the browser is allowed to know. No key, ever — not even its length."""
    needs_key = bool(p.get('api_key') or p.get('api_key_env'))
    has_key = bool(provider_key(p))
    note = ''
    if needs_key and not has_key:
        note = f"no key: set {p.get('api_key_env') or 'api_key'} and reload"
    return {
        'id': p['id'],
        'label': p.get('label') or p['id'],
        'kind': p.get('kind', 'openai'),
        'base': p.get('base', ''),
        'models': p.get('models', []),
        'control': bool(p.get('control')),
        'needs_key': needs_key,
        'has_key': has_key,
        'ready': has_key or not needs_key,
        'note': note,
    }


def control_url(p: dict, path: str, model: str | None = None) -> str | None:
    """The control base for a provider, or for one model of it.

    A gateway that swaps models puts each service behind its own path, so the
    diarizer's progress and re-clustering do not live where the recogniser's
    do. A model may therefore override the provider's `control`.
    """
    base = p.get('control')
    if model:
        for m in p.get('models') or []:
            if m.get('id') == model and m.get('control'):
                base = m['control']
                break
    if not base:
        return None
    return p['base'].rstrip('/') + '/' + base.strip('/') + path


# ------------------------------------------------------- request forwarding

def build_multipart(fields: dict, filename: str, filetype: str, blob: bytes) -> tuple[bytes, str]:
    boundary = '----praatweb' + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        if v is None:
            continue
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        out.write(str(v).encode('utf-8') + b'\r\n')
    out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
              f'filename="{filename}"\r\nContent-Type: {filetype}\r\n\r\n'.encode())
    out.write(blob + b'\r\n')
    out.write(f'--{boundary}--\r\n'.encode())
    return out.getvalue(), f'multipart/form-data; boundary={boundary}'


def normalise(payload: dict) -> dict:
    """Coerce a transcription response into this project's cue contract.

    Our own services already speak it. OpenAI-shaped services return `segments`,
    and word timings only if `timestamp_granularities[]=word` was asked for, in
    which case they arrive in a flat top-level `words` list that has to be put
    back into its segments. A response with neither still transcribes; it simply
    cannot draw a tier, and the page says so rather than inventing boundaries.
    """
    if not isinstance(payload, dict):
        return {'text': '', 'cues': []}
    if 'cues' in payload:
        return payload

    out = {
        'text': payload.get('text', ''),
        'language': payload.get('language'),
        'duration': payload.get('duration'),
        'cues': [],
    }
    segs = payload.get('segments') or []
    words = [w for w in (payload.get('words') or [])
             if w.get('start') is not None and w.get('end') is not None]

    def word_obj(w):
        return {'text': w.get('word') or w.get('text') or '',
                'start': float(w['start']), 'end': float(w['end']),
                'phones': [], 'phone_times': [], 'aligned': True}

    if segs:
        buckets = [[] for _ in segs]
        bounds = [(float(s.get('start', 0)), float(s.get('end', 0))) for s in segs]
        for w in words:
            mid = (float(w['start']) + float(w['end'])) / 2
            idx = next((i for i, (a, b) in enumerate(bounds) if a <= mid <= b), None)
            if idx is None:          # between segments: give it the nearest one
                idx = min(range(len(bounds)),
                          key=lambda i: min(abs(mid - bounds[i][0]), abs(mid - bounds[i][1])))
            buckets[idx].append(w)
        for s, (a, b), ws in zip(segs, bounds, buckets):
            cue = {'start': a, 'end': b, 'words': [word_obj(w) for w in ws]}
            if not ws:
                cue['plain'] = (s.get('text') or '').strip()
            out['cues'].append(cue)
    elif words:
        out['cues'].append({'start': float(words[0]['start']), 'end': float(words[-1]['end']),
                            'words': [word_obj(w) for w in words]})
    return out


def forward_transcription(p: dict, fields: dict, filename: str, filetype: str,
                          blob: bytes) -> tuple[int, dict]:
    kind = p.get('kind', 'openai')
    if kind != 'openai':
        return 400, {'error': f'provider kind {kind!r} is not implemented. '
                              'Add it to KINDS in serve.py — see normalise().'}
    send = {
        'model': fields.get('model'),
        'response_format': fields.get('response_format') or 'verbose_json',
    }
    if fields.get('language'):
        send['language'] = fields['language']
    # Pass through the extras our own services understand: whisper size and
    # translation target for the recognisers, clustering arguments for the
    # diarizer. A provider that does not know them ignores them; OpenAI rejects
    # unknown fields on some routes, which is why they are opt-in per provider
    # rather than always sent. Anything missing here is silently dropped, which
    # looks exactly like a model ignoring an argument -- add the name.
    for extra in ('asr_model', 'translate_to',
                  'speakers', 'threshold', 'min_on', 'min_off', 'embedding'):
        if fields.get(extra) and p.get('pass_extras', True):
            send[extra] = fields[extra]
    send.update(p.get('fields') or {})

    body, ctype = build_multipart(send, filename, filetype, blob)
    url = p['base'].rstrip('/') + '/v1/audio/transcriptions'
    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', ctype)
    key = provider_key(p)
    if key:
        req.add_header('Authorization', 'Bearer ' + key)
    for k, v in (p.get('headers') or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=ASR_TIMEOUT) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')[:600]
        try:
            detail = json.loads(detail).get('error', detail)
        except Exception:
            pass
        if isinstance(detail, dict):
            detail = detail.get('message') or json.dumps(detail)[:600]
        return e.code, {'error': detail, 'provider': p['id']}
    except urllib.error.URLError as e:
        return 502, {'error': f"cannot reach {p['id']} at {p.get('base')}: {e.reason}"}
    try:
        return 200, normalise(json.loads(raw))
    except json.JSONDecodeError:
        return 200, {'text': raw.decode('utf-8', 'replace'), 'cues': []}


def proxy_control(p: dict, path: str, method='GET', model=None,
                  body: bytes | None = None) -> tuple[int, dict]:
    url = control_url(p, path, model)
    if not url:
        return 404, {'error': 'this provider declares no control endpoint'}
    data = body if body is not None else (b'' if method == 'POST' else None)
    req = urllib.request.Request(url, method=method, data=data)
    if body is not None:
        req.add_header('Content-Type', 'application/json')
    key = provider_key(p)
    if key:
        req.add_header('Authorization', 'Bearer ' + key)
    try:
        with urllib.request.urlopen(req, timeout=CONTROL_TIMEOUT) as r:
            return 200, json.loads(r.read() or b'{}')
    except Exception as e:
        return 502, {'error': str(e)}


# ------------------------------------------------------------------ server

class Handler(BaseHTTPRequestHandler):
    server_version = 'lana-web'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *a):
        print(f'{self.address_string()} {fmt % a}', flush=True)

    # -- helpers ----------------------------------------------------------
    def route(self) -> str:
        """Path with any reverse-proxy prefix removed.

        llama-swap serves this page at /upstream/praat/, and the browser sends
        that prefix back on every request it makes.
        """
        path = urlparse(self.path).path
        for marker in ('/upstream/praat/', '/upstream/praat'):
            if path.startswith(marker):
                return '/' + path[len(marker):].lstrip('/')
        return path

    def query(self) -> dict:
        return parse_qs(urlparse(self.path).query)

    def send_json(self, status: int, value) -> None:
        body = json.dumps(value).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def pick_provider(self, pid=None):
        pid = pid or (self.query().get('provider') or [''])[0]
        p = find_provider(pid)
        if not p:
            self.send_json(404, {'error': f'no provider {pid!r} is configured'})
            return None
        return p

    def resolve(self):
        path = self.route().lstrip('/')
        if path in ('', 'index.html'):
            return HERE / 'index.html'
        if path == 'providers.json':
            return None                  # never serve the file with the keys in it
        base, rel = (VENDOR, path[len('vendor/'):]) if path.startswith('vendor/') else (HERE, path)
        target = (base / rel).resolve()
        if not str(target).startswith(str(base.resolve())):
            return None
        return target if target.is_file() else None

    # -- GET --------------------------------------------------------------
    def do_GET(self):
        route = self.route()
        if route.rstrip('/').endswith('/health') or route == '/api/health':
            return self.send_json(200, {'status': 'ok', 'service': 'lana-web',
                                        'providers': len(load_providers())})
        if route == '/api/providers':
            ps = [public_provider(p) for p in load_providers()]
            return self.send_json(200, {'providers': ps,
                                        'configured': str(PROVIDERS_FILE) if ps else None})
        if route in ('/api/progress', '/api/partial'):
            p = self.pick_provider()
            if not p:
                return
            if not p.get('control'):
                return self.send_json(200, {'active': False, 'unsupported': True})
            tail = '/progress' if route == '/api/progress' else \
                   '/partial?since=' + (self.query().get('since') or ['0'])[0]
            code, body = proxy_control(p, tail)
            return self.send_json(code if code == 200 else 200,
                                  body if code == 200 else {'active': False, 'unsupported': True})

        target = self.resolve()
        if target is None:
            self.send_error(404, 'Not found')
            return
        data = target.read_bytes()
        ctype = TYPES.get(target.suffix, 'application/octet-stream')
        if ctype.startswith(('text/', 'application/json')):
            ctype += '; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        if target.suffix in ('.wasm', '.mjs'):
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
        self.end_headers()
        self.wfile.write(data)

    # -- POST -------------------------------------------------------------
    def do_POST(self):
        route = self.route()
        if route == '/api/cancel':
            p = self.pick_provider()
            if not p:
                return
            code, body = proxy_control(p, '/cancel', method='POST')
            return self.send_json(code, body)
        if route == '/api/recluster':
            # A re-cluster carries no audio: the service still holds the
            # recording from the pass that produced this hash.
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 64 * 1024:
                return self.send_json(400, {'error': 'expected a small JSON body'})
            raw = self.rfile.read(length)
            try:
                want = json.loads(raw)
            except json.JSONDecodeError as exc:
                return self.send_json(400, {'error': f'bad JSON: {exc}'})
            p = self.pick_provider(want.get('provider'))
            if not p:
                return
            code, body = proxy_control(
                p, '/recluster', method='POST', model=want.get('model'),
                body=json.dumps({'hash': want.get('hash'),
                                 'speakers': want.get('speakers')}).encode())
            return self.send_json(code, body)
        if route != '/api/transcribe':
            return self.send_json(404, {'error': 'Not found'})

        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= MAX_UPLOAD:
            return self.send_json(413, {'error': f'upload must be 1 byte to {MAX_UPLOAD >> 20} MiB'})
        ct = self.headers.get('Content-Type', '')
        if not ct.startswith('multipart/form-data;'):
            return self.send_json(400, {'error': 'use multipart/form-data'})
        body = self.rfile.read(length)
        message = BytesParser(policy=email.policy.default).parsebytes(
            ('Content-Type: ' + ct + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + body)
        fields, blob, filename, filetype = {}, None, 'audio.wav', 'application/octet-stream'
        for part in message.iter_parts():
            name = part.get_param('name', header='content-disposition')
            if name == 'file':
                blob = part.get_payload(decode=True)
                filename = part.get_filename() or filename
                # get_content_type() never returns nothing: a part without a
                # Content-Type comes back as text/plain, so ask whether one was sent.
                sent = (part.get('Content-Type') or '').strip()
                filetype = (part.get_content_type() if sent else None) or \
                    mimetypes.guess_type(filename)[0] or filetype
            elif name:
                fields[name] = part.get_payload(decode=True).decode('utf-8')
        if not blob:
            return self.send_json(400, {'error': 'file is required'})

        p = self.pick_provider(fields.get('provider'))
        if not p:
            return
        if not public_provider(p)['ready']:
            return self.send_json(400, {'error': public_provider(p)['note']})
        started = time.time()
        code, payload = forward_transcription(p, fields, filename, filetype, blob)
        if code == 200:
            payload.setdefault('provider', p['id'])
            payload.setdefault('model', fields.get('model'))
            payload['wall_seconds'] = round(time.time() - started, 2)
        self.send_json(code, payload)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Serve Lana.')
    ap.add_argument('--port', type=int, default=8710)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    ensure_vendor()
    ps = load_providers()
    if ps:
        print(f'{len(ps)} provider(s) from {PROVIDERS_FILE}: '
              + ', '.join(f"{p['id']}{'' if public_provider(p)['ready'] else ' (no key)'}"
                          for p in ps), flush=True)
        if args.host not in ('127.0.0.1', 'localhost', '::1') and \
           any(provider_key(p) for p in ps):
            print('WARNING: bound to a reachable address with API keys configured. '
                  'Anyone who can open this page can spend them. There is no '
                  'authentication here — put it behind one, or bind 127.0.0.1.',
                  file=sys.stderr, flush=True)
    else:
        print(f'no providers configured (copy providers.example.json to '
              f'{PROVIDERS_FILE.name} to add some)', flush=True)
    print(f'Lana on http://{args.host}:{args.port}', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
