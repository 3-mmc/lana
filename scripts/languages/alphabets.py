"""Write src-tauri/src/local/alphabets.json: for each MMS language, the letters
of its alphabet, which Lana uses to hold Omnilingual ASR to the right script
(left free, it writes Dargwa in Arabic, Latin or Georgian letters).

The source is MMS-1b-all's vocab.json, the characters each language's adapter
was trained to write. Those lists carry stray letters from their training
text (Georgian's includes Latin a-z), so only letters of the scripts that
make up at least 15 % of a language's letters are kept (Japanese keeps Han,
hiragana and katakana), and Latin goes whenever another script qualifies:
in these lists Latin beside another script is loan text, not the
orthography (Georgian has 28 Latin letters to 33 Georgian). Kept with them: combining marks and apostrophes, which several
orthographies use as letters. Lower case only; Lana adds the capitals.

usage: alphabets.py [path to vocab.json]"""
import json, os, sys, unicodedata
from collections import Counter

src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/models/mms-1b-all/vocab.json')
out = os.path.join(os.path.dirname(__file__), '../../src-tauri/src/local/alphabets.json')

def script(c):
    name = unicodedata.name(c, '')
    if name.startswith('CJK'):
        return 'HAN'
    return name.split(' ')[0]

result = {}
for lang, vocab in sorted(json.load(open(src, encoding='utf-8')).items()):
    chars = [c for c in vocab if len(c) == 1]
    letters = [c for c in chars if c.isalpha()]
    counts = Counter(script(c) for c in letters)
    keep = {s for s, n in counts.items() if n >= 0.15 * len(letters)}
    if len(keep) > 1:
        keep.discard('LATIN')
    kept = [c for c in chars if (c.isalpha() and script(c) in keep)
            or unicodedata.category(c).startswith('M') or c in "'ʼ’"]
    result[lang] = ''.join(sorted(set(kept)))
json.dump(result, open(out, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print(len(result), 'languages,', os.path.getsize(out), 'bytes')
for k in ('dar', 'kat', 'oss', 'jpn', 'deu'):
    print(k, result.get(k, '')[:80])
