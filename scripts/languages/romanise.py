"""Write web/romanise.json, the data behind Lana's romanisation tiers: Han
characters as pinyin, and pinyin syllables as zhuyin (bopomofo) and IPA.

Sources, all fetched at build time only (Lana itself fetches nothing):
- pinyin-data by mozillazg (MIT), from Unicode's Unihan: every reading of
  each character (pinyin.txt), and the customary one first (kMandarin.txt).
  https://github.com/mozillazg/pinyin-data
- CC-CEDICT (CC BY-SA 4.0), as packaged in dragonmapper (MIT): readings of
  whole words, which settle a character with several readings (长 in 长安 is
  cháng). They also say which characters are really read more than one way,
  and which reading is the usual one. Only words with such a character in
  them are kept.
- dragonmapper's transcriptions.csv (MIT): the 400-odd toneless Mandarin
  syllables in pinyin, zhuyin and IPA.

Wade-Giles is derived by rule in web/romanise.js, not stored here.

usage: romanise.py <dir holding pinyin.txt, kMandarin.txt and an unpacked dragonmapper wheel in dm/>"""
import csv, json, os, re, sys, unicodedata

src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/scratch/romanise-src')
out = os.path.join(os.path.dirname(__file__), '../../web/romanise.json')
dm = os.path.join(src, 'dm/dragonmapper/data')

def readings(path):
    r = {}
    for line in open(path, encoding='utf-8'):
        m = re.match(r'U\+([0-9A-F]+): ([^#]+)#', line)
        if m:
            r[chr(int(m.group(1), 16))] = [p.strip() for p in m.group(2).split(',') if p.strip()]
    return r

every, first = readings(os.path.join(src, 'pinyin.txt')), readings(os.path.join(src, 'kMandarin.txt'))
chars = {}
for c, rs in every.items():
    head = first.get(c, rs)[:1]
    chars[c] = head + [r for r in rs if r not in head]

syllables = {}
for row in csv.DictReader(open(os.path.join(dm, 'transcriptions.csv'), encoding='utf-8')):
    syllables[row['Pinyin']] = [row['Zhuyin'], row['IPA']]
toneless = sorted(syllables, key=len, reverse=True)

def bare(s):
    s = unicodedata.normalize('NFD', s.lower())
    return unicodedata.normalize('NFC', ''.join(ch for ch in s if ch not in '̀́̄̌'))

def split(pinyin, n):
    """A word's pinyin as n syllables (one per character), or None."""
    p = unicodedata.normalize('NFC', pinyin.replace("'", '').replace(' ', ''))
    b = bare(p)
    out, i = [], 0
    def go(i, k):
        if k == n:
            return [] if i == len(b) else None
        for s in toneless:
            if b.startswith(s.replace('ü', 'ü'), i):
                rest = go(i + len(s), k + 1)
                if rest is not None:
                    return [p[i:i + len(s)].lower()] + rest
        return None
    return go(0, 0)

# Every word's reading, character by character, so that how often each
# reading of a character is used can be counted.
parsed = []
for line in open(os.path.join(dm, 'hanzi_pinyin_words.tsv'), encoding='utf-8'):
    hz, py = line.rstrip('\n').split('\t')[:2]
    py = py.split('/')[0]
    if len(hz) < 2 or not all(c in chars for c in hz):
        continue
    sy = split(py, len(hz))
    if sy:
        parsed.append((hz, sy))
used = {}
for hz, sy in parsed:
    for c, s in zip(hz, sy):
        used.setdefault(c, {}).setdefault(s, 0)
        used[c][s] += 1

# A character really read more than one way: its second reading is at least
# a tenth of its uses in CC-CEDICT, in three words or more. Unihan lists
# readings that are almost never used (去 qū beside qù), and marking those
# would mark nearly every character.
def doubtful(c):
    n = sorted(used.get(c, {}).values(), reverse=True)
    return len(n) > 1 and n[1] >= 3 and n[1] >= 0.1 * sum(n)
doubt = {c for c in chars if len(chars[c]) > 1 and doubtful(c)}
# The reading most used comes first, where CC-CEDICT has a clear view.
for c in doubt:
    common = max(used[c].items(), key=lambda kv: kv[1])[0]
    if common in chars[c]:
        chars[c] = [common] + [r for r in chars[c] if r != common]

# Words that settle a doubtful character: kept with their reading, or with
# "" when every character takes its first reading.
words = {}
for hz, sy in parsed:
    if any(c in doubt for c in hz):
        same = all(s == chars[c][0] for c, s in zip(hz, sy))
        words[hz] = '' if same else ' '.join(sy)

json.dump({'chars': {c: ' '.join(r) for c, r in chars.items()}, 'doubt': ''.join(sorted(doubt)),
           'words': words, 'syllables': syllables},
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print(len(chars), 'characters,', len(doubt), 'read more than one way,', len(words), 'words,', len(syllables), 'syllables,', os.path.getsize(out), 'bytes')
for w in ('长安', '银行', '行走', '音乐', '快乐', '我们'):
    print(w, words.get(w), [chars[c] for c in w])
