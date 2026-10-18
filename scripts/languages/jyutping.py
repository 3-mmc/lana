"""Write web/jyutping.json, the data behind Lana's Cantonese romanisation tiers:
Han characters as Jyutping (MMS's yue-script_traditional and Whisper write
Cantonese in Han characters). Yale and IPA are derived from Jyutping by rule
in web/romanise.js.

Sources, fetched at build time only (Lana itself fetches nothing):
- rime-cantonese by CanCLID (CC BY 4.0; its jyut6ping3.maps, which is ODbL,
  is not used): jyut6ping3.chars.dict.yaml, every reading of each character
  with a weight (none for the usual reading, 5%, 3% or 0% for the others),
  and jyut6ping3.words.dict.yaml, the readings of 100,000-odd words, which
  settle a character with several readings (行 in 銀行 is hong4).
  https://github.com/rime/rime-cantonese
- The Hong Kong Cantonese Corpus, HKCanCor (CC BY 4.0; K. K. Luke and May
  L. Y. Wong 2015), as shipped in PyCantonese 5.0.0: 16,000 utterances of
  conversation and radio from the 1990s with the Jyutping of every word.
  They say how often each reading is said where no word settles the
  character. https://github.com/jacksonllee/pycantonese

How a reading is chosen is as for Mandarin (scripts/languages/romanise.py):
a character inside a rime-cantonese word takes the word's reading; outside
one, it takes the reading said most often there in HKCanCor, when HKCanCor
leaves it outside a word at least 20 times, and otherwise the reading
rime-cantonese weights highest. Its weights are read as shares: the
unweighted reading(s) take what the weighted ones leave (嘅 ge3 0.91, ge2
and koi2 0.03 each), and a character whose readings are all weighted (行:
haang4, hang4, hang6, hong4 at 5%, hong2 3%) shares them out in proportion.
A character is marked, with its other readings offered, when its second
reading is at least a tenth of the uses (and, in HKCanCor, said at least 3
times). Words are kept when they settle a marked character, or when a
character in them is not read its first way. The JSON is shared under
CC BY 4.0, with the attributions above.

usage: jyutping.py [--hold-out] [dir holding rime/ with the two .dict.yaml
files and pycantonese/x/pycantonese/data/hkcancor/*.cha]

--hold-out counts from every other HKCanCor file and reports how the result
reads the rest; the JSON is not written."""
import glob, json, os, re, sys
from collections import Counter, defaultdict

argv = sys.argv[1:]
hold_out = '--hold-out' in argv
argv = [a for a in argv if a != '--hold-out']
src = argv[0] if argv else os.path.expanduser('~/scratch/romanise-src')
out = os.path.join(os.path.dirname(__file__), '../../web/jyutping.json')
SYL = re.compile(r'[a-z]+[1-6]')

def entries(path):
    """The lines of a Rime dictionary after its header, split at tabs."""
    body = False
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line == '...':
            body = True
        elif body and line and not line.startswith('#'):
            yield line.split('\t')

weights = defaultdict(dict)
for p in entries(os.path.join(src, 'rime/jyut6ping3.chars.dict.yaml')):
    if len(p[0]) == 1 and SYL.fullmatch(p[1]):
        weights[p[0]].setdefault(p[1], float(p[2].rstrip('%')) / 100 if len(p) > 2 and p[2] else None)
listed = {}
for c, w in weights.items():
    rest = [r for r, x in w.items() if x is None]
    given = sum(x for x in w.values() if x is not None)
    share = {r: x for r, x in w.items() if x is not None}
    if rest:
        share.update({r: max(1 - given, 0) / len(rest) for r in rest})
    total = sum(share.values()) or 1
    listed[c] = {r: x / total for r, x in share.items()}
chars = {c: sorted(s, key=s.get, reverse=True) for c, s in listed.items()}

# A word listed with several readings (一個人 jan1 or jan4) takes, to begin
# with, the one its characters' weights favour.
variants = defaultdict(list)
for p in entries(os.path.join(src, 'rime/jyut6ping3.words.dict.yaml')):
    hz, sy = p[0], p[1].split(' ') if len(p) > 1 else []
    if len(hz) >= 2 and len(sy) == len(hz) and all(SYL.fullmatch(s) for s in sy) and all(c in chars for c in hz):
        variants[hz].append(sy)
likely = lambda hz, sy: sum(listed[c].get(s, 0) for c, s in zip(hz, sy))
word = {hz: max(vs, key=lambda sy: likely(hz, sy)) for hz, vs in variants.items()}
known = defaultdict(set)
for c, rs in chars.items():
    known[c].update(rs)
for hz, sy in word.items():
    for c, s in zip(hz, sy):
        known[c].add(s)

def utterances(paths):
    """HKCanCor's utterances as (characters, the reading said of each, or None)."""
    for path in paths:
        text = open(path, encoding='utf-8').read().replace('\n\t', ' ')
        said = None
        for line in text.split('\n'):
            if line.startswith('*'):
                said = line.split('\t', 1)[1].split() if '\t' in line else []
            elif line.startswith('%mor:') and said is not None:
                mor = line[5:].split()
                hz, rs = [], []
                if len(mor) == len(said):
                    for w, m in zip(said, mor):
                        sy = SYL.findall(m.split('|', 1)[1]) if '|' in m else []
                        han = [c for c in w if c in known]
                        if han and len(han) == len(w) and len(sy) == len(w):
                            hz += han
                            rs += [s if s in known[c] else None for c, s in zip(han, sy)]
                        else:
                            hz.append(None); rs.append(None)  # anything else breaks the run
                yield hz, rs
                said = None

def cut(hz, lookup):
    i = 0
    while i < len(hz):
        for n in range(min(8, len(hz) - i), 1, -1):
            if None not in hz[i:i + n] and ''.join(hz[i:i + n]) in lookup:
                yield i, n
                i += n
                break
        else:
            i += 1

files = sorted(glob.glob(os.path.join(src, 'pycantonese/x/pycantonese/data/hkcancor/*.cha')))
train, test = (files[0::2], files[1::2]) if hold_out else (files, [])
heard = list(utterances(train))

# What HKCanCor says of the words longest match finds there, as for
# Mandarin: a word matched at least MIN_WORD times takes the listed reading
# said most (上邊 bin6 over bin1); if none is said half the time, the one
# said nine times in ten (會唔會 wui5 m4 wui5, listed wui3 m4 wui3 and wui6
# m4 wui5), and failing that the word is dropped when reading it character
# by character is right more often and gives other syllables, not only
# other tones (係嘞 is seldom the word: 係 hai6 then 嘞 laak3).
MIN_WORD = 5
matched = defaultdict(Counter)
for hz, rs in heard:
    for i, n in cut(hz, word):
        if all(rs[i:i + n]):
            matched[''.join(hz[i:i + n])][tuple(rs[i:i + n])] += 1
respelt, dropped = {}, set()
for w, got in matched.items():
    total = sum(got.values())
    if total < MIN_WORD:
        continue
    best = max(variants[w], key=lambda sy: (got[tuple(sy)], likely(w, sy)))
    own = got[tuple(best)]
    if own >= total / 2:
        if best != word[w]:
            respelt[w] = best
        continue
    r, k = got.most_common(1)[0]
    one_by_one = sum(k for r, k in got.items() if all(s == chars[c][0] for s, c in zip(r, w)))
    if k >= 0.9 * total:
        respelt[w] = list(r)
    elif one_by_one > own and any(s.rstrip('123456') != chars[c][0].rstrip('123456') for s, c in zip(best, w)):
        dropped.add(w)
for w in dropped:
    del word[w]
word.update(respelt)

said = defaultdict(Counter)
for hz, rs in heard:
    inside = set()
    for i, n in cut(hz, word):
        inside.update(range(i, i + n))
    for i, (c, r) in enumerate(zip(hz, rs)):
        if c and r and i not in inside:
            said[c][r] += 1

MIN_SAID = 20
doubt, basis = set(), {}
for c in chars:
    if len(known[c]) < 2:
        continue
    heard = sum(said[c].values()) >= MIN_SAID
    counts = said[c] if heard else listed[c]
    n = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if len(n) > 1 and n[1][1] >= 0.1 * sum(counts.values()) and (n[1][1] >= 3 if heard else True):
        doubt.add(c)
    if heard or c in doubt:
        basis[c] = counts
        chars[c] = [r for r, k in n if k] + [r for r in chars[c] if not counts.get(r)]
for c in known:
    chars[c] += [r for r in sorted(known[c]) if r not in chars[c]]

words = {}
for hz, sy in word.items():
    same = all(s == chars[c][0] for c, s in zip(hz, sy))
    if not same or any(c in doubt for c in hz):
        words[hz] = '' if same else ' '.join(sy)
shares = {}
for c in doubt:
    total = sum(basis[c].values())
    shares[c] = [round(basis[c].get(r, 0) / total, 2) for r in chars[c]]
data = {'chars': {c: ' '.join(r) for c, r in chars.items()}, 'doubt': ''.join(sorted(doubt)), 'shares': shares, 'words': words}

def convert(d, hz):
    C, W, D = d['chars'], d['words'], set(d['doubt'])
    got, i = [], 0
    while i < len(hz):
        for n in range(min(8, len(hz) - i), 1, -1):
            w = W.get(''.join(hz[i:i + n])) if None not in hz[i:i + n] else None
            if w is not None:
                got += [(s, False) for s in (w.split(' ') if w else [C[c].split(' ')[0] for c in hz[i:i + n]])]
                i += n
                break
        else:
            got.append((C[hz[i]].split(' ')[0], hz[i] in D) if hz[i] else (None, False))
            i += 1
    return got

if hold_out:
    k = Counter()
    for hz, rs in utterances(test):
        for (got, marked), r, c in zip(convert(data, hz), rs, hz):
            if r is None or len(known[c]) < 2:
                continue
            k['polyphonic'] += 1
            k['wrong'] += got != r
            k['marked'] += marked
            k['marked, right'] += marked and got == r
            k['wrong, unmarked'] += not marked and got != r
    print(f"{len(doubt)} characters marked; on the other half of HKCanCor, of {k['polyphonic']} characters with more than "
          f"one reading: {k['wrong']} read wrong ({k['wrong'] / k['polyphonic']:.2%}), {k['marked']} marked of which "
          f"{k['marked, right']} were right anyway, {k['wrong, unmarked']} wrong and not marked")
    sys.exit()

json.dump(data, open(out, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print(len(respelt), 'words respelt and', len(dropped), 'dropped by what HKCanCor heard;', len(chars), 'characters,', len(doubt), 'read more than one way,', len(words), 'words,', os.path.getsize(out), 'bytes;',
      sum(sum(said[c].values()) >= MIN_SAID for c in doubt), 'judged by HKCanCor')
for c in '的還了長行得著地都為嘅係佢冇睇咗乜嘢':
    print(c, ' '.join(chars[c][:4]), 'marked' if c in doubt else '-', shares.get(c, ''), dict(said[c].most_common(4)))
for w in ('銀行', '行路', '長大', '還錢', '音樂', '快樂'):
    print(w, words.get(w), [chars[c][0] for c in w])
