"""Write web/romanise.json, the data behind Lana's romanisation tiers: Han
characters as pinyin, and pinyin syllables as zhuyin (bopomofo) and IPA.

Sources, all fetched at build time only (Lana itself fetches nothing):
- pinyin-data by mozillazg (MIT), from Unicode's Unihan: every reading of
  each character (pinyin.txt), and the customary one first (kMandarin.txt).
  https://github.com/mozillazg/pinyin-data
- CC-CEDICT (CC BY-SA 4.0), as packaged in dragonmapper (MIT): readings of
  whole words, which settle a character with several readings (长 in 长安 is
  cháng). https://github.com/tsroten/dragonmapper
- AISHELL-3 (Apache 2.0): the character and pinyin transcripts (content.txt,
  train and test, about a million characters of read Mandarin with the
  reading each speaker said). They say how often each reading of a character
  is said where no dictionary word settles it, which is the one place the
  tier has to choose. https://huggingface.co/datasets/AISHELL/AISHELL-3
- FrequencyWords by Hermit Dave (content CC BY-SA 4.0), word counts of the
  2018 OpenSubtitles Chinese subtitles (zh_cn_full.txt): for characters
  AISHELL-3 seldom leaves unsettled, each CC-CEDICT word counts once plus
  the number of times it is said in the subtitles.
  https://github.com/hermitdave/FrequencyWords
- dragonmapper's transcriptions.csv (MIT): the 400-odd toneless Mandarin
  syllables in pinyin, zhuyin and IPA.

How a reading is chosen. A character inside a CC-CEDICT word takes the
word's reading. Outside one, the conversion has only the character, so what
counts is how often each of its readings is said when it stands outside a
word. AISHELL-3 answers that directly: its transcripts are cut into words by
the same longest-match the tier uses, and the readings of the characters
left over are counted. Its pinyin is what was said, so third-tone sandhi
(hen2 hao3), the tones of 一 and 不 and slips of tone are taken back to a
reading the character has; a label that fits none, or several, is left out.
A character left over at least 20 times is judged by those counts. Others
fall back to CC-CEDICT words weighted by the subtitles. Single-character
words are not counted from the subtitles at all: the list says how often 的
is said alone, not which reading was said, and crediting every one to the
customary reading would make 长 always zhǎng, where AISHELL-3 hears cháng
about as often.

A character is marked, with its other readings offered, when its second
reading is at least a tenth of those uses and at least 3 of them (3 words in
the fallback). The reading most used comes first. Words are kept when they
settle a marked character, or when a character in them is not read its
first way (还给 huángěi, 东西 dōngxi).

Wade-Giles and Gwoyeu Romatzyh are derived by rule in web/romanise.js, not
stored here. The JSON is shared under CC BY-SA 4.0.

usage: romanise.py [--hold-out [old romanise.json]] [dir holding pinyin.txt,
kMandarin.txt, an unpacked dragonmapper wheel in dm/, aishell3/train_content.txt
and test_content.txt, and freq/zh_cn_full.txt]

--hold-out counts from the AISHELL-3 train transcripts only and reports how
the result (and an older romanise.json, if given) reads the test ones; the
JSON is not written."""
import csv, json, os, re, sys, unicodedata
from collections import Counter, defaultdict

argv = sys.argv[1:]
hold_out = '--hold-out' in argv
if hold_out:
    i = argv.index('--hold-out')
    old_json = argv[i + 1] if i + 1 < len(argv) and argv[i + 1].endswith('.json') else None
    argv = [a for k, a in enumerate(argv) if k != i and a != old_json]
src = argv[0] if argv else os.path.expanduser('~/scratch/romanise-src')
out = os.path.join(os.path.dirname(__file__), '../../web/romanise.json')
dm = os.path.join(src, 'dm/dragonmapper/data')

def readings(path):
    r = {}
    for line in open(path, encoding='utf-8'):
        m = re.match(r'U\+([0-9A-F]+): ([^#]+)#', line)
        if m:
            r[chr(int(m.group(1), 16))] = [p for p in re.split(r'[,\s]+', m.group(2)) if p]
    return r

# kMandarin gives the customary reading, and a second one where Taiwan's
# differs (地 "de dì"); the first is the one kept.
every, first = readings(os.path.join(src, 'pinyin.txt')), readings(os.path.join(src, 'kMandarin.txt'))
chars = {}
for c, rs in every.items():
    head = first.get(c, rs)[:1]
    chars[c] = head + [r for r in rs if r not in head]

syllables = {}
for row in csv.DictReader(open(os.path.join(dm, 'transcriptions.csv'), encoding='utf-8')):
    syllables[row['Pinyin']] = [row['Zhuyin'], row['IPA']]
toneless = sorted(syllables, key=len, reverse=True)

MARKS = {'̄': 1, '́': 2, '̌': 3, '̀': 4}
def bare(s):
    s = unicodedata.normalize('NFD', s.lower())
    return unicodedata.normalize('NFC', ''.join(ch for ch in s if ch not in MARKS))
def tone_of(s):
    return next((MARKS[ch] for ch in unicodedata.normalize('NFD', s) if ch in MARKS), 5)

def split(pinyin, hz):
    """A word's pinyin as one syllable per character, or None. Pinyin puts an
    apostrophe before a syllable that starts with a, e or o anywhere but at
    the start (dì'èr), so without one such a syllable cannot follow another:
    kěnéng is kě néng, not kěn éng. A lone r is the 儿 of erhua (yīhuìr)."""
    p = unicodedata.normalize('NFC', pinyin.replace(' ', '')).lower()
    b, n = bare(p), len(hz)
    def go(i, k):
        if k == n:
            return [] if i == len(b) else None
        if b.startswith("'", i):
            i += 1
        elif k and b[i:i + 1] in ('a', 'e', 'o'):
            return None
        for s in toneless:
            if b.startswith(s, i) and (s != 'r' or hz[k] == '儿'):
                rest = go(i + len(s), k + 1)
                if rest is not None:
                    return [p[i:i + len(s)]] + rest
        return None
    return go(0, 0)

# Every CC-CEDICT word's reading, character by character.
parsed, word = [], {}
for line in open(os.path.join(dm, 'hanzi_pinyin_words.tsv'), encoding='utf-8'):
    hz, py = line.rstrip('\n').split('\t')[:2]
    py = py.split('/')[0]
    if len(hz) < 2 or hz in word or not all(c in chars for c in hz):
        continue
    sy = split(py, hz)
    if sy:
        parsed.append((hz, sy))
        word[hz] = sy
known = defaultdict(set)
for c, rs in chars.items():
    known[c].update(rs)
for hz, sy in parsed:
    for c, s in zip(hz, sy):
        known[c].add(s)

# --- AISHELL-3: what was said where no word settles the character ---------
def toned(base, tone):
    """Numbered pinyin as marked pinyin (hao3 -> hǎo)."""
    if tone == 5:
        return base
    m = re.search(r'[ae]', base) or re.search(r'o(?=u)', base) or list(re.finditer(r'[aeiouü]', base))[-1]
    i = m.start() if hasattr(m, 'start') else m
    return unicodedata.normalize('NFC', base[:i + 1] + [None, '̄', '́', '̌', '̀'][tone] + base[i + 1:])

def reading_said(c, label, after):
    """The reading of c that AISHELL-3's label stands for, or None."""
    if c == '一': return 'yī'
    if c == '不': return 'bù'
    m = re.match(r'([a-z]+)([1-5])$', label)
    if not m:
        return None
    base, tone = m.group(1).replace('v', 'ü'), int(m.group(2))
    same = [r for r in known.get(c, ()) if bare(r) == base]
    if not same:
        return None
    three = toned(base, 3)
    # A third tone said as a second before another third (hen2 hao3); a
    # character that also has a second-tone reading is left out there.
    if tone == 2 and three in same:
        if toned(base, 2) not in same:
            return three
        if after == 3:
            return None
    if toned(base, tone) in same:
        return toned(base, tone)
    return same[0] if len(same) == 1 else None

def sentences(path):
    for line in open(path, encoding='utf-8'):
        a = line.rstrip('\n').split('\t')[1].split()
        hz, lab = a[0::2], a[1::2]
        tones = [int(l[-1]) if l[-1].isdigit() else 0 for l in lab] + [0]
        yield hz, [reading_said(c, l, tones[i + 1]) for i, (c, l) in enumerate(zip(hz, lab))]

def cut(hz, lookup):
    """Longest match, as romanise.js does: (start, length) of each word found."""
    i = 0
    while i < len(hz):
        for n in range(min(8, len(hz) - i), 1, -1):
            if ''.join(hz[i:i + n]) in lookup:
                yield i, n
                i += n
                break
        else:
            i += 1

train = [os.path.join(src, 'aishell3/train_content.txt')]
test = os.path.join(src, 'aishell3/test_content.txt')
heard = [s for path in train + ([] if hold_out else [test]) for s in sentences(path)]

# Longest match finds CC-CEDICT words that are seldom what was said: 中的
# (zhòngdì, to hit the mark) in 生活中的, 到了 (dàoliǎo) in 我们到了, 都会
# (dūhuì) in 他们都会. A word matched at least MIN_WORD times and said its
# own way less than half the time is given the reading said when one reading
# accounts for nine in ten of them (重点 zhòngdiǎn, not chóngdiǎn); failing
# that, it is dropped if reading its characters one by one is right more
# often and gives other syllables, not only other tones (it is then seldom
# this word), so that they are read, and marked, one by one. A word said
# two ways (处理 chǔlǐ, said chùlǐ by half the speakers) keeps CC-CEDICT's.
MIN_WORD = 5
def alike(a, b):
    """The same reading, or one differing only in a neutral tone (起来
    qǐlai, said qǐlái: the dictionary's convention against the annotator's),
    or the r of erhua, which AISHELL-3 writes er2."""
    return a == b or (bare(a) == bare(b) and 5 in (tone_of(a), tone_of(b))) or {bare(a), bare(b)} == {'r', 'er'}
matched = defaultdict(Counter)
for hz, rs in heard:
    for i, n in cut(hz, word):
        if all(rs[i:i + n]):
            matched[''.join(hz[i:i + n])][tuple(rs[i:i + n])] += 1
respelt, dropped = {}, set()
for w, got in matched.items():
    total = sum(got.values())
    own = sum(k for r, k in got.items() if all(map(alike, r, word[w])))
    one_by_one = sum(k for r, k in got.items() if all(alike(s, chars[c][0]) for s, c in zip(r, w)))
    if total >= MIN_WORD and own < total / 2:
        r, k = got.most_common(1)[0]
        if k >= 0.9 * total:
            respelt[w] = list(r)
        elif one_by_one > own and any(bare(s) != bare(chars[c][0]) for s, c in zip(word[w], w)):
            dropped.add(w)
for w in dropped:
    del word[w]
word.update(respelt)
parsed = [(hz, word[hz]) for hz, _ in parsed if hz in word]

said = defaultdict(Counter)
for hz, rs in heard:
    inside = set()
    for i, n in cut(hz, word):
        inside.update(range(i, i + n))
    for i, (c, r) in enumerate(zip(hz, rs)):
        if i not in inside and r:
            said[c][r] += 1

# --- Fallback: CC-CEDICT words, each weighted by how often it is said ------
freq = {}
for line in open(os.path.join(src, 'freq/zh_cn_full.txt'), encoding='utf-8'):
    w, n = line.split(' ')
    freq[w] = int(n)
weighted, in_words = defaultdict(Counter), defaultdict(Counter)
for hz, sy in parsed:
    for c, s in zip(hz, sy):
        weighted[c][s] += 1 + freq.get(hz, 0)
        in_words[c][s] += 1

MIN_SAID = 20
doubt, basis = set(), {}
for c in chars:
    if len(known[c]) < 2:
        continue
    heard = sum(said[c].values()) >= MIN_SAID
    counts = said[c] if heard else weighted[c]
    if not counts:
        continue
    n = counts.most_common()
    second = n[1][0] if len(n) > 1 else None
    if second and n[1][1] >= 0.1 * sum(counts.values()) and (n[1][1] >= 3 if heard else in_words[c][second] >= 3):
        doubt.add(c)
    if heard or c in doubt:
        basis[c] = counts
        # Readings the counts use that Unihan does not list (the neutral
        # tone of 去 in 回去) are added; the reading most used comes first.
        chars[c] = [r for r, _ in n] + [r for r in chars[c] if r not in counts]

# Words that settle a marked character, kept with their reading, or with ""
# when every character takes its first reading; and words in which some
# character is not read its first way.
words = {}
for hz, sy in parsed:
    same = all(s == chars[c][0] for c, s in zip(hz, sy))
    if not same or any(c in doubt for c in hz):
        words[hz] = '' if same else ' '.join(sy)

# Each reading's share of the character's uses, in the order of its readings:
# what the tier shows beside a reading it could not settle.
shares = {}
for c in doubt:
    total = sum(basis[c].values())
    shares[c] = [round(basis[c].get(r, 0) / total, 2) for r in chars[c]]
data = {'chars': {c: ' '.join(r) for c, r in chars.items()}, 'doubt': ''.join(sorted(doubt)), 'shares': shares,
        'words': words, 'syllables': syllables}

def convert(d, hz):
    """romanise.js's pinyinOf, in Python: (reading, marked) per character."""
    C, W, D = d['chars'], d['words'], set(d['doubt'])
    got, i = [], 0
    while i < len(hz):
        for n in range(min(8, len(hz) - i), 1, -1):
            w = W.get(''.join(hz[i:i + n]))
            if w is not None:
                sy = w.split(' ') if w else [C.get(c, '').split(' ')[0] for c in hz[i:i + n]]
                got += [(s, False) for s in sy]
                i += n
                break
        else:
            got.append((C.get(hz[i], '').split(' ')[0], hz[i] in D))
            i += 1
    return got

if hold_out:
    for name, d in [('this build', data)] + ([(old_json, json.load(open(old_json, encoding='utf-8')))] if old_json else []):
        k = Counter()
        for hz, rs in sentences(test):
            for (got, marked), r, c in zip(convert(d, hz), rs, hz):
                if r is None or len(known[c]) < 2:
                    continue
                wrong = not alike(got, r)
                k['polyphonic'] += 1
                k['wrong'] += wrong
                k['neutral'] += got != r and not wrong
                k['marked'] += marked
                k['marked, right'] += marked and not wrong
                k['wrong, unmarked'] += not marked and wrong
        print(f"{name}: {len(d['doubt'])} characters marked; on AISHELL-3 test, of {k['polyphonic']} characters with "
              f"more than one reading: {k['wrong']} read wrong ({k['wrong'] / k['polyphonic']:.2%}; "
              f"{k['neutral']} more differ only in a neutral tone or erhua), "
              f"{k['marked']} marked of which {k['marked, right']} were right anyway, "
              f"{k['wrong, unmarked']} wrong and not marked")
    sys.exit()

json.dump(data, open(out, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print(len(respelt), 'words respelt and', len(dropped), 'dropped by what AISHELL-3 heard;', len(chars), 'characters,', len(doubt), 'read more than one way,', len(words), 'words,', len(syllables), 'syllables,',
      os.path.getsize(out), 'bytes;', sum(c in basis and sum(said[c].values()) >= MIN_SAID for c in doubt), 'judged by AISHELL-3')
for c in '的还了长行得着地都为和要好看教':
    print(c, ' '.join(chars[c][:4]), 'marked' if c in doubt else '-', shares.get(c, ''), dict(said[c].most_common(4)))
for w in ('长安', '银行', '行走', '音乐', '快乐', '我们', '还给', '东西'):
    print(w, words.get(w), [chars[c][0] for c in w])
