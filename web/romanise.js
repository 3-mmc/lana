// SPDX-License-Identifier: GPL-3.0-or-later
/* Romanisation tiers: what the recognisers write for Chinese languages,
   written again in the systems linguists read.

   Mandarin comes out of MMS, Whisper and Omnilingual as Han characters, one
   interval each. Each character becomes a pinyin syllable, and a pinyin
   syllable becomes Wade-Giles, zhuyin (bopomofo), Gwoyeu Romatzyh, IPA or
   numbered pinyin. Cantonese (MMS yue-script_traditional, Whisper) comes out
   as Han characters too, and becomes Jyutping, and Jyutping Yale or IPA.
   A character with more than one reading takes the reading of the longest
   dictionary word it stands in (长 in 长安 is cháng; 行 in 銀行 is hong4);
   when no word settles it, it takes the reading said most often where no
   word settles it in a transcribed corpus of speech (AISHELL-3 for Mandarin,
   HKCanCor for Cantonese) and, if it is one of the characters really said
   more than one way there, is marked with the others offered, as an ARPABET
   tier marks a flap: the conversion does not guess. Tones are citation
   tones; third-tone sandhi, the tones of 一 and 不 and Cantonese changed
   tones are not applied, since which applies depends on the prosody, which
   is the annotator's to judge.

   Min Nan comes out of MMS as Pe̍h-ōe-jī (POJ) and is rewritten as Tâi-lô.

   The data is web/romanise.json (scripts/languages/romanise.py) for
   Mandarin and web/jyutping.json (scripts/languages/jyutping.py) for
   Cantonese, each fetched the first time a tier needs it. */

let DATA = null, YUE = null;
async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('the romanisation data could not be loaded');
  return r.json();
}
export async function load(base = '') {
  if (!DATA) DATA = await fetchJson(base + 'romanise.json');
  return DATA;
}
export async function loadCantonese(base = '') {
  if (!YUE) YUE = await fetchJson(base + 'jyutping.json');
  return YUE;
}
/* For tests: data given directly. */
export function useData(d, yue = null) { DATA = d; YUE = yue; }

const HAN = /[㐀-䶿一-鿿豈-﫿]|[\ud840-\ud87f][\udc00-\udfff]/;
export const isHan = s => HAN.test(s);

/* Readings for a run of Han characters, as given (one per interval). Each
   item: { text: syllable or the character itself when unknown, alts: [other
   readings], doubt: true when the character has several readings and no word
   settled which, shares: for those, each reading's share of the character's
   uses outside words, text first, then alts }. Longest-word match, left to
   right. */
export const pinyinOf = chars => readingsOf(DATA, chars);
export const jyutpingOf = chars => readingsOf(YUE, chars);
function readingsOf(D, chars) {
  const { chars: C, words: W } = D;
  const out = [];
  let i = 0;
  while (i < chars.length) {
    let took = 0;
    for (let n = Math.min(8, chars.length - i); n >= 2; n--) {
      const w = W[chars.slice(i, i + n).join('')];
      if (w !== undefined) {
        const sy = w ? w.split(' ') : chars.slice(i, i + n).map(c => (C[c] || '').split(' ')[0]);
        for (let k = 0; k < n; k++) {
          const all = (C[chars[i + k]] || '').split(' ').filter(Boolean);
          out.push({ text: sy[k], alts: all.filter(r => r !== sy[k]), doubt: false });
        }
        took = n;
        break;
      }
    }
    if (took) { i += took; continue; }
    const all = (C[chars[i]] || '').split(' ').filter(Boolean);
    const doubt = all.length > 1 && D.doubt.includes(chars[i]);
    out.push(all.length ? { text: all[0], alts: all.slice(1), doubt, ...(doubt ? { shares: D.shares[chars[i]] } : {}) }
                        : { text: chars[i], alts: [], doubt: false });
    i++;
  }
  return out;
}

/* A toned pinyin syllable as its toneless letters and its tone (1-4, 5 for
   the neutral tone). */
const MARKS = { '̄': 1, '́': 2, '̌': 3, '̀': 4 };
export function untone(syl) {
  let tone = 5, base = '';
  for (const ch of syl.normalize('NFD').toLowerCase()) {
    if (MARKS[ch]) tone = MARKS[ch];
    else base += ch;
  }
  base = base.normalize('NFC').replace(/v/g, 'ü');
  const m = base.match(/^([a-zü]+)([1-5])$/);
  if (m) return { base: m[1], tone: +m[2] };
  return { base, tone };
}

export function numbered(syl) {
  const { base, tone } = untone(syl);
  return base + tone;
}

const ZHUYIN_TONE = { 1: '', 2: 'ˊ', 3: 'ˇ', 4: 'ˋ', 5: '' };
export function zhuyin(syl) {
  const { base, tone } = untone(syl);
  const row = DATA.syllables[base];
  if (!row) return null;
  return (tone === 5 ? '˙' : '') + row[0] + ZHUYIN_TONE[tone];
}

/* Chao tone letters for the citation tones; tone 3 is the full dipping
   contour, which in running speech is mostly ˨˩. */
const CHAO = { 1: '˥', 2: '˧˥', 3: '˨˩˦', 4: '˥˩', 5: '' };
export function ipa(syl) {
  const { base, tone } = untone(syl);
  const row = DATA.syllables[base];
  if (!row) return null;
  return row[1] + CHAO[tone];
}

/* Wade-Giles, with the tone as a superscript number. */
const WG_INITIAL = { b: 'p', p: "p'", m: 'm', f: 'f', d: 't', t: "t'", n: 'n', l: 'l', g: 'k', k: "k'", h: 'h',
  j: 'ch', q: "ch'", x: 'hs', zh: 'ch', ch: "ch'", sh: 'sh', r: 'j', z: 'ts', c: "ts'", s: 's' };
const WG_WHOLE = { zhi: 'chih', chi: "ch'ih", shi: 'shih', ri: 'jih', zi: 'tzu', ci: "tz'u", si: 'ssu', er: 'êrh',
  e: 'ê', ge: 'ko', ke: "k'o", he: 'ho', yi: 'i', you: 'yu', yu: 'yü', yue: 'yüeh', yuan: 'yüan', yun: 'yün',
  yan: 'yen', ye: 'yeh', yong: 'yung', wo: 'wo', weng: 'wêng', wen: 'wên', eng: 'êng', en: 'ên' };
const WG_TONE = { 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '' };
export function wadeGiles(syl) {
  const { base, tone } = untone(syl);
  if (!DATA.syllables[base]) return null;
  let wg = WG_WHOLE[base];
  if (!wg) {
    const ini = (base.match(/^(zh|ch|sh|[bpmfdtnlgkhjqxrzcs])/) || [''])[0];
    let fin = base.slice(ini.length);
    if ('jqx'.includes(ini) && ini) fin = fin.replace(/^u/, 'ü');
    if (fin === 'uo' && !['g', 'k', 'h', 'sh'].includes(ini)) fin = 'o';
    else if (fin === 'ui' && (ini === 'g' || ini === 'k')) fin = 'uei';
    else if (fin === 'e') fin = 'ê';
    else fin = fin.replace(/^ong$/, 'ung').replace(/^iong$/, 'iung').replace(/^ian$/, 'ien')
      .replace(/^ie$/, 'ieh').replace(/^üe$/, 'üeh').replace(/^ue$/, 'üeh')
      .replace(/^en$/, 'ên').replace(/^eng$/, 'êng');
    wg = (WG_INITIAL[ini] || '') + fin;
  }
  return wg + WG_TONE[tone];
}

/* Gwoyeu Romatzyh (National Romanisation, 1928), which spells the tone in
   the letters. The basic form is the first tone; the others change it:
   2nd adds r after the vowel (char), or turns a medial i/u into y/w (chyan,
   gwo; yi, wu, yn, yng where i or u is the vowel); 3rd doubles the vowel
   (maa), or turns a medial i/u into e/o (sheau, goang; but jiee, guoo) and
   ai/au into ae/ao; 4th turns endings i/u/n/ng/l into y/w/nn/nq/ll, or adds
   h. With m, n, l, r the basic form is the 2nd tone, and the 1st puts h
   after them (mha, lha). Without an initial, i/u are written y/w in the 3rd
   and 4th tones (yii, yeau, woo; yih, yaw, wuh). A neutral tone is the
   basic form after a dot (.de): which tone the syllable had before it was
   weakened, which GR would write there, is not in the pinyin. The r of
   erhua as a syllable of its own is GR's suffix l. Checked against
   Wiktionary's converter by scripts/languages/test-romanise.mjs. */
const GR_INITIAL = { b: 'b', p: 'p', m: 'm', f: 'f', d: 'd', t: 't', n: 'n', l: 'l', g: 'g', k: 'k', h: 'h',
  j: 'j', q: 'ch', x: 'sh', zh: 'j', ch: 'ch', sh: 'sh', r: 'r', z: 'tz', c: 'ts', s: 's' };
function grBasic(base) {
  if (base === 'er') return ['', 'el'];
  let ini = '', fin = base;
  if (/^y/.test(base)) fin = /^yi/.test(base) ? base.slice(1) : 'i' + base.slice(1);
  else if (/^w/.test(base)) fin = base === 'wu' ? 'u' : 'u' + base.slice(1);
  else {
    ini = (base.match(/^(zh|ch|sh|[bpmfdtnlgkhjqxrzcs])/) || [''])[0];
    fin = base.slice(ini.length);
    if (/^[jqx]$/.test(ini)) fin = fin.replace(/^u/, 'ü');
    if (fin === 'i' && /^(zh|ch|sh|r|z|c|s)$/.test(ini)) fin = 'y';
    fin = fin.replace(/^iu$/, 'iou').replace(/^ui$/, 'uei').replace(/^un$/, 'uen');
  }
  fin = fin.replace(/^ü/, 'iu').replace(/^(i?)ao$/, '$1au');
  return [GR_INITIAL[ini] ?? ini, fin];
}
export function gwoyeu(syl) {
  const { base, tone } = untone(syl);
  if (base === 'r') return 'l';
  if (!DATA.syllables[base]) return null;
  const [ini, fin] = grBasic(base);
  const sonorant = /^[mnlr]$/.test(ini);
  const medial = /^[iu][aeiou]/.test(fin);  // iu is ü, so iu itself counts too
  const lone = !ini;
  const out = s => ini + s;
  if (tone === 5) return '.' + ini + fin;
  if (tone === 1) return sonorant ? ini + 'h' + fin : out(fin);
  if (tone === 2) {
    if (sonorant) return out(fin);
    if (/^[iu]/.test(fin)) {
      const s = (fin[0] === 'i' ? 'y' : 'w') + fin.slice(1);
      return out(s === 'y' ? 'yi' : s === 'w' ? 'wu' : s);
    }
    return out(fin.replace(/^([aeiouy]+)/, '$1r'));
  }
  if (tone === 3) {
    let s;
    if (/^(i|in|ing|u)$/.test(fin)) s = fin[0] + fin;
    else if (fin === 'ie' || fin === 'uo') s = lone ? fin.slice(1) + fin.slice(1) : fin + fin.slice(-1);
    else if (medial) s =(fin[0] === 'i' ? 'e' : 'o') + fin.slice(1);
    else if (fin === 'ai' || fin === 'au') s = fin[0] + (fin[1] === 'i' ? 'e' : 'o');
    else s = fin.replace(/^([aeiouy])/, '$1$1');
    if (lone && /^[iu]/.test(fin)) s = (fin[0] === 'i' ? 'y' : 'w') + s;
    return out(s);
  }
  let s = fin;
  if (/[aeo][iu]$/.test(s)) s = s.slice(0, -1) + (s.endsWith('i') ? 'y' : 'w');
  else if (/ng$/.test(s)) s = s.slice(0, -2) + 'nq';
  else if (/n$/.test(s)) s += 'n';
  else if (/l$/.test(s)) s += 'l';
  else s += 'h';
  if (lone) s = /^[iu][aeiou]/.test(s) ? (s[0] === 'i' ? 'y' : 'w') + s.slice(1) : s.replace(/^i/, 'yi').replace(/^u/, 'wu');
  return out(s);
}

/* Pe̍h-ōe-jī -> Tâi-lô, syllable by syllable (syllables are joined by
   hyphens or spaces). The letters change (ch -> ts, chh -> tsh, o͘ -> oo,
   oa -> ua, oe -> ue, ek -> ik, eng -> ing, ⁿ -> nn) and the tone mark moves
   to the vowel Tâi-lô puts it on: a, then oo or o, then e, then of i and u
   whichever comes second; a syllabic m or ng takes it on the m or n. */
const TL_MARKS = /[̀́̂̄̆̋̌̍]/g;
export function taiLo(poj) {
  return poj.normalize('NFC').split(/([\s-]+)/).map(s => /^[\s-]*$/.test(s) ? s : tlSyllable(s)).join('');
}
function tlSyllable(s) {
  const d = s.normalize('NFD');
  const mark = (d.match(TL_MARKS) || [''])[0];
  const cap = /^[A-Z]/.test(d);
  let b = d.replace(TL_MARKS, '').toLowerCase()
    .replace(/o͘/g, 'oo').replace(/ⁿ/g, 'nn')
    .replace(/^chh/, 'tsh').replace(/^ch/, 'ts')
    .replace(/oa/g, 'ua').replace(/oe/g, 'ue')
    .replace(/ek$/, 'ik').replace(/eng$/, 'ing');
  if (mark) {
    let at = -1;
    for (const v of ['a', 'oo', 'o', 'e']) { at = b.indexOf(v); if (at >= 0) break; }
    if (at < 0) {
      const iu = b.search(/[iu][iu]/);
      at = iu >= 0 ? iu + 1 : b.search(/[iu]/);
    }
    if (at < 0) at = b.search(/[mn]/);
    if (at >= 0) b = b.slice(0, at + 1) + mark + b.slice(at + 1);
  }
  b = b.normalize('NFC');
  return cap ? b[0].toUpperCase() + b.slice(1) : b;
}

/* Cantonese Yale from Jyutping, as Wiktionary's converter writes it: aa as a
   when it is the whole final, oe and eo as eu, j as y (jyu as yu), z as j, c
   as ch; the low tones (4-6) add h after the vowels, and the tone mark
   (1 macron, 2 and 5 acute, 4 grave) goes on the first vowel, or on the m
   or n of a syllabic m or ng. Tone 1 takes the macron of the high level
   tone, which is how Hong Kong speech has it; Yale's grave for a high
   falling first tone is not used. Finals Yale has no spelling for (a alone,
   em, eu, oen ...) give null. */
const YALE_FINAL = { aa: 'a', oe: 'eu', oeng: 'eung', oek: 'euk', eoi: 'eui', eon: 'eun', eot: 'eut' };
const YALE_NONE = /^(a|eu|em|en|ep|et|om|op|um|up|oem|oen|oep|oet)$/;
const YALE_MARK = { 1: '̄', 2: '́', 4: '̀', 5: '́' };
export function yale(syl) {
  const m = /^([a-z]+)([1-6])$/.exec(syl);
  if (!m) return null;
  const t = +m[2], s = m[1].replace(/^jy?/, 'y').replace(/^z/, 'j').replace(/^c/, 'ch');
  const v = s.search(/[aeiou]/);
  let ini, a, b = '', c = '';
  if (v < 0) {
    const q = /^(h?)(m|ng)$/.exec(s);
    if (!q) return null;
    [, ini, a] = q;
  } else {
    ini = s.slice(0, v);
    let fin = s.slice(v);
    if (YALE_NONE.test(fin)) return null;
    const q = /^([aeiou])([aiu]*)([mnptk]?g?)$/.exec(YALE_FINAL[fin] || fin);
    if (!q) return null;
    [, a, b, c] = q;
  }
  return (ini + a[0] + (YALE_MARK[t] || '') + a.slice(1) + b + (t >= 4 ? 'h' : '') + c).normalize('NFC');
}

/* Cantonese IPA from Jyutping (Hong Kong pronunciation, after Wiktionary's
   converter: ei, eoi, ing, ik, ou, ung, uk have their closer vowels), with
   Chao tone letters as for Mandarin: 1 ˥, 2 ˧˥, 3 ˧, 4 ˨˩, 5 ˩˧, 6 ˨; in a
   syllable ending in p, t or k, tones 1, 3 and 6 are short, with the same
   letters. */
const YUE_INITIAL = { b: 'p', p: 'pʰ', m: 'm', f: 'f', d: 't', t: 'tʰ', n: 'n', l: 'l', g: 'k', k: 'kʰ', ng: 'ŋ',
  gw: 'kʷ', kw: 'kʷʰ', z: 't͡s', c: 't͡sʰ', s: 's', h: 'h', w: 'w', j: 'j', '': '' };
const YUE_NUCLEUS = { aa: 'aː', a: 'ɐ', e: 'ɛː', i: 'iː', o: 'ɔː', oe: 'œː', eo: 'ɵ', u: 'uː', yu: 'yː' };
const YUE_CODA = { i: 'i̯', u: 'u̯', y: 'y̯', m: 'm', n: 'n', ng: 'ŋ', p: 'p̚', t: 't̚', k: 'k̚', '': '' };
const YUE_CLOSE = { ei: 'ei̯', eoi: 'ɵy̯', ing: 'ɪŋ', ik: 'ɪk̚', ou: 'ou̯', ung: 'ʊŋ', uk: 'ʊk̚' };
const YUE_TONE = { 1: '˥', 2: '˧˥', 3: '˧', 4: '˨˩', 5: '˩˧', 6: '˨' };
export function cantoneseIpa(syl) {
  const m = /^(gw|kw|ng|[bpmfdtnlgkzcshwj]?)([a-z]*)([1-6])$/.exec(syl);
  if (!m) return null;
  const [, ini, fin, t] = m;
  const syllabic = /^(h?)(m|ng)$/.exec(ini + fin);
  if (syllabic) return YUE_INITIAL[syllabic[1]] + (syllabic[2] === 'm' ? 'm̩' : 'ŋ̍') + YUE_TONE[t];
  let body = YUE_CLOSE[fin];
  if (!body) {
    const q = /^(yu|aa|oe|eo|[aeiou])(ng|[iuymnptk]?)$/.exec(fin);
    if (!q) return null;
    body = YUE_NUCLEUS[q[1]] + YUE_CODA[q[2]];
  }
  return YUE_INITIAL[ini] + body + YUE_TONE[t];
}

/* The systems a tier can be written in, by what it is made from (the page
   lists them again, to build its menu before this module is loaded). */
export const SYSTEMS = {
  han: [['pinyin', 'pinyin'], ['pinyin-numbers', 'pinyin with tone numbers'], ['wade-giles', 'Wade-Giles'],
        ['zhuyin', 'zhuyin (bopomofo)'], ['gwoyeu', 'Gwoyeu Romatzyh'], ['ipa', 'IPA, citation tones']],
  yue: [['jyutping', 'Jyutping'], ['yale', 'Yale'], ['ipa-yue', 'IPA']],
  poj: [['tai-lo', 'Tâi-lô']],
};
export const cantonese = system => SYSTEMS.yue.some(([s]) => s === system);

/* One syllable of pinyin in another system, or null where it has no
   counterpart. */
export function fromPinyin(syl, system) {
  switch (system) {
    case 'pinyin': return syl;
    case 'pinyin-numbers': return numbered(syl);
    case 'wade-giles': return wadeGiles(syl);
    case 'zhuyin': return zhuyin(syl);
    case 'gwoyeu': return gwoyeu(syl);
    case 'ipa': return ipa(syl);
  }
  return null;
}

/* One syllable of Jyutping in another system, or null. */
export function fromJyutping(syl, system) {
  switch (system) {
    case 'jyutping': return syl;
    case 'yale': return yale(syl);
    case 'ipa-yue': return cantoneseIpa(syl);
  }
  return null;
}
