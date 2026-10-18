// SPDX-License-Identifier: GPL-3.0-or-later
/* Romanisation tiers: what the recognisers write for Chinese languages,
   written again in the systems linguists read.

   Mandarin comes out of MMS, Whisper and Omnilingual as Han characters, one
   interval each. Each character becomes a pinyin syllable, and a pinyin
   syllable becomes Wade-Giles, zhuyin (bopomofo), IPA or numbered pinyin.
   A character with more than one reading takes the reading of the longest
   CC-CEDICT word it stands in (长 in 长安 is cháng); when no word settles it,
   it keeps its usual reading and, if it is one of the characters really read
   more than one way (by their use in CC-CEDICT), is marked with the others
   offered, as an ARPABET tier marks a flap: the conversion does not guess. Tones are
   citation tones; third-tone sandhi and the tones of 一 and 不 are not
   applied, since which applies depends on the prosody, which is the
   annotator's to judge.

   Min Nan comes out of MMS as Pe̍h-ōe-jī (POJ) and is rewritten as Tâi-lô.

   The data is web/romanise.json (scripts/languages/romanise.py), fetched the
   first time a romanisation tier is made. */

let DATA = null;
export async function load(base = '') {
  if (!DATA) {
    const r = await fetch(base + 'romanise.json');
    if (!r.ok) throw new Error('the romanisation data could not be loaded');
    DATA = await r.json();
  }
  return DATA;
}
/* For tests: data given directly. */
export function useData(d) { DATA = d; }

const HAN = /[㐀-䶿一-鿿豈-﫿]|[\ud840-\ud87f][\udc00-\udfff]/;
export const isHan = s => HAN.test(s);

/* Pinyin for a run of Han characters, as given (one per interval). Each item:
   { text: syllable or the character itself when unknown, alts: [other
   readings], doubt: true when the character has several readings and no word
   settled which, shares: for those, each reading's share of the character's
   uses in CC-CEDICT, text first, then alts }. Longest-word match, left to right. */
export function pinyinOf(chars) {
  const { chars: C, words: W } = DATA;
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
    const doubt = DATA.doubt.includes(chars[i]);
    out.push(all.length ? { text: all[0], alts: all.slice(1), doubt, ...(doubt ? { shares: DATA.shares[chars[i]] } : {}) }
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

/* The systems a tier can be written in, by what it is made from. */
export const SYSTEMS = {
  han: [['pinyin', 'pinyin'], ['pinyin-numbers', 'pinyin with tone numbers'], ['wade-giles', 'Wade-Giles'],
        ['zhuyin', 'zhuyin (bopomofo)'], ['ipa', 'IPA (citation tones)']],
  poj: [['tai-lo', 'Tâi-lô']],
};

/* One syllable of pinyin in another system, or null where it has no
   counterpart. */
export function fromPinyin(syl, system) {
  switch (system) {
    case 'pinyin': return syl;
    case 'pinyin-numbers': return numbered(syl);
    case 'wade-giles': return wadeGiles(syl);
    case 'zhuyin': return zhuyin(syl);
    case 'ipa': return ipa(syl);
  }
  return null;
}
