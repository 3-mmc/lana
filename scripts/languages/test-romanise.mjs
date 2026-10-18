// Checks web/romanise.js against its data and reference tables:
//   node scripts/languages/test-romanise.mjs
// - Gwoyeu Romatzyh of every syllable in every tone against Wiktionary's
//   converter (gwoyeu-reference.tsv, see its header);
// - Yale and Cantonese IPA of every Jyutping syllable in every tone against
//   Wiktionary's converter (cantonese-reference.tsv);
// - Han to pinyin and to Jyutping on characters and words whose reading the
//   data was built to settle, and the marks left on the ones it cannot.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as R from '../../web/romanise.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.join(here, '../../web');
R.useData(JSON.parse(fs.readFileSync(path.join(web, 'romanise.json'), 'utf8')),
          JSON.parse(fs.readFileSync(path.join(web, 'jyutping.json'), 'utf8')));
let failed = 0, passed = 0;
const check = (what, got, want) => {
  if (JSON.stringify(got) === JSON.stringify(want)) passed++;
  else { failed++; if (failed <= 40) console.log(`FAIL ${what}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); }
};

// Gwoyeu Romatzyh, every syllable x tone.
const ref = fs.readFileSync(path.join(here, 'gwoyeu-reference.tsv'), 'utf8').split('\n').filter(l => l && !l.startsWith('#'));
for (const line of ref) {
  const [py, gr] = line.split('\t');
  check('GR ' + py, R.gwoyeu(py), gr);
}
console.log(`Gwoyeu Romatzyh: ${ref.length} syllables and tones checked`);
// Words as published in GR, from marked pinyin.
for (const [py, gr] of [['guó yǔ luó mǎ zì', 'gwo yeu luo maa tzyh'], ['zhōng wén', 'jong wen'],
  ['běi jīng', 'beei jing'], ['yǒu', 'yeou'], ['ér', 'erl'], ['nǚ', 'neu'], ['xuě', 'sheue'], ['de', '.de']])
  check('GR ' + py, py.split(' ').map(R.gwoyeu).join(' '), gr);

// Han to pinyin: words settle readings; a character alone takes its usual
// reading and is marked only when it is really read more than one way.
const py = s => R.pinyinOf([...s]).map(x => x.text + (x.doubt ? '?' : '')).join(' ');
for (const [hz, want] of [
  ['长安', 'cháng ān'], ['银行', 'yín háng'], ['还给', 'huán gěi'], ['还是', 'hái shi'], ['东西', 'dōng xi'],
  ['可能', 'kě néng'], ['第二', 'dì èr'], ['女儿', 'nǚ ér'],      // apostrophes in CC-CEDICT's pinyin
  ['生活中的', 'shēng huó zhōng de'], ['我们到了', 'wǒ men dào le'], ['他们都会', 'tā men dōu huì'],
  ['的', 'de'], ['还', 'hái'], ['了', 'le'], ['着', 'zhe'], ['都', 'dōu'],
  ['长', 'cháng?'], ['行', 'xíng?'], ['得', 'dé?'], ['为', 'wéi?']])
  check('pinyin ' + hz, py(hz), want);

// Cantonese: Yale and IPA of every syllable x tone against Wiktionary's
// converter, its tone numbers as Lana's Chao letters.
const CHAO = { '⁵⁵': '˥', '³⁵': '˧˥', '³³': '˧', '²¹': '˨˩', '¹³': '˩˧', '²²': '˨', '⁵': '˥', '³': '˧', '²': '˨' };
const yref = fs.readFileSync(path.join(here, 'cantonese-reference.tsv'), 'utf8').split('\n').filter(l => l && !l.startsWith('#'));
for (const line of yref) {
  const [jp, yale, ipa] = line.split('\t');
  check('Yale ' + jp, R.yale(jp)?.normalize('NFC') ?? 'false', yale.normalize('NFC'));
  check('Cantonese IPA ' + jp, R.cantoneseIpa(jp), ipa.replace(/[¹²³⁵]+$/, d => CHAO[d]));
}
console.log(`Yale and Cantonese IPA: ${yref.length} syllables and tones checked`);
const jp = s => R.jyutpingOf([...s]).map(x => x.text + (x.doubt ? '?' : '')).join(' ');
for (const [hz, want] of [['銀行', 'ngan4 hong4'], ['行路', 'haang4 lou6'], ['長大', 'zoeng2 daai6'], ['音樂', 'jam1 ngok6'],
  ['快樂', 'faai3 lok6'], ['會唔會', 'wui5 m4 wui5'], ['佢', 'keoi5'], ['嘅', 'ge3'], ['為', 'wai6?']])
  check('Jyutping ' + hz, jp(hz), want);

// Other systems from pinyin.
check('Wade-Giles', ['zhōng', 'guó', 'xī', 'rì', 'zì'].map(R.wadeGiles), ['chung¹', 'kuo²', 'hsi¹', 'jih⁴', 'tzu⁴']);
check('zhuyin', R.zhuyin('zhōng'), 'ㄓㄨㄥ');

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
