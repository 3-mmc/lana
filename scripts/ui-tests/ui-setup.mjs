// The recording's two toggles (multiple speakers, codeswitching), language ID
// and routing, in the desktop server. Needs langid-mms-1024 and
// omnilingual-300m in the models folder, and a codeswitched recording:
// LANA_UI_MIX=<wav of Georgian then German> (see the lana-mms-lid memory).
import { open, DIR } from './ui-common.mjs';
const { b, p, status, waitStatus, grab, errors } = await open('http://127.0.0.1:47710/');
const MIX = process.env.LANA_UI_MIX;
const shown = id => p.$eval('#' + id, e => !e.hidden && e.offsetParent !== null);
const top = () => p.$eval('#mdWord', e => Math.round(e.getBoundingClientRect().top));
await p.waitForTimeout(1500);
await p.setInputFiles('#file', MIX);
console.log(await waitStatus(/analysed in/));
console.log('A. defaults: speakers', await p.isChecked('#spkOn'), '| codeswitching', await p.isChecked('#csOn'),
            '| speaker row', await shown('trkSpeaker'), '| language row', await shown('trkLang'),
            '| count field enabled', !(await p.isDisabled('#spkN')));
const y0 = await top();
await p.check('#csOn'); await p.waitForTimeout(300);
console.log('B. codeswitching on: language row', await shown('trkLang'), '| model', await p.$eval('#mdLang', e => e.value),
            '| note:', await p.$eval('#csNote', e => e.textContent));
await p.fill('#csLangs', 'Georgian, deu, xx-nonsense'); await p.waitForTimeout(200);
const y1 = await top();
console.log('   note:', await p.$eval('#csNote', e => e.textContent), '| word select moved', y1 - y0, 'px (the language row only)');
await p.fill('#csLangs', 'kat, deu'); await p.$eval('#csLangs', e => e.dispatchEvent(new Event('change')));
await p.check('#spkOn'); await p.fill('#spkN', '2'); await p.$eval('#spkN', e => e.dispatchEvent(new Event('change')));
console.log('C. speakers on: row', await shown('trkSpeaker'), '| model', await p.$eval('#mdSpeaker', e => e.value));
await p.uncheck('#spkOn');
await p.selectOption('#mdWord', 'lana::omnilingual-300m');
await p.selectOption('#mdPhone', '');
await p.click('#run');
// The queue moves on at once, so wait on it rather than on the status line.
const idle = async () => { for (;;) { const st = await p.$$eval('#queueList .qrow', rs => rs.map(r => r.dataset.s));
  if (st.length && st.every(x => !['queued', 'running'].includes(x))) return st; await p.waitForTimeout(500); } };
console.log('   jobs:', await idle());
const lana = async () => JSON.parse(await (await import('fs')).promises.readFile(await grab('setup', () => p.click('#dlLana')), 'utf8'));
const items = (d, k) => d.tiers.find(t => t.key === k)?.items || [];
console.log('D. language tier:', items(await lana(), 'lang').map(x => `${x.text}[${x.start.toFixed(2)}-${x.end.toFixed(2)} ${x.conf}${x.alts ? ' ' + JSON.stringify(x.alts) : ''}]`).join(' '),
            '| to check:', await p.$eval('#checkCount', e => e.textContent));
await p.waitForTimeout(500);
console.log('E. queue:', await p.$eval('#queueList', e => e.innerText.replace(/\s+/g, ' ')));
const d = await lana();
console.log('   word tier:', items(d, 'word').map(x => x.text).join(' ').slice(0, 400));
console.log('   source:', d.tiers.find(t => t.key === 'word').source, '| speaker tier:', items(d, 'speaker').length, 'intervals');
await p.$eval('#tiers', e => e.scrollIntoView({ block: 'center' }));
await p.screenshot({ path: DIR + 'ui-setup.png', fullPage: false });
console.log('F. saved setup:', JSON.stringify(d.setup), '| language tier in file:', d.tiers.find(t => t.key === 'lang')?.items.length);
// Reload: the toggles come back with the recording.
await p.waitForTimeout(1200);
await p.reload(); await waitStatus(/Praat ready|idle|no recogniser/);
await p.waitForTimeout(1200);
await p.setInputFiles('#file', MIX);
await waitStatus(/analysed in/);
await p.waitForTimeout(500);
console.log('G. after reload: codeswitching', await p.isChecked('#csOn'), '| languages', await p.$eval('#csLangs', e => e.value),
            '| speakers', await p.isChecked('#spkOn'), '| count', await p.$eval('#spkN', e => e.value),
            '| language intervals', items(await lana(), 'lang').length);
// A different recording starts from the defaults again.
await p.setInputFiles('#file', new URL('../../web/demo/de-vinson.wav', import.meta.url).pathname);
await waitStatus(/analysed in/);
console.log('H. another recording: codeswitching', await p.isChecked('#csOn'), '| languages', JSON.stringify(await p.$eval('#csLangs', e => e.value)));
console.log('errors:', errors.length ? errors : 'none');
await b.close();
