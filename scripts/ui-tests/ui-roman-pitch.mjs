// Romanisation tiers and the CREPE track, on a FLEURS Mandarin clip read by
// Omnilingual. Needs omnilingual-300m and crepe-tiny in the models folder;
// the clip's path is given as CLIP.
import { open, DIR, readJson } from './ui-common.mjs';
import fs from 'fs';
const { b, p, status, waitStatus, grab, errors } = await open('http://127.0.0.1:47710/');
await p.waitForTimeout(1500);
await p.setInputFiles('#file', process.env.CLIP);
console.log(await waitStatus(/analysed in/));
await p.fill('#langFind', 'Mandarin'); await p.waitForTimeout(300); await p.keyboard.press('Enter');
await p.selectOption('#mdWord', 'lana::omnilingual-300m');
await p.selectOption('#mdPhone', '');
await p.click('#run');
console.log(await waitStatus(/omnilingual-300m finished/, 300000));
const lana = async () => readJson(await grab('rom', () => p.click('#dlLana')));
let d = await lana();
console.log('words:', d.tiers.find(t => t.key === 'word').items.map(x => x.text).join(''));
for (const sys of ['Pinyin from', 'Wade-Giles from', 'Zhuyin (bopomofo) from', 'IPA, citation tones from']) {
  await p.click('#tierAdd'); await p.waitForTimeout(300);
  const item = p.locator('.ctxmenu div', { hasText: sys }).first();
  if (!await item.count()) { console.log('no menu item', sys, await p.$$eval('.ctxmenu div', es => es.map(e => e.textContent))); continue; }
  await item.click();
  console.log(sys, '->', await waitStatus(/intervals from/, 30000));
}
d = await lana();
for (const t of [...(d.tiers || []), ...(d.extra || [])].filter(t => t.items?.length))
  console.log((t.label || t.key) + ':', t.items.map(x => x.text + (x.why === 'reading' ? '?' : '')).join(' '));
console.log('keys:', Object.keys(d));
console.log('to check:', await p.$eval('#checkCount', e => e.textContent));
await p.keyboard.press('.'); await p.waitForTimeout(400);
console.log('step:', await status());
console.log('bar:', await p.$eval('#altBar', e => e.innerText.replace(/\s+/g, ' ')));
// CREPE
await p.check('#showCrepe');
console.log(await waitStatus(/dashed beside|failed|needs a pitch/, 120000));
const csv = fs.readFileSync(await grab('trk', () => p.$eval('#mTracks', e => e.click())), 'utf8').split('\n');
console.log('csv header:', csv[0]);
const rows = csv.slice(1).filter(Boolean).map(r => r.split(','));
const both = rows.filter(r => r[1] && r[r.length - 2]);
const ratio = both.map(r => +r[r.length - 2] / +r[1]).sort((a, b) => a - b);
console.log('frames with both f0:', both.length, 'of', rows.length, '| CREPE/Praat median ratio', ratio[ratio.length >> 1]?.toFixed(3));
await p.$eval('#spec', e => e.scrollIntoView({ block: 'center' }));
const r = await p.$eval('#spec', e => { const q = e.closest('.pane').getBoundingClientRect(); return { x: q.left, y: q.top, width: q.width, height: q.height }; });
await p.screenshot({ path: DIR + 'ui-crepe.png', clip: r });
console.log('errors', errors.filter(e => !/404/.test(e)));
await b.close();
