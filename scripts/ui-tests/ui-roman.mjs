// Romanisation tiers without a recogniser: a TextGrid with a Mandarin word
// tier and a Cantonese tier is imported over a demo clip, and the tier menu's
// Mandarin and Cantonese systems are made into tiers. Needs only a server for
// the page (web/serve.py or the desktop one); the address is given as LANA_URL.
import { open, DIR, WEB, readJson } from './ui-common.mjs';
import fs from 'fs';
const { b, p, status, waitStatus, grab, errors } = await open(process.env.LANA_URL || 'http://127.0.0.1:47710/');
await p.setInputFiles('#file', WEB + 'demo/ka-vinson.wav');
console.log(await waitStatus(/analysed in|ready/));
const tier = (name, chars) => {
  const step = 15 / chars.length;
  return `    item [] :\n        class = "IntervalTier"\n        name = "${name}"\n        xmin = 0\n        xmax = 15\n        intervals: size = ${chars.length}\n` +
    chars.map((c, i) => `        intervals [${i + 1}]:\n            xmin = ${(i * step).toFixed(3)}\n            xmax = ${((i + 1) * step).toFixed(3)}\n            text = "${c}"\n`).join('');
};
const mandarin = [...'我们到了银行还是很长的'], cantonese = [...'佢哋喺銀行度行咗好耐為咗'];
const tg = `File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 15\ntiers? <exists>\nsize = 2\nitem []:\n` +
  tier('words', mandarin) + tier('Cantonese', cantonese);
fs.writeFileSync(DIR + 'roman.TextGrid', tg);
await p.setInputFiles('#annFile', DIR + 'roman.TextGrid'); await p.waitForTimeout(1200);
const menu = async () => {
  await p.click('#tierAdd'); await p.waitForTimeout(300);
  const items = await p.$$eval('.ctxmenu div', es => es.map(e => e.textContent));
  await p.keyboard.press('Escape'); await p.waitForTimeout(200);
  return items.filter(t => / from /.test(t));
};
const make = async label => {
  const before = await status();
  await p.click('#tierAdd'); await p.waitForTimeout(300);
  await p.locator('.ctxmenu div', { hasText: label }).first().click();
  // The first Cantonese tier waits for jyutping.json, so wait for a new status.
  for (let t = 0; t < 150 && await status() === before; t++) await p.waitForTimeout(200);
  console.log(label, '->', await waitStatus(/intervals from|romanisation:/, 30000));
};
console.log('menu, no language:', (await menu()).join(' | '));
for (const label of ['Gwoyeu Romatzyh from words', 'Pinyin from words', 'Jyutping (Cantonese) from Cantonese', 'Yale (Cantonese) from Cantonese', 'IPA, Cantonese from Cantonese'])
  await make(label);
await p.fill('#langFind', 'Cantonese'); await p.waitForTimeout(300); await p.keyboard.press('Enter'); await p.waitForTimeout(300);
console.log('menu, Cantonese:', (await menu()).join(' | '));
const d = readJson(await grab('roman', () => p.click('#dlLana')));
for (const t of [...(d.tiers || []), ...(d.extra || [])].filter(t => t.items?.length))
  console.log((t.label || t.key) + ':', t.items.map(x => x.text + (x.why === 'reading' ? '?' + JSON.stringify(x.alts) : '')).join(' '));
console.log('to check:', await p.$eval('#checkCount', e => e.textContent));
console.log('errors', errors.filter(e => !/404/.test(e)));
await b.close();
