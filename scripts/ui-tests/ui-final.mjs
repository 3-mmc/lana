// End-to-end on ka-vinson in the desktop server, from an empty store.
import { open, DIR, WEB, geom, laneY, timeX, win, readJson } from './ui-common.mjs';
import fs from 'fs';
const { b, p, status, waitStatus, grab, errors } = await open('http://127.0.0.1:47710/');
const lana = async () => readJson(await grab('fin', () => p.click('#dlLana')));
const count = () => p.$eval('#checkCount', e => e.textContent);
const fmt = xs => xs.map(x => `${x.text}[${x.start.toFixed(3)}-${x.end.toFixed(3)} ${x.conf ?? '-'}${x.why ? ' ' + x.why : ''}${x.rv ? ' rv' : ''}]`).join(' ');
const tiersShot = async name => {
  await p.$eval('#tiers', e => e.scrollIntoView({ block: 'center' }));
  const r = await p.$eval('#tiers', e => { const q = e.closest('.pane').getBoundingClientRect(); return { x: q.left, y: q.top, width: q.width, height: q.height }; });
  await p.screenshot({ path: DIR + name, clip: r });
};
await p.waitForTimeout(1500);
await p.setInputFiles('#file', WEB + 'demo/ka-vinson.wav');
console.log(await waitStatus(/analysed in/));
await p.fill('#langFind', 'Georgian'); await p.waitForTimeout(300); await p.keyboard.press('Enter');
await p.selectOption('#mdWord', 'lana::omnilingual-300m');
await p.selectOption('#mdPhone', 'lana::zipa-small');
await p.click('#run');
console.log(await waitStatus(/omnilingual-300m finished/, 300000));
console.log(await waitStatus(/zipa-small finished/, 300000));
await p.waitForTimeout(600);
let d = await lana();
let ph = d.tiers.find(t => t.key === 'phone').items;
console.log('A. phones', ph.length, 'below 0.6:', ph.filter(x => x.conf < 0.6).length, '| words below 0.6:', d.tiers.find(t => t.key === 'word').items.filter(x => x.conf < 0.6).length, '| to check:', await count());
// zoom on the first two words
const W = d.tiers.find(t => t.key === 'word').items;
let g = await geom(p), w = await win(p);
const wb = await p.$eval('#wave', e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; });
const wx = (t, w) => wb[0] + 56 + (t - w[0]) / (w[1] - w[0]) * (wb[2] - 56 - 46);
await p.$eval('#wave', e => e.scrollIntoView({ block: 'center' }));
const wb2 = await p.$eval('#wave', e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; });
await p.mouse.move(wb2[0] + 56 + (0.7 / 10) * (wb2[2] - 102), wb2[1] + 40); await p.mouse.down();
await p.mouse.move(wb2[0] + 56 + (3.0 / 10) * (wb2[2] - 102), wb2[1] + 40, { steps: 5 }); await p.mouse.up();
await p.keyboard.press('z'); await p.waitForTimeout(1500);
await tiersShot('ui-ka-doubt-zoom.png');
await p.selectOption('#theme', 'paper'); await p.waitForTimeout(1200);
await tiersShot('ui-ka-doubt-zoom-paper.png');
await p.selectOption('#theme', 'graphite'); await p.waitForTimeout(300);
// B. triage: step, keep (1), take a runner-up (2), undo
const edH = () => p.$eval('.edbar', e => e.getBoundingClientRect().height);
const tierTop = () => p.$eval('#tiers', e => e.getBoundingClientRect().top);
const h0 = await edH(), y0 = await tierTop();
await p.keyboard.press('.'); await p.waitForTimeout(400);
console.log('   edbar height', h0, '->', await edH(), '| tiers top', y0, '->', await tierTop());
console.log('B. step 1:', await status(), '| bar:', await p.$eval('#altBar', e => e.innerText.replace(/\s+/g, ' ')));
const c0 = await count();
await p.keyboard.press('1'); await p.waitForTimeout(400);
console.log('   after 1:', c0, '->', await count(), '|', (await p.$eval('#altBar', e => e.innerText.replace(/\s+/g, ' '))));
await p.keyboard.press('2'); await p.waitForTimeout(400);
console.log('   after 2:', await count(), '|', await status());
await p.keyboard.press('Control+z'); await p.waitForTimeout(300);
await p.keyboard.press('Control+z'); await p.waitForTimeout(300);
console.log('   after 2x undo:', await count());
// C. re-time: drag two boundaries off, mistype one phone, re-time the word
d = await lana();
const W0 = d.tiers.find(t => t.key === 'word').items[0];
await p.$eval('#tiers', e => e.scrollIntoView({ block: 'center' }));
g = await geom(p); w = await win(p);
await p.mouse.click(timeX(g, w, (W0.start + W0.end) / 2), laneY(g, 2, 0));
await p.keyboard.press('z'); await p.waitForTimeout(1200);
g = await geom(p); w = await win(p);
const inW = d => d.tiers.find(t => t.key === 'phone').items.filter(x => (x.start + x.end) / 2 >= W0.start && (x.start + x.end) / 2 <= W0.end);
ph = inW(d);
console.log('C. before ', fmt(ph));
const y1 = laneY(g, 2, 1);
for (const [B, dt] of [[ph[5].end, -0.05], [ph[1].end, +0.03]]) {
  await p.mouse.move(timeX(g, w, B), y1); await p.mouse.down();
  await p.mouse.move(timeX(g, w, B + dt), y1, { steps: 6 }); await p.mouse.up(); await p.waitForTimeout(200);
}
const R = ph.find(x => x.text === 'r');
await p.mouse.click(timeX(g, w, (R.start + R.end) / 2), y1);
await p.keyboard.press('Enter'); await p.waitForTimeout(150);
await p.keyboard.insertText('m'); await p.keyboard.press('Enter'); await p.waitForTimeout(200);
d = await lana(); console.log('   edited ', fmt(inW(d)));
await p.mouse.click(timeX(g, w, (W0.start + W0.end) / 2), laneY(g, 2, 0));
await p.mouse.click(timeX(g, w, (R.start + R.end) / 2), y1, { button: 'right' });
await p.click('.ctxmenu div:has-text("Re-time from the sound")');
console.log('  ', await waitStatus(/re-timed|re-timing failed/, 120000));
d = await lana(); console.log('   retimed', fmt(inW(d)));
await tiersShot('ui-ka-retime.png');
const rows = await p.$$eval('#checkList .crow', rs => rs.map(r => r.innerText.replace(/\s+/g, ' ')));
console.log('   triage row for m:', rows.find(r => /\bm phones/.test(r)));
await p.$eval('#checkPane', e => e.scrollIntoView({ block: 'center' }));
await p.screenshot({ path: DIR + 'ui-ka-triage.png' });
const n1 = await count();
// D. reload: the autosave keeps conf, alts and the flag
await p.waitForTimeout(1500);
await p.goto('http://127.0.0.1:47710/?r=' + Date.now());
await waitStatus(/Praat ready|idle|no recogniser/); await p.waitForTimeout(800);
await p.setInputFiles('#file', WEB + 'demo/ka-vinson.wav');
await p.waitForTimeout(3500);
d = await lana();
ph = d.tiers.find(t => t.key === 'phone').items;
console.log('D. reload:', await p.$eval('#saveState', e => e.textContent), '| to check', n1, '->', await count(),
            '| conf', ph.filter(x => x.conf != null).length, 'alts', ph.filter(x => x.alts).length, 'flagged', ph.filter(x => x.why).length);
// E. a model's warning, from a Lana file, in the triage list
d.warnings = [{ start: 3.0, end: 4.5, reason: 'dropped a piece whose text repeated', tier: 'word', model: 'whisper-base' }];
fs.writeFileSync(DIR + 'fin-warn.lana.json', JSON.stringify(d));
await p.setInputFiles('#annFile', DIR + 'fin-warn.lana.json'); await p.waitForTimeout(1200);
const n2 = await count();
const wrows = await p.$$eval('#checkList .crow', rs => rs.map(r => r.innerText.replace(/\s+/g, ' ')));
const wi = wrows.findIndex(r => /whisper-base/.test(r));
console.log('E. warning row', wi, wrows[wi], '| count', n2);
await p.click(`#checkList .crow[data-i="${wi}"] .tx`); await p.waitForTimeout(400);
console.log('   bar:', await p.$eval('#altBar', e => e.innerText.replace(/\s+/g, ' ')));
await p.keyboard.press('1'); await p.waitForTimeout(400);
console.log('   after 1:', n2, '->', await count());
// F. TextGrid keeps no scores and round-trips
const tg = await grab('fin', () => p.click('#dlTextGrid'));
const tgText = fs.readFileSync(tg, 'utf8');
console.log('F. TextGrid mentions conf?', /conf|0\.\d{3}/.test(tgText.split('\n').filter(l => /text =/.test(l)).join('\n')), '| size', tgText.length);
await p.setInputFiles('#annFile', tg); await p.waitForTimeout(1200);
console.log('   imported:', await status(), '| to check', await count());
const tg2 = await grab('fin', () => p.click('#dlTextGrid'));
console.log('   round trip identical:', fs.readFileSync(tg2, 'utf8') === tgText);
console.log('errors', errors.filter(e => !/501|404/.test(e)));
await b.close();
