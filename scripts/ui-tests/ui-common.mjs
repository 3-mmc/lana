import { createRequire } from 'module';
import fs from 'fs';
// playwright-core, wherever it is installed ($PLAYWRIGHT_DIR: a node_modules parent).
const require = createRequire((process.env.PLAYWRIGHT_DIR || process.cwd()) + '/node_modules/');
const { chromium } = require('playwright-core');
/* Downloads and screenshots go to $LANA_UI_OUT (a folder that exists). */
export const DIR = (process.env.LANA_UI_OUT || '.') + '/';
export const WEB = new URL('../../web/', import.meta.url).pathname;
export async function open(url, opts = {}) {
  const b = await chromium.launch({ executablePath: process.env.CHROMIUM || process.env.HOME + '/.cache/ms-playwright/chromium-1148/chrome-linux/chrome' });
  const ctx = await b.newContext({ acceptDownloads: true, viewport: { width: 1500, height: 1100 }, ...opts });
  const p = await ctx.newPage();
  const errors = [];
  p.on('console', m => { if (m.type() === 'error') { errors.push(m.text()); console.log('console:', m.text()); } });
  p.on('pageerror', e => { errors.push(e.message); console.log('PAGEERROR', e.message); });
  const status = () => p.$eval('#status', e => e.textContent);
  const waitStatus = async (re, ms = 180000) => {
    const t = Date.now();
    for (;;) { const s = await status(); if (re.test(s)) return s; if (Date.now() - t > ms) throw new Error('timeout; status: ' + s); await p.waitForTimeout(200); }
  };
  let nDl = 0;
  const grab = async (tag, click) => {
    const [d] = await Promise.all([p.waitForEvent('download'), click()]);
    const f = DIR + tag + '_' + (++nDl) + '_' + d.suggestedFilename(); await d.saveAs(f); return f;
  };
  await p.goto(url + (url.includes('?') ? '&' : '?') + 'r=' + Date.now());
  await waitStatus(/Praat ready|idle|no recogniser/);
  return { b, p, status, waitStatus, grab, errors };
}
/* Lane geometry on #tiers: lane index -> y, time -> x (client coords). */
export async function geom(p) {
  return p.evaluate(() => {
    const r = document.getElementById('tiers').getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height };
  });
}
export const PAD = { l: 56, r: 46, t: 6, b: 17 };
export function laneY(g, n, i) { const lh = (g.height - PAD.t - PAD.b) / n; return g.top + PAD.t + lh * (i + 0.5); }
export function timeX(g, win, t) { return g.left + PAD.l + (t - win[0]) / (win[1] - win[0]) * (g.width - PAD.l - PAD.r); }
export async function win(p) {
  const s = await p.$eval('#wininfo', e => e.textContent);
  const m = s.match(/window ([\d.]+)–([\d.]+)/);
  return [+m[1], +m[2]];
}
export const readJson = f => JSON.parse(fs.readFileSync(f, 'utf8'));
