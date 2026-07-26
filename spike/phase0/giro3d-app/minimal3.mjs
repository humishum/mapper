import { chromium } from 'playwright';
const tries = [
  { label: 'channel:chromium', opts: { channel: 'chromium' } },
  { label: 'execPath:1194-full', opts: { executablePath: '/home/ape/.cache/ms-playwright/chromium-1194/chrome-linux/chrome' } },
  { label: 'execPath:1228', opts: { executablePath: '/home/ape/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome' } },
];
for (const t of tries) {
  try {
    const b = await chromium.launch({ headless: true, args: ['--no-sandbox','--disable-dev-shm-usage'], timeout: 25000, ...t.opts });
    const p = await b.newPage();
    await p.goto('data:text/html,<h1>hi</h1>');
    const gl = await p.evaluate(() => { const c=document.createElement('canvas'); const g=c.getContext('webgl2'); if(!g) return 'no webgl2'; const d=g.getExtension('WEBGL_debug_renderer_info'); return d? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER); });
    console.log(t.label, '=> OK, webgl:', gl);
    await b.close();
  } catch (e) { console.log(t.label, '=> FAIL:', String(e).split('\n')[0].slice(0,120)); }
}
