import { chromium } from 'playwright';
const b = await chromium.launch({ headless: true, executablePath: '/home/ape/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome', args: ['--no-sandbox','--disable-dev-shm-usage'], timeout: 30000 });
const p = await b.newPage();
await p.goto('data:text/html,<h1>hi</h1>', { timeout: 20000 });
const gl = await p.evaluate(() => { const c=document.createElement('canvas'); const g=c.getContext('webgl2'); if(!g) return 'no webgl2'; const d=g.getExtension('WEBGL_debug_renderer_info'); return d? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER); });
console.log('OK webgl:', gl);
await b.close();
