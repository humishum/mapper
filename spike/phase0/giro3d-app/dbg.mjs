import { chromium } from 'playwright';
const b = await chromium.launch({ headless:true, args:['--use-angle=gl','--use-gl=egl','--no-sandbox','--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport:{width:1280,height:720} });
p.on('console', m => console.log('[console]', m.type(), m.text().slice(0,300)));
p.on('pageerror', e => console.log('[pageerror]', String(e).slice(0,500)));
p.on('requestfailed', r => console.log('[reqfail]', r.url().slice(0,120), r.failure()?.errorText));
await p.goto('http://127.0.0.1:5180/?url=http://127.0.0.1:8123/window000.copc.laz&budget=2000000', { waitUntil:'domcontentloaded' });
await new Promise(r => setTimeout(r, 20000));
const state = await p.evaluate(() => ({
  loaded: window.__spike?.loaded, failed: window.__spike?.failed,
  marks: window.__spike?.report?.().marks, errors: window.__spike?.report?.().errors,
  renderer: window.__spike?.report?.().webglRenderer,
  snap: window.__spike?.snapshot?.(),
}));
console.log(JSON.stringify(state, null, 2));
await p.screenshot({ path: '/tmp/claude-1000/dbg.png' });
await b.close();
