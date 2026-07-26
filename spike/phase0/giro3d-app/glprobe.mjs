import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
const CH='/home/ape/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome';
const modes = {
  swiftshader: ['--use-angle=swiftshader','--enable-unsafe-swiftshader'],
  'angle-gl': ['--use-angle=gl','--enable-gpu','--ignore-gpu-blocklist'],
  'angle-vulkan': ['--use-angle=vulkan','--enable-features=Vulkan','--enable-gpu','--ignore-gpu-blocklist'],
  default: [],
};
let port = 9400;
for (const [name, flags] of Object.entries(modes)) {
  port++;
  const child = spawn(CH, ['--headless=new',`--remote-debugging-port=${port}`,`--user-data-dir=/tmp/claude-1000/glprobe-${port}`,'--no-sandbox','--disable-dev-shm-usage',...flags], {stdio:['ignore','ignore','ignore']});
  try {
    for (let i=0;i<60;i++){ try { const r=await fetch(`http://127.0.0.1:${port}/json/version`); if(r.ok) break; } catch{} await new Promise(r=>setTimeout(r,250)); }
    const b = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
    const ctx = await b.newContext();
    const p = await ctx.newPage();
    await p.goto('data:text/html,<canvas id=c></canvas>', {timeout:15000});
    const info = await p.evaluate(() => {
      const c=document.createElement('canvas'); const g=c.getContext('webgl2');
      if(!g) return {webgl2:false};
      const d=g.getExtension('WEBGL_debug_renderer_info');
      return {webgl2:true, renderer: d?g.getParameter(d.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER), vendor: d?g.getParameter(d.UNMASKED_VENDOR_WEBGL):null, maxTex:g.getParameter(g.MAX_TEXTURE_SIZE)};
    });
    console.log(name, '=>', JSON.stringify(info));
    await b.close();
  } catch(e) { console.log(name, '=> FAIL', String(e).split('\n')[0].slice(0,100)); }
  child.kill('SIGKILL');
}
