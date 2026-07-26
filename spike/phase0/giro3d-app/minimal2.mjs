import { chromium } from 'playwright';
const b = await chromium.launch({ headless:true, args:['--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-zygote','--single-process'] });
const p = await b.newPage();
await p.goto('data:text/html,<h1>hi</h1>');
console.log('OK title:', await p.title());
await b.close();
