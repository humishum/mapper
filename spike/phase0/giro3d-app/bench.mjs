#!/usr/bin/env node
/**
 * Phase 0 benchmark driver: load Mapper COPC artifacts in the Giro3D spike app, replay
 * fixed camera paths, and record app marks, renderer counters and server-side byte ranges.
 *
 * Every scenario gets a fresh browser context so "cold" means an empty HTTP cache, and the
 * server request log is rotated per scenario so bytes-ranged is measured at the server
 * rather than trusted from the client.
 *
 * Prerequisites (both started outside this script):
 *   python spike/phase0/range_server.py --root <artifacts> --port 8123 --log <log.jsonl>
 *   npx vite --host 127.0.0.1 --port 5180        (inside spike/phase0/giro3d-app)
 *
 * Usage:
 *   node bench.mjs --out <results-dir> [--gl swiftshader|egl|vulkan] [--only <name>]
 */

import { chromium } from 'playwright';
import { mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
    args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
}

const OUT_DIR = path.resolve(args.get('out') ?? '/home/ape/mapper_output/phase0_spike/bench');
const APP = 'http://127.0.0.1:5180';
const ARTIFACT_BASE = 'http://127.0.0.1:8123';
const RANGE_LOG = args.get('range-log')
    ?? '/home/ape/mapper_output/phase0_spike/reports/range-requests.jsonl';
const GL_MODE = args.get('gl') ?? 'swiftshader';
const VIEWPORT = { width: 1600, height: 900 };

/** Camera paths are replayed in this order for every scenario. */
const PATHS = [
    { name: 'orbit', durationMs: 8000 },
    { name: 'dive', durationMs: 6000 },
    { name: 'traverse', durationMs: 6000 },
];

const SCENARIOS = [
    {
        name: 'window000-small',
        artifact: 'window000.copc.laz',
        params: { budget: 2_000_000, sse: 1, stride: 1 },
    },
    {
        name: 'mp7-voxel2cm-budget2m',
        artifact: 'mp7-voxel2cm.copc.laz',
        params: { budget: 2_000_000, sse: 1, stride: 1 },
    },
    {
        name: 'mp7-voxel2cm-budget2m-stride2',
        artifact: 'mp7-voxel2cm.copc.laz',
        params: { budget: 2_000_000, sse: 1, stride: 2 },
    },
    {
        name: 'mp7-voxel2cm-budget1m',
        artifact: 'mp7-voxel2cm.copc.laz',
        params: { budget: 1_000_000, sse: 1, stride: 1 },
    },
    {
        name: 'mp7-voxel2cm-budget5m',
        artifact: 'mp7-voxel2cm.copc.laz',
        params: { budget: 5_000_000, sse: 1, stride: 1 },
    },
    {
        name: 'mp7-full-budget2m',
        artifact: 'mp7-full.copc.laz',
        params: { budget: 2_000_000, sse: 1, stride: 1 },
    },
    {
        name: 'mp7-full-budget2m-stride2',
        artifact: 'mp7-full.copc.laz',
        params: { budget: 2_000_000, sse: 1, stride: 2 },
    },
    {
        name: 'mp7-full-budget1m',
        artifact: 'mp7-full.copc.laz',
        params: { budget: 1_000_000, sse: 1, stride: 1 },
    },
    {
        name: 'mp7-full-budget2m-sse2',
        artifact: 'mp7-full.copc.laz',
        params: { budget: 2_000_000, sse: 2, stride: 1 },
    },
    {
        name: 'mp7-voxel2cm-sourceindex',
        artifact: 'mp7-voxel2cm.copc.laz',
        params: { budget: 2_000_000, sse: 1, stride: 1, attr: 'PointSourceId' },
    },
];

const GL_ARGS = {
    swiftshader: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    // Chrome's Linux ANGLE backend is selected with use-gl=angle. use-gl=egl is
    // rejected by current Chromium builds before a WebGL context is attempted.
    egl: ['--use-gl=angle', '--use-angle=gl', '--enable-gpu', '--ignore-gpu-blocklist'],
    vulkan: [
        '--use-gl=angle',
        '--use-angle=vulkan',
        '--enable-features=Vulkan',
        '--enable-gpu',
        '--ignore-gpu-blocklist',
    ],
};

// Playwright's own launcher is unusable in this environment: it drives the browser over
// --remote-debugging-pipe, which never connects here (the browser starts and then no CDP
// session is ever established). Launching the browser ourselves with a TCP debugging port
// and attaching with connectOverCDP works, so the driver does that instead.
const CHROME_BIN =
    args.get('chrome')
    ?? '/home/ape/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome';
const CDP_PORT = Number(args.get('cdp-port') ?? 9333);

function launchChrome() {
    const child = spawn(
        CHROME_BIN,
        [
            '--headless=new',
            `--remote-debugging-port=${CDP_PORT}`,
            `--user-data-dir=/tmp/claude-1000/phase0-chrome-${CDP_PORT}`,
            `--window-size=${VIEWPORT.width},${VIEWPORT.height}`,
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--hide-scrollbars',
            '--mute-audio',
            '--no-first-run',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            ...(GL_ARGS[GL_MODE] ?? []),
        ],
        { stdio: ['ignore', 'pipe', 'pipe'] },
    );
    child.stderr.on('data', chunk => {
        const text = String(chunk);
        if (/ERROR|FATAL/.test(text) && !/DEPRECATED_ENDPOINT|gcm/.test(text)) {
            process.stdout.write(`[chrome] ${text}`);
        }
    });
    return child;
}

async function waitForCdp(timeoutMs = 30000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            const response = await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`);
            if (response.ok) return await response.json();
        } catch {
            /* not up yet */
        }
        await new Promise(resolve => setTimeout(resolve, 250));
    }
    throw new Error(`no CDP endpoint on port ${CDP_PORT} after ${timeoutMs} ms`);
}

async function readRangeLog() {
    if (!existsSync(RANGE_LOG)) return { requests: 0, bytes: 0, aborted: 0, fullGets: 0 };
    const text = await readFile(RANGE_LOG, 'utf8');
    let requests = 0;
    let bytes = 0;
    let aborted = 0;
    let fullGets = 0;
    for (const line of text.split('\n')) {
        if (!line.trim()) continue;
        const entry = JSON.parse(line);
        if (entry.event === 'range') {
            requests += 1;
            bytes += entry.length;
        } else if (entry.event === 'aborted') {
            aborted += 1;
        } else if (entry.event === 'full') {
            fullGets += 1;
        }
    }
    return { requests, bytes, aborted, fullGets };
}

async function runScenario(browser, scenario) {
    console.log(`\n=== ${scenario.name} (${scenario.artifact}) ===`);
    await rm(RANGE_LOG, { force: true });

    // Over CDP, newContext() never returns in this environment, so scenarios share the
    // default context and cold-cache isolation comes from an explicit cache clear.
    const context = browser.contexts()[0];
    const page = await context.newPage();
    await page.setViewportSize(VIEWPORT);
    const cdp = await context.newCDPSession(page);
    await cdp.send('Network.enable');
    await cdp.send('Network.clearBrowserCache');
    const consoleErrors = [];
    page.on('console', msg => {
        if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    page.on('pageerror', error => consoleErrors.push(String(error)));

    const query = new URLSearchParams({
        url: `${ARTIFACT_BASE}/${scenario.artifact}`,
        ...Object.fromEntries(Object.entries(scenario.params).map(([k, v]) => [k, String(v)])),
    });

    const startedAt = Date.now();
    await page.goto(`${APP}/?${query}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__spike?.loaded || window.__spike?.failed, null, {
        timeout: 300_000,
    });
    const loadWallMs = Date.now() - startedAt;

    const afterLoad = await readRangeLog();
    const loadReport = await page.evaluate(() => window.__spike.report());

    await page.screenshot({ path: path.join(OUT_DIR, `${scenario.name}-loaded.png`) });

    const pathResults = {};
    if (!loadReport.errors.length) {
        for (const { name, durationMs } of PATHS) {
            pathResults[name] = await page.evaluate(
                ([n, d]) => window.__spike.runPath(n, d),
                [name, durationMs],
            );
            await page.screenshot({
                path: path.join(OUT_DIR, `${scenario.name}-${name}.png`),
            });
        }
    }

    const afterPaths = await readRangeLog();
    const heap = await page.evaluate(() => performance.memory?.usedJSHeapSize ?? null);
    const finalReport = await page.evaluate(() => window.__spike.report());

    await page.close();

    return {
        scenario: scenario.name,
        artifact: scenario.artifact,
        params: scenario.params,
        loadWallMs,
        marks: finalReport.marks,
        metadata: finalReport.metadata,
        webglRenderer: finalReport.webglRenderer,
        serverRanges: { onLoad: afterLoad, afterPaths },
        paths: pathResults,
        longTasksOver50ms: finalReport.longTasks.filter(t => t.duration >= 50).length,
        longTasksMaxMs: finalReport.longTasks.reduce((m, t) => Math.max(m, t.duration), 0),
        jsHeapBytes: heap,
        final: finalReport.final,
        errors: [...finalReport.errors, ...consoleErrors],
    };
}

async function main() {
    await mkdir(OUT_DIR, { recursive: true });
    const only = args.get('only');
    const scenarios = only ? SCENARIOS.filter(s => s.name === only) : SCENARIOS;
    if (scenarios.length === 0) throw new Error(`no scenario matched --only ${only}`);

    const chrome = launchChrome();
    const version = await waitForCdp();
    console.log(`browser: ${version.Browser} (gl mode: ${GL_MODE})`);
    const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP_PORT}`);

    const results = [];
    for (const scenario of scenarios) {
        try {
            results.push(await runScenario(browser, scenario));
            const last = results.at(-1);
            console.log(
                `  first geometry: ${last.marks.firstGeometry ?? 'n/a'} ms  ` +
                `idle: ${last.marks.idle ?? 'n/a'} ms  ` +
                `points: ${last.final?.displayedPoints?.toLocaleString() ?? 'n/a'}  ` +
                `ranged: ${(last.serverRanges.onLoad.bytes / 1e6).toFixed(1)} MB in ` +
                `${last.serverRanges.onLoad.requests} requests`,
            );
            for (const [name, result] of Object.entries(last.paths)) {
                console.log(
                    `  path ${name}: p50 ${result.frameMs?.p50} ms, p95 ${result.frameMs?.p95} ms, ` +
                    `max ${result.frameMs?.max} ms, points ${result.displayedPoints?.max?.toLocaleString()}`,
                );
            }
            if (last.errors.length) console.log(`  errors: ${last.errors.slice(0, 3).join(' | ')}`);
        } catch (error) {
            console.error(`  FAILED: ${error.message}`);
            results.push({ scenario: scenario.name, fatal: String(error) });
        }
    }

    await browser.close();
    chrome.kill('SIGTERM');

    const summary = {
        recordedAt: new Date().toISOString(),
        machine: {
            platform: process.platform,
            arch: process.arch,
            cpus: (await import('node:os')).cpus().length,
            totalMemBytes: (await import('node:os')).totalmem(),
        },
        browser: version.Browser,
        glMode: GL_MODE,
        viewport: VIEWPORT,
        paths: PATHS,
        results,
    };
    const outFile = path.join(OUT_DIR, `results-${GL_MODE}.json`);
    await writeFile(outFile, `${JSON.stringify(summary, null, 2)}\n`);
    console.log(`\nwrote ${outFile}`);
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
