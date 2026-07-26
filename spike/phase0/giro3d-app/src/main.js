/**
 * Phase 0 spike: load a Mapper COPC artifact in Giro3D and measure it.
 *
 * This is throwaway scaffolding, but it is deliberately structured the way the plan's
 * renderer boundary will be: everything Giro3D-specific is created in `buildScene()` from
 * a plain descriptor (artifact url, budgets, style) and everything measured is read back
 * through `snapshot()`. Nothing else in the file knows what renderer is underneath.
 *
 * Query parameters:
 *   url=<copc url>          artifact to open (default: mp7 2 cm consolidated)
 *   budget=<n>              point budget (default 2000000)
 *   sse=<n>                 subdivision threshold (default 1)
 *   stride=<n>              source decode decimation (default 2)
 *   size=<n>                point size in px, 0 = automatic (default 0)
 *   attr=<name>             active attribute: Color | Z | PointSourceId | Intensity
 *   up=z|-y                 which model axis is up (default -y, MUSt3R camera frame)
 *   autopath=<name>         run a camera path immediately after load, then report
 */

import Instance from '@giro3d/giro3d/core/Instance.js';
import ColorMap from '@giro3d/giro3d/core/ColorMap.js';
import CoordinateSystem from '@giro3d/giro3d/core/geographic/CoordinateSystem.js';
import { aggregateMemoryUsage } from '@giro3d/giro3d/core/MemoryUsage.js';
import PointCloud from '@giro3d/giro3d/entities/PointCloud.js';
import COPCSource from '@giro3d/giro3d/sources/COPCSource.js';
import { setLazPerfPath } from '@giro3d/giro3d/sources/las/config.js';
import { Color, Vector3 } from 'three';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';

// Local-first: no CDN fetch for the LAZ decoder.
setLazPerfPath('/wasm');

const DEFAULT_URL = 'http://127.0.0.1:8123/mp7-voxel2cm.copc.laz';
const params = new URLSearchParams(window.location.search);
const config = {
    url: params.get('url') ?? DEFAULT_URL,
    budget: Number(params.get('budget') ?? 2_000_000),
    sse: Number(params.get('sse') ?? 1),
    stride: Number(params.get('stride') ?? 2),
    pointSize: Number(params.get('size') ?? 0),
    attribute: params.get('attr') ?? 'Color',
    sourceMax: Number(params.get('sourceMax') ?? 42),
    up: params.get('up') ?? '-y',
    autopath: params.get('autopath'),
};

// ---------------------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------------------

const t0 = performance.now();
const measurements = {
    config,
    userAgent: navigator.userAgent,
    webglRenderer: null,
    marks: {},
    metadata: {},
    longTasks: [],
    paths: {},
    errors: [],
};

// Exported before the scene is built: if renderer construction throws (no WebGL, bad
// artifact url), a harness waiting on window.__spike must still get an answer rather than
// hanging until its timeout.
window.__spike = {
    loaded: false,
    failed: false,
    report: () => measurements,
};

for (const event of ['error', 'unhandledrejection']) {
    window.addEventListener(event, e => {
        measurements.errors.push(String(e.reason ?? e.message ?? e.type));
        window.__spike.failed = true;
    });
}

function mark(name) {
    if (measurements.marks[name] === undefined) {
        measurements.marks[name] = Math.round((performance.now() - t0) * 10) / 10;
    }
}

if (typeof PerformanceObserver !== 'undefined') {
    try {
        new PerformanceObserver(list => {
            for (const entry of list.getEntries()) {
                measurements.longTasks.push({
                    start: Math.round(entry.startTime - t0),
                    duration: Math.round(entry.duration),
                });
            }
        }).observe({ entryTypes: ['longtask'] });
    } catch {
        /* longtask not supported: recorded as an empty list */
    }
}

const frameTimes = [];
let lastFrameAt = null;

function percentile(values, p) {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
    return Math.round(sorted[index] * 100) / 100;
}

// ---------------------------------------------------------------------------------------
// scene
// ---------------------------------------------------------------------------------------

const instance = new Instance({
    target: 'view',
    // A reconstruction in an un-georeferenced local frame must not claim a CRS. Giro3D's
    // `unknown` coordinate system is the local-scene mode the plan asks for.
    crs: CoordinateSystem.unknown,
    backgroundColor: 0x0d1014,
    renderer: { antialias: false, alpha: false },
});

{
    const gl = instance.renderer.getContext();
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    measurements.webglRenderer = dbg
        ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
        : gl.getParameter(gl.RENDERER);
}

const source = new COPCSource({
    url: config.url,
    enableWorkers: true,
    decimate: config.stride,
});
const cloud = new PointCloud({ source });
cloud.pointBudget = config.budget;
cloud.subdivisionThreshold = config.sse;
cloud.pointSize = config.pointSize;

const controls = new MapControls(instance.view.camera, instance.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.15;
controls.screenSpacePanning = false;

/** Model output is in the first camera's frame (x right, y down, z forward), not ENU. */
function upVector() {
    return config.up === 'z' ? new Vector3(0, 0, 1) : new Vector3(0, -1, 0);
}

function frameCamera(bbox) {
    const center = bbox.getCenter(new Vector3());
    const size = bbox.getSize(new Vector3());
    const diagonal = size.length();
    const up = upVector();
    const camera = instance.view.camera;
    camera.up.copy(up);
    camera.near = Math.max(diagonal / 5000, 0.01);
    camera.far = diagonal * 12;
    camera.updateProjectionMatrix();
    // Stand off along the model's "forward" axis, lifted along the up axis.
    const back = new Vector3(0, 0, -1).multiplyScalar(diagonal * 0.55);
    const lift = up.clone().multiplyScalar(diagonal * 0.28);
    camera.position.copy(center).add(back).add(lift);
    controls.target.copy(center);
    controls.object.up.copy(up);
    controls.update();
    instance.notifyChange(camera);
    return { center, size, diagonal };
}

function snapshot() {
    const ctx = { renderer: instance.renderer, objects: new Map() };
    cloud.getMemoryUsage(ctx);
    const memory = aggregateMemoryUsage(ctx);
    const info = instance.renderer.info;
    return {
        elapsedMs: Math.round(performance.now() - t0),
        totalPoints: cloud.pointCount ?? null,
        displayedPoints: cloud.displayedPointCount,
        decimation: cloud.decimation,
        progress: Math.round(instance.progress * 1000) / 1000,
        loading: instance.loading,
        drawCalls: info.render.calls,
        cpuMemoryBytes: memory.cpuMemory,
        gpuMemoryBytes: memory.gpuMemory,
        lastFrameMs: frameTimes.length ? Math.round(frameTimes.at(-1) * 100) / 100 : null,
        p95FrameMs: percentile(frameTimes.slice(-240), 95),
    };
}

function safeSnapshot() {
    try {
        return snapshot();
    } catch {
        return null;
    }
}

// ---------------------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------------------

const el = id => document.getElementById(id);
const fmtInt = n => (n == null ? '-' : n.toLocaleString('en-US'));
const fmtBytes = n => (n == null ? '-' : `${(n / 1024 / 1024).toFixed(1)} MB`);

function updateHud() {
    const s = safeSnapshot();
    if (s == null) return;
    el('m-artifact').textContent = config.url.split('/').pop();
    el('m-total').textContent = fmtInt(s.totalPoints);
    el('m-displayed').textContent = `${fmtInt(s.displayedPoints)}${s.decimation > 1 ? ` /${s.decimation}` : ''}`;
    el('m-frame').textContent = `${s.lastFrameMs ?? '-'} / ${s.p95FrameMs ?? '-'} ms`;
    el('m-calls').textContent = fmtInt(s.drawCalls);
    el('m-mem').textContent = `${fmtBytes(s.cpuMemoryBytes)} cpu / ${fmtBytes(s.gpuMemoryBytes)} gpu`;
    el('m-loading').textContent = s.loading ? `yes (${Math.round(s.progress * 100)}%)` : 'no';
    el('m-first').textContent = measurements.marks.firstGeometry
        ? `${measurements.marks.firstGeometry} ms`
        : '-';
    el('m-idle').textContent = measurements.marks.idle ? `${measurements.marks.idle} ms` : '-';
    el('progress').firstElementChild.style.width = `${(s.loading ? s.progress : 1) * 100}%`;
}

// ---------------------------------------------------------------------------------------
// camera paths
// ---------------------------------------------------------------------------------------

let sceneExtent = null;

function pathPoses(name, u) {
    const { center, diagonal } = sceneExtent;
    const up = upVector();
    const forward = new Vector3(0, 0, 1);
    const side = new Vector3().crossVectors(up, forward).normalize();
    switch (name) {
        case 'orbit': {
            const angle = u * Math.PI * 2;
            const radius = diagonal * 0.5;
            const offset = side
                .clone()
                .multiplyScalar(Math.cos(angle) * radius)
                .add(forward.clone().multiplyScalar(Math.sin(angle) * radius))
                .add(up.clone().multiplyScalar(diagonal * 0.25));
            return { position: center.clone().add(offset), target: center.clone() };
        }
        case 'dive': {
            const start = center
                .clone()
                .add(forward.clone().multiplyScalar(-diagonal * 0.55))
                .add(up.clone().multiplyScalar(diagonal * 0.28));
            const end = center.clone().add(up.clone().multiplyScalar(diagonal * 0.01));
            return { position: start.lerp(end, u), target: center.clone() };
        }
        case 'traverse': {
            const half = diagonal * 0.35;
            const along = forward.clone().multiplyScalar((u - 0.5) * 2 * half);
            const position = center.clone().add(along).add(up.clone().multiplyScalar(diagonal * 0.05));
            const target = position.clone().add(forward.clone().multiplyScalar(diagonal * 0.2));
            return { position, target };
        }
        default:
            throw new Error(`unknown camera path: ${name}`);
    }
}

async function runPath(name = 'orbit', durationMs = 8000) {
    const camera = instance.view.camera;
    const samples = [];
    frameTimes.length = 0;
    const startedAt = performance.now();
    measurements.paths[name] = { status: 'running' };

    await new Promise(resolve => {
        function step(now) {
            const u = Math.min((now - startedAt) / durationMs, 1);
            const { position, target } = pathPoses(name, u);
            camera.position.copy(position);
            controls.target.copy(target);
            controls.update();
            instance.notifyChange(camera);
            samples.push(snapshot().displayedPoints);
            if (u < 1) requestAnimationFrame(step);
            else resolve();
        }
        requestAnimationFrame(step);
    });

    const longTasksDuring = measurements.longTasks.filter(
        t => t.start >= startedAt - t0 && t.duration >= 50,
    );
    const result = {
        status: 'done',
        durationMs: Math.round(performance.now() - startedAt),
        frames: frameTimes.length,
        frameMs: {
            p50: percentile(frameTimes, 50),
            p95: percentile(frameTimes, 95),
            max: percentile(frameTimes, 100),
        },
        longTasksOver50ms: longTasksDuring.length,
        displayedPoints: {
            min: Math.min(...samples),
            max: Math.max(...samples),
        },
        endState: snapshot(),
    };
    measurements.paths[name] = result;
    return result;
}

function waitForIdle(timeoutMs = 120000, stableFrames = 8) {
    return new Promise(resolve => {
        let stable = 0;
        const startedAt = performance.now();
        const tick = () => {
            if (!instance.loading) stable += 1;
            else stable = 0;
            if (stable >= stableFrames) {
                mark('idle');
                resolve(true);
            } else if (performance.now() - startedAt > timeoutMs) {
                resolve(false);
            } else {
                requestAnimationFrame(tick);
            }
        };
        requestAnimationFrame(tick);
    });
}

// ---------------------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------------------

instance.addEventListener('after-render', () => {
    const now = performance.now();
    if (lastFrameAt != null) frameTimes.push(now - lastFrameAt);
    lastFrameAt = now;
    if (cloud.displayedPointCount > 0) mark('firstGeometry');
    if (frameTimes.length % 6 === 0) updateHud();
});

function buildUi(metadata) {
    if (metadata.attributes.some(attribute => attribute.name === 'PointSourceId')) {
        cloud.setAttributeColorMap(
            'PointSourceId',
            new ColorMap({
                colors: [
                    new Color('#2f6fed'),
                    new Color('#27ae60'),
                    new Color('#f2c94c'),
                    new Color('#eb5757'),
                    new Color('#9b51e0'),
                ],
                min: 0,
                max: config.sourceMax + 1,
            }),
        );
    }

    const attrSelect = el('attr');
    attrSelect.innerHTML = '';
    for (const attribute of metadata.attributes) {
        const option = document.createElement('option');
        option.value = attribute.name;
        option.textContent = `${attribute.name} (${attribute.interpretation})`;
        attrSelect.append(option);
    }
    attrSelect.value = metadata.attributes.some(a => a.name === config.attribute)
        ? config.attribute
        : metadata.attributes[0].name;
    const applyAttribute = () => {
        cloud.setActiveAttribute(attrSelect.value);
        cloud.setColoringMode('attribute');
        instance.notifyChange(cloud);
    };
    applyAttribute();
    attrSelect.addEventListener('change', applyAttribute);

    const bind = (id, initial, apply, format = v => v) => {
        const input = el(id);
        input.value = initial;
        el(`${id}-val`).textContent = format(initial);
        input.addEventListener('input', () => {
            const value = Number(input.value);
            el(`${id}-val`).textContent = format(value);
            apply(value);
            instance.notifyChange(cloud);
        });
    };
    bind('budget', config.budget, v => (cloud.pointBudget = v), v => v.toLocaleString('en-US'));
    bind('sse', config.sse, v => (cloud.subdivisionThreshold = v));
    bind('size', config.pointSize, v => (cloud.pointSize = v), v => (v === 0 ? 'auto' : `${v} px`));

    el('run-path').addEventListener('click', async () => {
        const result = await runPath('orbit');
        console.log('camera path result', result);
    });
    el('copy-json').addEventListener('click', async () => {
        const text = JSON.stringify(window.__spike.report(), null, 2);
        await navigator.clipboard.writeText(text).catch(() => console.log(text));
    });
}

async function main() {
    try {
        await instance.add(cloud);
        mark('entityReady');

        const metadata = await source.getMetadata();
        measurements.metadata = {
            pointCount: metadata.pointCount,
            crs: metadata.crs?.name ?? String(metadata.crs),
            isGeographic: metadata.crs?.isGeographic?.() ?? null,
            attributes: metadata.attributes.map(a => ({
                name: a.name,
                interpretation: a.interpretation,
                type: a.type,
                size: a.size,
            })),
            volume: metadata.volume
                ? {
                      min: metadata.volume.min.toArray(),
                      max: metadata.volume.max.toArray(),
                  }
                : null,
        };
        mark('metadata');

        el('m-crs').textContent = measurements.metadata.crs ?? 'unknown';
        if (!measurements.metadata.isGeographic) {
            const note = el('frame-note');
            note.hidden = false;
            note.innerHTML =
                'Local scene mode: artifact declares no CRS. Positions are model units, ' +
                'not georeferenced &mdash; nothing here is placed on a map.';
        }

        const bbox = cloud.getBoundingBox();
        sceneExtent = frameCamera(bbox);
        buildUi(metadata);
        instance.view.setControls(controls);

        await waitForIdle();
        updateHud();

        if (config.autopath) {
            await runPath(config.autopath);
            updateHud();
        }
        window.__spike.loaded = true;
    } catch (error) {
        measurements.errors.push(String(error?.stack ?? error));
        window.__spike.failed = true;
        console.error(error);
        const note = el('frame-note');
        note.hidden = false;
        note.textContent = `Failed to load: ${error}`;
    }
}

Object.assign(window.__spike, {
    instance,
    cloud,
    source,
    snapshot,
    runPath,
    waitForIdle,
    report() {
        return {
            ...measurements,
            frameMsSummary: {
                samples: frameTimes.length,
                p50: percentile(frameTimes, 50),
                p95: percentile(frameTimes, 95),
                max: percentile(frameTimes, 100),
            },
            final: safeSnapshot(),
        };
    },
});

main();
