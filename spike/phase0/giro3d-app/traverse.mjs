#!/usr/bin/env node
/**
 * Phase 0: drive Giro3D's COPCSource directly, without a GPU, and measure the data path.
 *
 * Chromium cannot reach localhost HTTP in this sandbox, so the browser scenarios in
 * bench.mjs cannot run here. This harness measures the part of the question that does not
 * need a GPU, using the same Giro3D code the viewer would use: octree hierarchy loading,
 * screen-space-error node selection, byte-range fetching, LAZ decode and attribute
 * extraction. What it does NOT measure is GPU upload, draw time or how flying through the
 * scene feels - those still need bench.mjs on a machine with a display.
 *
 * Usage:
 *   node traverse.mjs --artifacts http://127.0.0.1:8123 --out results.json
 */

import { readFileSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Vector3 } from 'three';
import COPCSource from '@giro3d/giro3d/sources/COPCSource.js';
import { traverseNode } from '@giro3d/giro3d/sources/PointCloudSource.js';
import { setLazPerfWasmBinary } from '@giro3d/giro3d/sources/las/config.js';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
    args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
}
const BASE = args.get('artifacts') ?? 'http://127.0.0.1:8123';
const OUT = args.get('out') ?? '/home/ape/mapper_output/phase0_spike/bench/traverse.json';
const RANGE_LOG = args.get('range-log')
    ?? '/home/ape/mapper_output/phase0_spike/reports/range-requests.jsonl';

// Node has no Web Workers, so decode runs on the main thread (enableWorkers: false) and the
// wasm binary is handed over directly instead of being fetched from a CDN.
// Absolute default rather than import.meta.url: this file is run through esbuild (Giro3D
// ships extensionless bundler-style imports that Node cannot resolve), so at runtime it
// lives next to the bundle, not next to this source.
setLazPerfWasmBinary(
    readFileSync(
        args.get('wasm')
            ?? '/home/ape/repos/mapper/spike/phase0/giro3d-app/public/wasm/laz-perf.wasm',
    ).buffer,
);

const VIEWPORT_HEIGHT = 900;
const FOV_DEG = 50;

/** Screen-space error of a node, in pixels, for a perspective camera. */
function screenSpaceError(node, cameraPosition, viewportHeight = VIEWPORT_HEIGHT) {
    const distance = Math.max(node.center.distanceTo(cameraPosition), 1e-6);
    const halfFov = (FOV_DEG * Math.PI) / 360;
    return (node.geometricError * viewportHeight) / (2 * Math.tan(halfFov) * distance);
}

/**
 * Select the nodes a screen-space-error renderer would show, nearest-first, until the point
 * budget runs out. Mirrors the rule Giro3D applies per node (refine while the node's
 * geometric error projects to more than `threshold` pixels) rather than reimplementing its
 * scheduler.
 */
function selectNodes(root, cameraPosition, { threshold = 1, budget = 2_000_000 } = {}) {
    const visible = [];
    let visited = 0;
    traverseNode(root, node => {
        visited += 1;
        const sse = screenSpaceError(node, cameraPosition);
        if (node.hasData) visible.push({ node, sse, distance: node.center.distanceTo(cameraPosition) });
        // Stop descending when this node is already fine enough for the screen.
        return sse > threshold;
    });
    visible.sort((a, b) => a.distance - b.distance);
    const selected = [];
    let points = 0;
    for (const entry of visible) {
        if (points + (entry.node.pointCount ?? 0) > budget) continue;
        selected.push(entry);
        points += entry.node.pointCount ?? 0;
    }
    return { visited, candidates: visible.length, selected, points };
}

function readRangeLog() {
    try {
        const text = readFileSync(RANGE_LOG, 'utf8');
        let requests = 0;
        let bytes = 0;
        for (const line of text.split('\n')) {
            if (!line.trim()) continue;
            const entry = JSON.parse(line);
            if (entry.event === 'range') {
                requests += 1;
                bytes += entry.length;
            }
        }
        return { requests, bytes };
    } catch {
        return { requests: 0, bytes: 0 };
    }
}

function percentile(values, p) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return Math.round(sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))] * 100) / 100;
}

async function measureArtifact(file, { budget = 2_000_000, threshold = 1 } = {}) {
    const url = `${BASE}/${file}`;
    console.log(`\n=== ${file} (budget ${budget.toLocaleString()}, sse ${threshold}) ===`);
    const rangesBefore = readRangeLog();

    const source = new COPCSource({ url, enableWorkers: false });

    const tStart = performance.now();
    await source.initialize();
    const tInit = performance.now() - tStart;

    const metadata = await source.getMetadata();
    const tMetadata = performance.now() - tStart;

    const hierarchy = await source.getHierarchy();
    const tHierarchy = performance.now() - tStart;
    const afterHierarchy = readRangeLog();

    let nodeCount = 0;
    let deepest = 0;
    traverseNode(hierarchy, node => {
        nodeCount += 1;
        deepest = Math.max(deepest, node.depth);
        return true;
    });

    const volumeMin = metadata.volume.min;
    const volumeMax = metadata.volume.max;
    const center = new Vector3().addVectors(volumeMin, volumeMax).multiplyScalar(0.5);
    const size = new Vector3().subVectors(volumeMax, volumeMin);
    const diagonal = size.length();

    const cameras = {
        overview: center.clone().add(new Vector3(0, -diagonal * 0.28, -diagonal * 0.55)),
        mid: center.clone().add(new Vector3(0, -diagonal * 0.08, -diagonal * 0.15)),
        close: center.clone().add(new Vector3(0, -diagonal * 0.01, -diagonal * 0.02)),
    };

    const colorAttribute = metadata.attributes.find(a => a.name === 'Color');
    const views = {};

    for (const [viewName, cameraPosition] of Object.entries(cameras)) {
        const selection = selectNodes(hierarchy, cameraPosition, { threshold, budget });
        const beforeView = readRangeLog();
        const tView = performance.now();
        const decodeTimes = [];
        let decodedPoints = 0;
        let firstNodeMs = null;

        for (const { node } of selection.selected) {
            const t = performance.now();
            const data = await source.getNodeData({
                node,
                position: true,
                attributes: colorAttribute ? [colorAttribute] : [],
            });
            const dt = performance.now() - t;
            decodeTimes.push(dt);
            if (firstNodeMs == null) firstNodeMs = performance.now() - tView;
            decodedPoints += data.pointCount ?? 0;
        }

        const wall = performance.now() - tView;
        const afterView = readRangeLog();
        views[viewName] = {
            cameraPosition: cameraPosition.toArray().map(v => Math.round(v * 100) / 100),
            nodesVisited: selection.visited,
            nodesCandidate: selection.candidates,
            nodesLoaded: selection.selected.length,
            pointsSelected: selection.points,
            pointsDecoded: decodedPoints,
            firstNodeMs: firstNodeMs == null ? null : Math.round(firstNodeMs * 10) / 10,
            allNodesMs: Math.round(wall),
            perNodeMs: {
                p50: percentile(decodeTimes, 50),
                p95: percentile(decodeTimes, 95),
                max: percentile(decodeTimes, 100),
            },
            decodedPointsPerSecond: Math.round(decodedPoints / (wall / 1000)),
            rangeRequests: afterView.requests - beforeView.requests,
            rangeBytes: afterView.bytes - beforeView.bytes,
        };
        console.log(
            `  ${viewName}: ${views[viewName].nodesLoaded} nodes / ` +
            `${decodedPoints.toLocaleString()} points in ${views[viewName].allNodesMs} ms ` +
            `(${(views[viewName].rangeBytes / 1e6).toFixed(1)} MB, ` +
            `first node ${views[viewName].firstNodeMs} ms)`,
        );
    }

    source.dispose();

    const result = {
        artifact: file,
        url,
        budget,
        sseThreshold: threshold,
        metadata: {
            pointCount: metadata.pointCount,
            crs: String(metadata.crs?.name ?? metadata.crs),
            attributes: metadata.attributes.map(a => `${a.name}:${a.type}${a.size * 8}${a.dimension === 3 ? 'x3' : ''}`),
            volumeMin: volumeMin.toArray(),
            volumeMax: volumeMax.toArray(),
        },
        hierarchy: { nodes: nodeCount, deepestLevel: deepest },
        timings: {
            initializeMs: Math.round(tInit),
            metadataMs: Math.round(tMetadata),
            hierarchyMs: Math.round(tHierarchy),
        },
        openCost: {
            rangeRequests: afterHierarchy.requests - rangesBefore.requests,
            rangeBytes: afterHierarchy.bytes - rangesBefore.bytes,
        },
        views,
    };
    console.log(
        `  open: ${result.openCost.rangeRequests} requests / ` +
        `${(result.openCost.rangeBytes / 1e3).toFixed(0)} kB, hierarchy ${nodeCount} nodes, ` +
        `init ${result.timings.initializeMs} ms`,
    );
    return result;
}

async function main() {
    const results = [];
    results.push(await measureArtifact('window000.copc.laz'));
    results.push(await measureArtifact('mp7-voxel2cm.copc.laz'));
    results.push(await measureArtifact('mp7-voxel2cm.copc.laz', { budget: 5_000_000 }));
    results.push(await measureArtifact('mp7-full.copc.laz'));
    results.push(await measureArtifact('mp7-full.copc.laz', { threshold: 2 }));

    const os = await import('node:os');
    const summary = {
        recordedAt: new Date().toISOString(),
        harness: 'spike/phase0/giro3d-app/traverse.mjs',
        measures: 'giro3d COPCSource: hierarchy, byte ranges, LAZ decode (main thread, no GPU)',
        excludes: 'GPU upload, draw calls, frame times, interaction feel',
        machine: { cpus: os.cpus().length, totalMemBytes: os.totalmem(), node: process.version },
        viewport: { height: VIEWPORT_HEIGHT, fovDeg: FOV_DEG },
        results,
    };
    await writeFile(OUT, `${JSON.stringify(summary, null, 2)}\n`);
    console.log(`\nwrote ${OUT}`);
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
