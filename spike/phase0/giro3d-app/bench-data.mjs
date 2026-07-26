#!/usr/bin/env node
/**
 * Bundle and run traverse.mjs under Node.
 *
 * Giro3D publishes browser-oriented extensionless imports and includes worker and
 * laz-perf modules in its static graph. The banner supplies the small Node shims
 * those modules require when workers are disabled by the traversal harness.
 */

import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';

const outfile = '/tmp/phase0-traverse.bundle.mjs';
const banner = [
    "import{createRequire as __cr}from'module';",
    "import{dirname as __dn}from'path';",
    "import{fileURLToPath as __f2p}from'url';",
    'const require=__cr(import.meta.url);',
    'const __filename=__f2p(import.meta.url);',
    'const __dirname=__dn(__filename);',
    'globalThis.self=globalThis;',
    'var onmessage,onerror;',
].join('');

await build({
    entryPoints: ['traverse.mjs'],
    bundle: true,
    platform: 'node',
    format: 'esm',
    outfile,
    logLevel: 'error',
    banner: { js: banner },
});

await import(pathToFileURL(outfile));
