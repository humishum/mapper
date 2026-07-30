import Ajv2020, { type ErrorObject } from 'ajv/dist/2020'
import manifestSchema from '../schemas/manifest.v1.json'
import metricsSchema from '../schemas/metrics.v1.json'
import sourcesSchema from '../schemas/sources.v1.json'
import tabularSchema from '../schemas/tabular-contracts.v1.json'
import type { Manifest, SourceRecord } from '../types/contracts'

const ajv = new Ajv2020({ allErrors: true, strict: false, validateFormats: false })
ajv.addSchema(tabularSchema)
ajv.addSchema(manifestSchema)
ajv.addSchema(metricsSchema)
ajv.addSchema(sourcesSchema)

const manifestValidator = ajv.getSchema('https://mapper.local/schemas/manifest.v1.json')
const sourcesValidator = ajv.getSchema('https://mapper.local/schemas/sources.v1.json')
const metricsValidator = ajv.getSchema('https://mapper.local/schemas/metrics.v1.json')

function validationMessage(subject: string, errors?: ErrorObject[] | null): string {
  const details = (errors ?? [])
    .slice(0, 5)
    .map(error => `${error.instancePath || '/'} ${error.message ?? 'is invalid'}`)
    .join('; ')
  return `${subject} failed package schema validation${details ? `: ${details}` : ''}`
}

export function validateManifest(value: unknown): Manifest {
  if (!manifestValidator?.(value)) {
    throw new Error(validationMessage('Manifest', manifestValidator?.errors))
  }
  const manifest = value as Manifest
  validateTabularDeclarations(manifest)
  return manifest
}

export function validateSources(value: unknown): SourceRecord[] {
  if (!Array.isArray(value)) throw new Error('Sources response must be an array')
  const kind = (value[0] as SourceRecord | undefined)?.kind ?? 'capture'
  const document = {
    schema_version: '1.0.0',
    provenance_dimension: 'PointSourceId',
    granularity: kind,
    sources: value,
  }
  if (!sourcesValidator?.(document)) {
    throw new Error(validationMessage('Sources', sourcesValidator?.errors))
  }
  return value as SourceRecord[]
}

export function validateMetrics(value: unknown): void {
  if (!metricsValidator?.(value)) {
    throw new Error(validationMessage('Metrics', metricsValidator?.errors))
  }
}

function validateTabularDeclarations(manifest: Manifest): void {
  const tables = (tabularSchema as {
    tables: Record<string, { required_columns: Record<string, { dtype: string }> }>
  }).tables
  for (const artifact of manifest.artifacts) {
    const contract = tables[artifact.path]
    if (!contract) continue
    const columns = new Map((artifact.columns ?? []).map(column => [column.name, column.dtype]))
    for (const [name, definition] of Object.entries(contract.required_columns)) {
      const dtype = columns.get(name)
      if (dtype === undefined) {
        throw new Error(`${artifact.path} is missing required browser-declared column ${name}`)
      }
      if (dtype !== definition.dtype) {
        throw new Error(
          `${artifact.path} column ${name} declares ${dtype}; expected ${definition.dtype}`,
        )
      }
    }
  }
}

export function availableColorModes(manifest: Manifest): Array<'rgb' | 'elevation' | 'source' | 'confidence'> {
  const pointFiles = manifest.artifacts.filter(item => item.kind === 'points')
  const dimensions = new Set(pointFiles.flatMap(item => item.required_dimensions ?? []))
  const modes: Array<'rgb' | 'elevation' | 'source' | 'confidence'> = ['elevation']
  if (['Red', 'Green', 'Blue'].every(name => dimensions.has(name))) modes.unshift('rgb')
  if (dimensions.has('PointSourceId') || dimensions.has('SourceIndex')) modes.push('source')
  if (dimensions.has('Confidence')) modes.push('confidence')
  return modes
}
