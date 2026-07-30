import type { ColorMode, LayerState, Manifest, PerformanceState } from '../types/contracts'

interface Props {
  layers: LayerState[]
  manifests: Map<string, Manifest>
  performance: PerformanceState
  onChange: (artifactId: string, patch: Partial<LayerState>) => void
}

const LABELS: Record<ColorMode, string> = {
  rgb: 'RGB',
  elevation: 'Elevation',
  source: 'Source',
  confidence: 'Confidence',
}

export function LayerControls({ layers, manifests, performance, onChange }: Props) {
  return (
    <aside className="layer-controls" aria-label="Detail layer controls">
      <header>
        <p className="eyebrow">Local detail</p>
        <h2>Layers</h2>
      </header>
      {layers.map(layer => {
        const manifest = manifests.get(layer.artifactId)
        return (
          <section key={layer.artifactId} className="layer-card">
            <div className="layer-title">
              <strong>{layer.artifactId}</strong>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={layer.visible}
                  onChange={event => onChange(layer.artifactId, { visible: event.target.checked })}
                />
                Visible
              </label>
            </div>
            {manifest?.alignment.status === 'unaligned' && (
              <p
                className="warning"
                data-testid={`alignment-warning-${layer.artifactId}`}
              >
                Local coordinates only: {manifest.alignment.rejection_reason ?? 'geographic alignment rejected'}
              </p>
            )}
            <label>
              Opacity <output>{Math.round(layer.opacity * 100)}%</output>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={layer.opacity}
                onChange={event => onChange(layer.artifactId, { opacity: Number(event.target.value) })}
              />
            </label>
            <label>
              Point size <output>{layer.pointSize.toFixed(1)} px</output>
              <input
                type="range"
                min="0.5"
                max="8"
                step="0.5"
                value={layer.pointSize}
                onChange={event => onChange(layer.artifactId, { pointSize: Number(event.target.value) })}
              />
            </label>
            <label>
              Color
              <select
                data-testid={`color-mode-${layer.artifactId}`}
                value={layer.colorMode}
                onChange={event => onChange(layer.artifactId, { colorMode: event.target.value as ColorMode })}
              >
                {(Object.keys(LABELS) as ColorMode[]).map(mode => (
                  <option key={mode} value={mode} disabled={!layer.availableColorModes.includes(mode)}>
                    {LABELS[mode]}{!layer.availableColorModes.includes(mode) ? ' — unavailable' : ''}
                  </option>
                ))}
              </select>
            </label>
            {layer.loading && <progress value={layer.progress} max="1" aria-label="Layer loading progress" />}
          </section>
        )
      })}
      <footer>
        <span>{performance.visiblePoints.toLocaleString()} / {performance.pointBudget.toLocaleString()} visible points</span>
        <span>CPU/GPU pools ≤ 256 MB each</span>
      </footer>
    </aside>
  )
}
