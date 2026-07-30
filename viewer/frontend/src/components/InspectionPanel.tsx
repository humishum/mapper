import type { Inspection } from '../types/contracts'

export function InspectionPanel({ inspection, onClose }: {
  inspection: Inspection
  onClose: () => void
}) {
  const source = inspection.source
  return (
    <aside
      className="inspection-panel"
      aria-label="Point inspection"
      data-testid="inspection-panel"
    >
      <button className="close" onClick={onClose} aria-label="Close inspection">×</button>
      <p className="eyebrow">Picked point</p>
      <h2>{inspection.artifactId}</h2>
      <dl>
        <dt>Node / index</dt><dd>{inspection.nodeId} / {inspection.pointIndex}</dd>
        <dt>Local XYZ</dt><dd>{inspection.localCoordinate.map(value => value.toFixed(3)).join(', ')}</dd>
        <dt>Source</dt><dd data-testid="inspection-source">{inspection.pointSourceId ?? 'not present'}</dd>
        <dt>Kind</dt><dd>{source?.kind ?? 'unresolved'}</dd>
        <dt>Capture</dt><dd data-testid="inspection-capture">{source?.capture_id ?? '—'}</dd>
        <dt>Run</dt><dd>{source?.run_id ?? '—'}</dd>
        <dt>Frames</dt><dd>{source ? `${source.frame_start ?? '—'}–${source.frame_end ?? '—'}` : '—'}</dd>
        <dt>Confidence</dt><dd>{inspection.confidence?.toFixed(4) ?? 'not present'}</dd>
        <dt>Contributors</dt><dd>{inspection.contributorCount ?? 'not present'}</dd>
        <dt>Alignment</dt><dd>{inspection.alignmentStatus}</dd>
        <dt>Horizontal RMSE</dt><dd>{inspection.horizontalRmseM == null ? '—' : `${inspection.horizontalRmseM.toFixed(2)} m`}</dd>
        <dt>Vertical RMSE</dt><dd>{inspection.verticalRmseM == null ? '—' : `${inspection.verticalRmseM.toFixed(2)} m`}</dd>
      </dl>
    </aside>
  )
}
