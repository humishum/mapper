import type { CatalogArtifact, SceneMode } from '../types/contracts'

interface Props {
  geographic: CatalogArtifact[]
  unaligned: CatalogArtifact[]
  activeIds: string[]
  mode: SceneMode
  loading: boolean
  onOpen: (artifactId: string, compare?: boolean) => void
}

function count(value: number | null): string {
  return value === null ? 'unknown size' : `${value.toLocaleString()} points`
}

export function CatalogPanel({
  geographic,
  unaligned,
  activeIds,
  mode,
  loading,
  onOpen,
}: Props) {
  const renderArtifact = (artifact: CatalogArtifact, unalignedScene: boolean) => (
    <li
      key={artifact.artifact_id}
      className={activeIds.includes(artifact.artifact_id) ? 'active' : ''}
      data-testid={`catalog-artifact-${artifact.artifact_id}`}
    >
      <div>
        <strong>{artifact.artifact_id}</strong>
        <small>{artifact.kind} · {count(artifact.point_count)}</small>
        {unalignedScene && <span className="warning-chip">Local only</span>}
      </div>
      <div className="row-actions">
        <button
          data-testid={`open-artifact-${artifact.artifact_id}`}
          onClick={() => onOpen(artifact.artifact_id)}
        >
          Open
        </button>
        {mode === 'detail' && !activeIds.includes(artifact.artifact_id) && activeIds.length < 2 && (
          <button
            className="secondary"
            data-testid={`compare-artifact-${artifact.artifact_id}`}
            onClick={() => onOpen(artifact.artifact_id, true)}
          >
            Compare
          </button>
        )}
      </div>
    </li>
  )

  return (
    <aside className="catalog-panel" aria-label="Scene catalog">
      <header>
        <p className="eyebrow">Mapper catalog</p>
        <h1>Reconstruction viewer</h1>
        <span>{loading ? 'Updating visible bounds…' : `${geographic.length + unaligned.length} scenes`}</span>
      </header>
      <section data-testid="geographic-scenes">
        <h2>Geographic sites</h2>
        <ul>{geographic.map(item => renderArtifact(item, false))}</ul>
        {!geographic.length && !loading && <p className="empty">No aligned sites in this view.</p>}
      </section>
      <section data-testid="local-scenes">
        <h2>Unaligned local scenes</h2>
        <ul>{unaligned.map(item => renderArtifact(item, true))}</ul>
        {!unaligned.length && !loading && <p className="empty">No local-only scenes.</p>}
      </section>
    </aside>
  )
}
