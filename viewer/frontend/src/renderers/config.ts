export interface BasemapConfig {
  enabled: boolean
  url: string
  attribution: string
}

function envBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback
  return !['0', 'false', 'off', 'no'].includes(value.toLowerCase())
}

export const basemapConfig: BasemapConfig = {
  enabled: envBoolean(import.meta.env.VITE_BASEMAP_ENABLED as string | undefined, true),
  url: (import.meta.env.VITE_BASEMAP_URL as string | undefined)
    ?? 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  attribution: (import.meta.env.VITE_BASEMAP_ATTRIBUTION as string | undefined)
    ?? '© OpenStreetMap contributors',
}
