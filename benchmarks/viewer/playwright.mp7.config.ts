import { defineConfig } from '@playwright/test'
import base from './playwright.config'

export default defineConfig(base, {
  testMatch: 'mp7.functional.spec.ts',
  timeout: 180_000,
})
