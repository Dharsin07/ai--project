import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  // Demo mode: single execution, no retries, no parallelism
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0, // No retries - single execution only
  workers: 1, // Single worker to prevent parallel execution
  timeout: 120000, // Global timeout: 120 seconds for slow demo
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results.json' }],
    ['junit', { outputFolder: 'test-results.xml' }]
  ],
  use: {
    baseURL: 'http://localhost:8000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    // Demo mode: slowMo makes actions visible like a real human
    slowMo: process.env.CI ? 0 : 500, // 500ms delay between actions for demo mode
  },
  projects: [
    // Demo project for website testing with Chrome extension - SLOW VISIBLE ACTIONS
    {
      name: 'chrome-extension-demo-slow',
      testMatch: '**/website-rag-with-extension.spec.js',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        headless: false, // Always headful for demo mode
        slowMo: process.env.CI ? 0 : 500, // 500ms delay between actions
        launchOptions: {
          args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection'
          ]
        }
      },
    },
    // Demo project for rag-flow tests
    {
      name: 'chrome-extension-demo-rag',
      testMatch: '**/rag-flow.spec.js',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        headless: false, // Always headful for demo mode
        slowMo: process.env.CI ? 0 : 500, // 500ms delay between actions
        launchOptions: {
          args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection'
          ]
        }
      },
    },
    // Separate project for other tests (basic only)
    {
      name: 'chromium',
      testMatch: '**/basic.spec.js',
      use: { 
        ...devices['Desktop Chrome'],
        channel: 'chrome',
      },
    },
  ],
  webServer: {
    command: 'python gemini_server.py',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
