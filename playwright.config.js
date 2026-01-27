import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  // Demo mode: single execution, no retries, no parallelism
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0, // No retries - single execution only
  workers: 1, // Single worker to prevent parallel execution
  timeout: 300000, // Global timeout: 300 seconds (5 minutes) for ultimate comprehensive
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
    slowMo: process.env.CI ? 0 : 300, // 300ms delay between actions for balanced speed
  },
  projects: [
    // Demo project for basic tests
    {
      name: 'chrome-extension-demo-basic',
      testMatch: '**/basic.spec.js',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        headless: false, // Always headful for demo mode
        slowMo: process.env.CI ? 0 : 300, // 300ms delay between actions
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
    // Demo project for ultimate comprehensive workflow (all three workflows)
    {
      name: 'chrome-extension-demo-ultimate',
      testMatch: '**/ultimate-comprehensive-workflow.spec.js',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        headless: false, // Always headful for demo mode
        slowMo: process.env.CI ? 0 : 300, // 300ms delay between actions (balanced speed)
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
  ],
  webServer: {
    command: 'python gemini_server.py',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
    timeout: 300 * 1000, // 5 minutes timeout
  },
});
