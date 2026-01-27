import { test, expect, chromium } from '@playwright/test';
import path from 'path';

// DEMO MODE - Real human-like actions for interview/demo purposes
// This test runs with visible actions and delays
test.describe('RAG Flow Demo - Human Like Actions', () => {
  let context;
  let extensionPath;

  test.beforeAll(async () => {
    // Get the path to the existing Chrome extension
    console.log('🎬 Setting up DEMO mode with Chrome extension...');
    extensionPath = path.resolve(__dirname, '..');
    console.log(`📁 Extension path: ${extensionPath}`);
  });

  test.beforeEach(async () => {
    // Launch Chrome with extension loaded (headful mode) - DEMO MODE
    console.log('🚀 Launching Chrome with extension for DEMO...');
    
    context = await chromium.launchPersistentContext('', {
      headless: false, // Always headful for demo mode
      args: [
        `--disable-extensions-except=${extensionPath}`,
        `--load-extension=${extensionPath}`,
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-web-security',
        '--disable-features=TranslateUI',
        '--disable-ipc-flooding-protection',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding'
      ],
      viewport: { width: 1280, height: 720 },
      ignoreDefaultArgs: ['--enable-blink-features=IdleDetection'],
      // Demo mode: slowMo makes actions visible like a real human
      slowMo: 800, // 800ms delay between actions
    });

    // Wait for extension to fully load - DEMO WAIT
    console.log('⏳ Waiting for extension to load (DEMO mode)...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    console.log('✅ Chrome launched with extension loaded for DEMO');
  });

  // DEMO TEST - Visible human-like actions
  test('should complete RAG flow with human-like demo actions', async ({ page }) => {
    console.log('🎬 Starting DEMO RAG flow test with human-like actions...');
    
    // Step 1: Open the existing web application - DEMO MODE
    console.log('📄 Opening website for DEMO: http://localhost:8000');
    await page.goto('http://localhost:8000');
    
    // Patient wait for full page load - DEMO WAIT
    console.log('⏳ Waiting for full page load (DEMO mode)...');
    await page.waitForLoadState('networkidle');
    
    // Step 2: Verify website loaded successfully - DEMO CHECK
    console.log('� Verifying website load (DEMO mode)...');
    await expect(page).toHaveTitle(/Research Assistant/i);
    
    // Take initial screenshot for DEMO
    await page.screenshot({ 
      path: 'test-results/demo-initial-state.png',
      fullPage: true 
    });

    // Step 3: Upload PDF file on the website - DEMO ACTION
    console.log('📁 Uploading PDF file (DEMO mode - visible action)...');
    
    // Wait for file input to be available - DEMO WAIT
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeVisible({ timeout: 30000 });
    
    // Get the absolute path to sample PDF
    const pdfPath = path.resolve(__dirname, 'sample.pdf');
    console.log(`📄 PDF path: ${pdfPath}`);
    
    // Upload the PDF file on the website - DEMO ACTION (visible)
    await fileInput.setInputFiles(pdfPath);
    
    // Click upload button to process the file on the website - DEMO ACTION
    const uploadButton = page.locator('#uploadBtn');
    await expect(uploadButton).toBeVisible({ timeout: 30000 });
    await uploadButton.click();
    
    // Patient wait for upload completion - WAIT FOR BACKEND CONFIRMATION
    console.log('⏳ Waiting for backend upload confirmation (DEMO mode)...');
    
    // Wait for question input to be enabled (indicates upload complete)
    const questionInput = page.locator('#ragQuestion');
    await expect(questionInput).toBeVisible({ timeout: 45000 });
    
    // Verify the input is no longer disabled (upload complete)
    const isDisabled = await questionInput.isDisabled();
    expect(isDisabled).toBeFalsy();
    console.log('✅ Upload confirmed by backend - question input enabled (DEMO)');
    
    // Step 4: Enter RAG question on the website - DEMO TYPING
    console.log('❓ Entering RAG question (DEMO mode - visible typing)...');
    
    // Type the question on the website - DEMO TYPING (visible character by character)
    const question = 'What is the main conclusion of this document about climate change?';
    await questionInput.fill(question);
    console.log(`📝 Question entered on website (DEMO): ${question}`);

    // Step 5: Click Ask button on the website - DEMO CLICK
    console.log('🔘 Clicking Ask button (DEMO mode - visible click)...');
    
    const askButton = page.locator('#ragQueryBtn');
    await expect(askButton).toBeVisible({ timeout: 30000 });
    await expect(askButton).toBeEnabled({ timeout: 15000 });
    
    await askButton.click();
    
    // Step 6: Wait patiently for AI response - LONG PATIENT WAIT (DEMO)
    console.log('⏳ Waiting patiently for AI response (DEMO mode - up to 45 seconds)...');
    
    // Wait for results section to appear on website - PATIENT WAIT
    const resultsSection = page.locator('#resultsSection, .results-section, [data-testid="results"]');
    await expect(resultsSection).toBeVisible({ timeout: 45000 });
    
    // Step 7: Verify AI answer is rendered on the website - DEMO VERIFICATION
    console.log('🤖 Verifying AI answer (DEMO mode)...');
    
    const answerContainer = page.locator('.answer-box');
    await expect(answerContainer).toBeVisible({ timeout: 30000 });
    
    // Check that answer contains actual content (not placeholder text)
    const answerText = await answerContainer.textContent();
    expect(answerText).toBeTruthy();
    expect(answerText && answerText.trim().length).toBeGreaterThan(10);
    
    // Ensure it's not just placeholder text
    expect(answerText).not.toContain('AI answer will appear here');
    expect(answerText).not.toContain('Loading...');
    expect(answerText).not.toContain('Processing...');
    
    console.log(`✅ AI Answer received on website (DEMO): ${answerText ? answerText.substring(0, 100) : 'No answer'}...`);

    // Step 8: Verify source highlights are visible on the website - DEMO VERIFICATION
    console.log('📚 Verifying source highlights (DEMO mode)...');
    
    const sourceContainer = page.locator('.source-highlight');
    await expect(sourceContainer).toBeVisible({ timeout: 30000 });
    
    // Check that sources contain actual content
    const sourceText = await sourceContainer.textContent();
    expect(sourceText).toBeTruthy();
    expect(sourceText && sourceText.trim().length).toBeGreaterThan(5);
    
    // Ensure it's not just placeholder text
    expect(sourceText).not.toContain('Sources will appear here');
    expect(sourceText).not.toContain('No sources available');
    
    console.log(`✅ Sources found on website (DEMO): ${sourceText ? sourceText.substring(0, 100) : 'No sources'}...`);

    // Step 9: DEMO DELAY - Keep browser open for 10 seconds after AI answer
    console.log('⏸️ DEMO MODE: Keeping browser open for 10 seconds to show results...');
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Step 10: Take final screenshot for DEMO
    console.log('📸 Taking final DEMO screenshot...');
    await page.screenshot({ 
      path: 'test-results/demo-final-success.png',
      fullPage: true 
    });

    // Step 11: Log completion
    console.log('🎉 DEMO RAG flow test completed successfully!');
    
    // Additional verification: Check that the question is displayed on website
    const questionDisplay = page.locator('#researchTopicDisplay, .question-display, [data-testid="question-display"]');
    if (await questionDisplay.isVisible()) {
      const displayedQuestion = await questionDisplay.textContent();
      expect(displayedQuestion).toContain('climate change');
      console.log(`✅ Question displayed correctly on website (DEMO): ${displayedQuestion}`);
    }
    
    // DEMO MODE: Don't close browser immediately - let user see the result
    console.log('👀 DEMO MODE: Browser will stay open for a few more seconds...');
    await new Promise(resolve => setTimeout(resolve, 3000));
  });

  test.afterEach(async () => {
    // Clean up the browser context after the demo test
    if (context) {
      console.log('🧹 Cleaning up browser context (DEMO mode)...');
      await context.close();
    }
  });
});
