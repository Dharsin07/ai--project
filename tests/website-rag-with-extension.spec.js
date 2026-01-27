import { test, expect, chromium } from '@playwright/test';
import path from 'path';

// SLOW DEMO MODE - Interview-friendly with explicit delays
// This test runs with visible actions and specific timing delays
test.describe('Slow Demo RAG Flow with Chrome Extension', () => {
  let context;
  let extensionPath;

  test.beforeAll(async () => {
    // Get the path to the existing Chrome extension
    console.log('🎬 Setting up SLOW DEMO mode with Chrome extension...');
    extensionPath = path.resolve(__dirname, '..');
    console.log(`📁 Extension path: ${extensionPath}`);
  });

  test.beforeEach(async () => {
    // Launch Chrome with extension loaded (headful mode) - SLOW DEMO MODE
    console.log('🚀 Launching Chrome with extension for SLOW DEMO...');
    
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
      slowMo: 500, // 500ms delay between actions
    });

    // Wait for extension to fully load - DEMO WAIT
    console.log('⏳ Waiting for Chrome extension to load (2 seconds)...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    console.log('✅ Chrome launched with extension loaded for SLOW DEMO');
  });

  // SLOW DEMO TEST - Visible human-like actions with explicit delays
  test('should complete RAG flow with slow demo-friendly delays', async () => {
    console.log('🎬 Starting SLOW DEMO RAG flow test with explicit delays...');
    
    // Step 1: Open the existing web application - DEMO MODE
    console.log('📄 Opening website for SLOW DEMO: http://localhost:8000');
    const page = await context.newPage();
    await page.goto('http://localhost:8000');
    
    // Wait 2 seconds after website loads
    console.log('⏳ Waiting 2 seconds after website loads...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 2: Upload PDF file on the website - DEMO ACTION
    console.log('📁 Uploading PDF file (SLOW DEMO mode)...');
    
    // Wait for file input to be available
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeVisible({ timeout: 30000 });
    
    // Get the absolute path to sample PDF
    const pdfPath = path.resolve(__dirname, 'sample.pdf');
    console.log(`📄 PDF path: ${pdfPath}`);
    
    // Upload the PDF file on the website
    await fileInput.setInputFiles(pdfPath);
    
    // Wait 3 seconds before clicking upload button
    console.log('⏳ Waiting 3 seconds before clicking upload button...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Click upload button to process the file on the website
    const uploadButton = page.locator('#uploadBtn');
    await expect(uploadButton).toBeVisible({ timeout: 30000 });
    await uploadButton.click();
    
    // Wait for upload completion
    console.log('⏳ Waiting for upload completion...');
    
    // Wait for question input to be enabled (indicates upload complete)
    const questionInput = page.locator('#ragQuestion');
    await expect(questionInput).toBeVisible({ timeout: 60000 });
    
    // Verify the input is no longer disabled (upload complete)
    const isDisabled = await questionInput.isDisabled();
    expect(isDisabled).toBeFalsy();
    console.log('✅ Upload completed successfully');
    
    // Wait 2 seconds after upload completes
    console.log('⏳ Waiting 2 seconds after upload completes...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 3: Enter RAG question on the website - SLOW TYPING
    console.log('❓ Typing RAG question slowly (SLOW DEMO mode)...');
    
    // Type the question on the website with character delay
    const question = 'What is the main conclusion of this document about climate change?';
    
    // Use page.type for slower, more visible typing
    await questionInput.clear();
    await page.type('#ragQuestion', question, { delay: 100 }); // 100ms delay per character
    
    console.log(`📝 Question typed slowly: ${question}`);
    
    // Wait 2 seconds after typing the question
    console.log('⏳ Waiting 2 seconds after typing question...');
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Step 4: Click Ask button on the website - DEMO CLICK
    console.log('🔘 Clicking Ask button (SLOW DEMO mode)...');
    
    const askButton = page.locator('#ragQueryBtn');
    await expect(askButton).toBeVisible({ timeout: 30000 });
    await expect(askButton).toBeEnabled({ timeout: 15000 });
    
    await askButton.click();
    
    // Step 5: Wait for AI response - PATIENT WAIT
    console.log('⏳ Waiting for AI response to appear...');
    
    // Wait for results section to appear on website
    const resultsSection = page.locator('#resultsSection, .results-section, [data-testid="results"]');
    await expect(resultsSection).toBeVisible({ timeout: 60000 });
    
    // Step 6: Verify AI answer is rendered on the website
    console.log('🤖 Verifying AI answer (SLOW DEMO mode)...');
    
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
    
    console.log(`✅ AI Answer received: ${answerText ? answerText.substring(0, 100) : 'No answer'}...`);

    // Step 7: Verify source highlights are visible on the website
    console.log('📚 Verifying source highlights (SLOW DEMO mode)...');
    
    const sourceContainer = page.locator('.source-highlight');
    await expect(sourceContainer).toBeVisible({ timeout: 30000 });
    
    // Check that sources contain actual content
    const sourceText = await sourceContainer.textContent();
    expect(sourceText).toBeTruthy();
    expect(sourceText && sourceText.trim().length).toBeGreaterThan(5);
    
    // Ensure it's not just placeholder text
    expect(sourceText).not.toContain('Sources will appear here');
    expect(sourceText).not.toContain('No sources available');
    
    console.log(`✅ Sources found: ${sourceText ? sourceText.substring(0, 100) : 'No sources'}...`);

    // Step 8: Keep browser open for 5 seconds so output can be clearly seen
    console.log('⏸️ SLOW DEMO: Keeping browser open for 5 seconds to show results...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Take final screenshot for documentation
    console.log('📸 Taking final SLOW DEMO screenshot...');
    await page.screenshot({ 
      path: 'test-results/slow-demo-final-success.png',
      fullPage: true 
    });

    // Step 9: Log completion
    console.log('🎉 SLOW DEMO RAG flow test completed successfully!');
    
    // Additional verification: Check that the question is displayed on website
    const questionDisplay = page.locator('#researchTopicDisplay, .question-display, [data-testid="question-display"]');
    if (await questionDisplay.isVisible()) {
      const displayedQuestion = await questionDisplay.textContent();
      expect(displayedQuestion).toContain('climate change');
      console.log(`✅ Question displayed correctly: ${displayedQuestion}`);
    }
    
    // Close the website page cleanly
    await page.close();
  }, 120000); // Increase test timeout to 120 seconds

  test.afterEach(async () => {
    // Clean up the browser context after the demo test
    if (context) {
      console.log('🧹 Cleaning up browser context (SLOW DEMO mode)...');
      await context.close();
    }
  });
});
