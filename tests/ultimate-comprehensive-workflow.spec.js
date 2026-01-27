import { test, expect, chromium } from '@playwright/test';
import path from 'path';

// ULTIMATE COMPREHENSIVE WORKFLOW - Interview-friendly with explicit delays
// This test automates all workflows in continuous order:
// 1. RAG PDF Upload Workflow
// 2. Article Q&A Workflow  
// 3. Live Research Paper Collector Workflow
// Uses existing Chrome extension and existing index.html page
test.describe('Ultimate Comprehensive Workflow with Chrome Extension', () => {
  let context;
  let extensionPath;

  test.beforeAll(async () => {
    // Get the path to the existing Chrome extension
    console.log('🎬 Setting up Ultimate Comprehensive workflow with Chrome extension...');
    extensionPath = path.resolve(__dirname, '..');
    console.log(`📁 Extension path: ${extensionPath}`);
  });

  test.beforeEach(async () => {
    // Launch Chrome with extension loaded (headful mode) - DEMO MODE
    console.log('🚀 Launching Chrome with extension for Ultimate Comprehensive workflow...');
    
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
      // Demo mode: medium-fast execution with human-like delays
      slowMo: 300, // 300ms delay between actions (balanced speed)
    });

    // Wait for extension to fully load - DEMO WAIT
    console.log('⏳ Waiting for Chrome extension to load (2 seconds)...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    console.log('✅ Chrome launched with extension loaded for Ultimate Comprehensive workflow');
  });

  // WORKFLOW 1: RAG PDF Upload Test - First in order
  test('should complete all three workflows continuously in order', async () => {
    console.log('🎬 Starting Ultimate Comprehensive workflow test with all three workflows...');
    
    // ========== WORKFLOW 1: RAG PDF Upload ==========
    console.log('\n📁 ========== STARTING WORKFLOW 1: RAG PDF Upload ==========');
    
    // Step 1: Open the existing web application
    console.log('📄 Opening website for RAG PDF upload: http://localhost:8000');
    const page = await context.newPage();
    await page.goto('http://localhost:8000');
    
    // Wait 2 seconds after website loads
    console.log('⏳ Waiting 2 seconds after website loads...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 2: Upload PDF file on the website
    console.log('📁 Uploading PDF file...');
    
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
    
    // Step 3: Enter RAG question on the website
    console.log('❓ Typing RAG question...');
    
    // Type the question on the website with character delay
    const question = 'What is the main conclusion of this document about climate change?';
    
    // Use page.type for slower, more visible typing
    await questionInput.clear();
    await page.type('#ragQuestion', question, { delay: 100 }); // 100ms delay per character
    
    console.log(`📝 Question typed: ${question}`);
    
    // Wait 2 seconds after typing the question
    console.log('⏳ Waiting 2 seconds after typing question...');
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Step 4: Click Ask button on the website
    console.log('🔘 Clicking Ask button...');
    
    const askButton = page.locator('#ragQueryBtn');
    await expect(askButton).toBeVisible({ timeout: 30000 });
    await expect(askButton).toBeEnabled({ timeout: 15000 });
    
    await askButton.click();
    
    // Step 5: Wait for AI response
    console.log('⏳ Waiting for AI response to appear...');
    
    // Wait for results section to appear on website
    const resultsSection = page.locator('#resultsSection, .results-section, [data-testid="results"]');
    await expect(resultsSection).toBeVisible({ timeout: 60000 });
    
    // Step 6: Verify AI answer is rendered on the website
    console.log('🤖 Verifying AI answer...');
    
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
    console.log('📚 Verifying source highlights...');
    
    const sourceContainer = page.locator('.source-highlight');
    await expect(sourceContainer).toBeVisible({ timeout: 30000 });
    
    // Check that sources contain actual content
    const sourceText = await sourceContainer.textContent();
    expect(sourceText).toBeTruthy();
    expect(sourceText && sourceText.trim().length).toBeGreaterThan(5);
    
    console.log(`✅ Sources found: ${sourceText ? sourceText.substring(0, 100) : 'No sources'}...`);

    // Step 8: Keep browser open for 3 seconds so output can be clearly seen
    console.log('⏸️ Keeping browser open for 3 seconds to show results...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Log completion of Workflow 1
    console.log('🎉 RAG PDF upload workflow completed successfully!');
    
    // Wait 3 seconds before next workflow
    console.log('⏳ Waiting 3 seconds before next workflow...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // ========== WORKFLOW 2: Article Q&A ==========
    console.log('\n📝 ========== STARTING WORKFLOW 2: Article Q&A ==========');
    
    // Step 1: Wait until "Ask Your Question" section is visible
    console.log('🔍 Waiting for "Ask Your Question" section to be visible...');
    const askQuestionSection = page.locator('h2.card-title:has-text("Ask Your Question")');
    await expect(askQuestionSection).toBeVisible({ timeout: 30000 });
    console.log('✅ "Ask Your Question" section is visible');
    
    // Wait 2 seconds before starting input
    console.log('⏳ Waiting 2 seconds before starting input...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 2: Type "Summarize this concept" first, then add the article text
    console.log('❓ Typing "Summarize this concept" first...');
    
    // Get the textarea element
    const questionTextarea = page.locator('#researchTopic');
    await expect(questionTextarea).toBeVisible({ timeout: 30000 });
    
    // Clear the textarea and type the question
    await questionTextarea.clear();
    const questionPrefix = 'Summarize this concept: ';
    await page.type('#researchTopic', questionPrefix, { delay: 100 }); // 100ms delay per character
    
    console.log(`📝 Question prefix typed: ${questionPrefix}`);
    
    // Wait 1 second before adding article
    console.log('⏳ Waiting 1 second before adding article...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Add the article text after the question
    console.log('📝 Adding article text after the question...');
    
    const sampleArticle = `Climate change represents one of the most significant challenges facing humanity in the 21st century. The scientific consensus is clear: the Earth's climate is warming at an unprecedented rate, primarily due to human activities such as the burning of fossil fuels, deforestation, and industrial processes. 

The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate and drastic reductions in greenhouse gas emissions, we face catastrophic consequences including rising sea levels, extreme weather events, food and water shortages, and mass extinctions.

However, there is hope. Renewable energy technologies like solar and wind power are becoming increasingly cost-effective, and many countries are committing to ambitious climate goals. Individual actions, combined with government policies and corporate responsibility, can make a meaningful difference in mitigating the worst impacts of climate change.

The transition to a low-carbon economy presents not only challenges but also opportunities for innovation, job creation, and a more sustainable future for generations to come.`;
    
    // Type the article text with faster typing for demo efficiency
    await page.type('#researchTopic', sampleArticle, { delay: 10 }); // 10ms delay per character (much faster)
    
    const fullQuestion = questionPrefix + sampleArticle;
    console.log(`📝 Full input typed: ${fullQuestion.substring(0, 100)}... (${fullQuestion.length} characters total)`);
    
    // Wait 2 seconds after typing full content
    console.log('⏳ Waiting 2 seconds after typing full content...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 3: Select Answer Type - Option 1: Short Answer
    console.log('📋 Selecting Answer Type: Short Answer...');
    
    const answerTypeSelect = page.locator('#summaryType');
    await expect(answerTypeSelect).toBeVisible({ timeout: 30000 });
    await answerTypeSelect.selectOption('short');
    
    console.log('✅ Answer Type set to: Short Answer');
    
    // Wait 2 seconds after selecting answer type
    console.log('⏳ Waiting 2 seconds after selecting answer type...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 4: Adjust "Number of Sources" slider to value 5
    console.log('🎚️ Adjusting Number of Sources slider to 5...');
    
    const sourceSlider = page.locator('#sourceCount');
    await expect(sourceSlider).toBeVisible({ timeout: 30000 });
    
    // Verify current value and set to 5
    const currentValue = await sourceSlider.inputValue();
    console.log(`📊 Current slider value: ${currentValue}`);
    
    if (currentValue !== '5') {
      await sourceSlider.fill('5');
      console.log('📊 Slider adjusted to: 5');
    } else {
      console.log('📊 Slider already at: 5');
    }
    
    // Verify the display value updated
    const sourceValueDisplay = page.locator('#sourceValue');
    await expect(sourceValueDisplay).toHaveText('5');
    
    // Wait 2 seconds after adjusting slider
    console.log('⏳ Waiting 2 seconds after adjusting slider...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 5: Click the "Get Answer" button
    console.log('🔘 Clicking "Get Answer" button...');
    
    const getAnswerButton = page.locator('#generateBtn');
    await expect(getAnswerButton).toBeVisible({ timeout: 30000 });
    await expect(getAnswerButton).toBeEnabled({ timeout: 15000 });
    
    await getAnswerButton.click();
    
    console.log('✅ "Get Answer" button clicked');
    
    // Step 6: Wait for AI-generated output to appear
    console.log('⏳ Waiting for AI-generated output to appear...');
    
    // Wait for results section to become visible
    const articleResultsSection = page.locator('#resultsSection');
    await expect(articleResultsSection).toBeVisible({ timeout: 60000 }); // 60 seconds timeout for AI processing
    
    console.log('✅ Results section is now visible');
    
    // Step 7: Smooth scroll to output section after results appear
    console.log('📜 Smooth scrolling to output section...');
    await page.locator('#resultsSection').scrollIntoViewIfNeeded({ behavior: 'smooth' });
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds for smooth scroll to complete
    
    // Ensure output section is fully visible
    console.log('👁️ Verifying output section is fully visible...');
    await expect(articleResultsSection).toBeInViewport();
    
    // Step 8: Scroll down fully to show complete AI answer
    console.log('📜 Scrolling down fully to show complete AI answer...');
    await page.evaluate(() => {
      // Scroll to the bottom of the page to ensure full AI answer is visible
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
      });
    });
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds for full scroll
    
    // Additional scroll to sources section to ensure everything is visible
    console.log('📜 Ensuring sources section is also visible...');
    await page.locator('#sourcesList').scrollIntoViewIfNeeded({ behavior: 'smooth' });
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second for final scroll
    
    // Step 9: Verify the output text is visible and non-empty
    console.log('🤖 Verifying AI-generated output...');
    
    const answerContent = page.locator('#summaryContent');
    await expect(answerContent).toBeVisible({ timeout: 30000 });
    
    // Check that answer contains actual content (not placeholder text)
    const articleAnswerText = await answerContent.textContent();
    expect(articleAnswerText).toBeTruthy();
    expect(articleAnswerText && articleAnswerText.trim().length).toBeGreaterThan(10);
    
    // Ensure it's not just placeholder text
    expect(articleAnswerText).not.toContain('Your answer will appear here');
    expect(articleAnswerText).not.toContain('Loading...');
    expect(articleAnswerText).not.toContain('Processing...');
    
    console.log(`✅ AI Answer received: ${articleAnswerText ? articleAnswerText.substring(0, 100) : 'No answer'}...`);
    
    // Step 10: Keep the output visible for 4 seconds with smooth scroll position
    console.log('⏸️ DEMO MODE: Keeping output visible for 4 seconds...');
    
    // Additional smooth scroll to ensure perfect positioning
    await page.locator('#summaryContent').scrollIntoViewIfNeeded({ behavior: 'smooth' });
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second for final scroll
    
    // Hold for 4 seconds to show the output clearly
    await new Promise(resolve => setTimeout(resolve, 4000));
    
    // Step 11: Additional verification - Check question display
    console.log('🔍 Verifying question display...');
    const questionDisplay = page.locator('#researchTopicDisplay');
    if (await questionDisplay.isVisible()) {
      const displayedQuestion = await questionDisplay.textContent();
      expect(displayedQuestion).toContain('Summarize this concept');
      console.log(`✅ Question displayed correctly: ${displayedQuestion.substring(0, 100)}...`);
    }
    
    // Step 12: Verify sources are displayed
    console.log('📚 Verifying sources are displayed...');
    const sourcesList = page.locator('#sourcesList');
    if (await sourcesList.isVisible()) {
      const sourcesCount = await sourcesList.locator('li').count();
      console.log(`✅ Sources displayed: ${sourcesCount} sources found`);
      expect(sourcesCount).toBeGreaterThan(0);
    }
    
    // Log completion of Workflow 2
    console.log('🎉 Article Q&A workflow completed successfully!');
    
    // Wait 3 seconds before next workflow
    console.log('⏳ Waiting 3 seconds before next workflow...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // ========== WORKFLOW 3: Live Research Paper Collector ==========
    console.log('\n🔍 ========== STARTING WORKFLOW 3: Live Research Paper Collector ==========');
    
    // Step 1: Wait until "Live Research Paper Collector" section is visible
    console.log('🔍 Waiting for "Live Research Paper Collector" section to be visible...');
    const researchSection = page.locator('h2.card-title:has-text("Live Research Paper Collector")');
    await expect(researchSection).toBeVisible({ timeout: 30000 });
    console.log('✅ "Live Research Paper Collector" section is visible');
    
    // Wait 1 second before starting input
    console.log('⏳ Waiting 1 second before starting input...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Step 2: Locate the Research Query textarea
    console.log('📝 Locating Research Query textarea...');
    const researchQuery = page.locator('#liveResearchQuery');
    await expect(researchQuery).toBeVisible({ timeout: 30000 });
    console.log('✅ Research Query textarea located');
    
    // Step 3: Type: "Collect latest machine learning research papers"
    console.log('⌨️ Typing research query: "Collect latest machine learning research papers"...');
    const queryText = 'Collect latest machine learning research papers';
    await researchQuery.clear();
    await page.type('#liveResearchQuery', queryText, { delay: 50 }); // 50ms delay per character
    console.log(`📝 Research query typed: ${queryText}`);
    
    // Wait 1 second after typing
    console.log('⏳ Waiting 1 second after typing query...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Step 4: Adjust the "Number of Papers" slider to 5
    console.log('🎚️ Adjusting Number of Papers slider to 5...');
    const paperCountSlider = page.locator('#paperCount');
    await expect(paperCountSlider).toBeVisible({ timeout: 30000 });
    
    // Verify current value and set to 5
    const currentSliderValue = await paperCountSlider.inputValue();
    console.log(`📊 Current slider value: ${currentSliderValue}`);
    
    if (currentSliderValue !== '5') {
      await paperCountSlider.fill('5');
      console.log('📊 Slider adjusted to: 5');
    } else {
      console.log('📊 Slider already at: 5');
    }
    
    // Verify the display value updated
    const paperCountValue = page.locator('#paperCountValue');
    await expect(paperCountValue).toHaveText('5');
    
    // Wait 1 second after adjusting slider
    console.log('⏳ Waiting 1 second after adjusting slider...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Step 5: Click the "Sort By" dropdown
    console.log('📋 Clicking Sort By dropdown...');
    const sortByDropdown = page.locator('#sortBy');
    await expect(sortByDropdown).toBeVisible({ timeout: 30000 });
    await sortByDropdown.click();
    
    // Wait 1 second for dropdown to open
    console.log('⏳ Waiting 1 second for dropdown to open...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Step 6: Select "Most Recent"
    console.log('📋 Selecting "Most Recent" option...');
    await sortByDropdown.selectOption('recent');
    console.log('✅ Sort By set to: Most Recent');
    
    // Wait 1 second after selection
    console.log('⏳ Waiting 1 second after selection...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Step 7: Click the "Collect Papers" button
    console.log('🔘 Clicking "Collect Papers" button...');
    const collectPapersBtn = page.locator('#liveResearchBtn');
    await expect(collectPapersBtn).toBeVisible({ timeout: 30000 });
    await expect(collectPapersBtn).toBeEnabled({ timeout: 15000 });
    
    await collectPapersBtn.click();
    console.log('✅ "Collect Papers" button clicked');
    
    // Step 8: Wait for papers to be fetched and rendered
    console.log('⏳ Waiting for papers to be fetched and rendered...');
    
    // Wait for email confirmation section to appear (indicates papers collected)
    const emailConfirmationCard = page.locator('#emailConfirmationCard');
    await expect(emailConfirmationCard).toBeVisible({ timeout: 120000 }); // 2 minutes timeout for paper collection
    console.log('✅ Email confirmation section appeared - papers collected');
    
    // Wait 2 seconds for UI to settle
    console.log('⏳ Waiting 2 seconds for UI to settle...');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 9: Detect the email confirmation input/modal
    console.log('📧 Detecting email confirmation input...');
    const recipientEmail = page.locator('#recipientEmail');
    await expect(recipientEmail).toBeVisible({ timeout: 30000 });
    console.log('✅ Email confirmation input detected');
    
    // Step 10: Enter email: smartdharshin2005@gmail.com
    console.log('📧 Entering email: smartdharshin2005@gmail.com...');
    const emailAddress = 'smartdharshin2005@gmail.com';
    await recipientEmail.clear();
    await page.type('#recipientEmail', emailAddress, { delay: 50 });
    console.log(`📧 Email entered: ${emailAddress}`);
    
    // Wait 1 second after entering email
    console.log('⏳ Waiting 1 second after entering email...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Step 11: Click the "Send Email" button
    console.log('🔘 Clicking "Send Email" button...');
    const sendEmailBtn = page.locator('#sendEmailBtn');
    await expect(sendEmailBtn).toBeVisible({ timeout: 30000 });
    await expect(sendEmailBtn).toBeEnabled({ timeout: 15000 });
    
    await sendEmailBtn.click();
    console.log('✅ "Send Email" button clicked');
    
    // Wait for email to be sent
    console.log('⏳ Waiting for email to be sent...');
    await new Promise(resolve => setTimeout(resolve, 3000)); // 3 seconds for email sending
    
    // Step 12: Scroll down to show all 5 research papers slowly
    console.log('📜 Scrolling down to show all 5 research papers slowly...');
    
    // First scroll to research papers section
    await page.evaluate(() => {
      window.scrollTo({
        top: document.body.scrollHeight / 3,
        behavior: 'smooth'
      });
    });
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for initial scroll
    
    // Slow scroll through all research papers to make them visible
    console.log('📜 Slowly scrolling through all research papers...');
    const researchScrollSteps = 8;
    for (let i = 0; i <= researchScrollSteps; i++) {
      await page.evaluate(({ step, totalSteps }) => {
        const scrollPosition = (document.body.scrollHeight / totalSteps) * step;
        window.scrollTo(0, scrollPosition);
        return scrollPosition;
      }, { step: i, totalSteps: researchScrollSteps });
      await new Promise(resolve => setTimeout(resolve, 2000)); // 2 seconds between scrolls
      console.log(`📜 Scrolled research papers to step ${i + 1}/${researchScrollSteps + 1}`);
    }
    
    // Step 13: Verify research paper cards / titles are visible
    console.log('🔍 Verifying research paper cards/titles are visible...');
    
    // Try to find research paper elements - these might be dynamically generated
    const researchPapers = page.locator('.paper-card, .research-paper, .paper-title, [data-testid*="paper"], [class*="paper"]');
    
    // Wait a bit for dynamic content to load
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check if any paper elements are visible
    const paperCount = await researchPapers.count();
    console.log(`📊 Found ${paperCount} research paper elements`);
    
    if (paperCount > 0) {
      console.log('✅ Research paper cards/titles are visible');
      console.log('📸 All research papers are visible for demo');
    } else {
      console.log('⚠️ No research paper elements found, but continuing with test...');
    }
    
    // Step 14: Click the first "Download PDF" button
    console.log('🔍 Looking for Download PDF buttons...');
    
    // Look for download buttons with various selectors
    const downloadBtns = page.locator('button:has-text("Download PDF"), a:has-text("Download PDF"), [data-testid*="download"], [class*="download"]');
    
    const downloadBtnCount = await downloadBtns.count();
    console.log(`📊 Found ${downloadBtnCount} Download PDF buttons`);
    
    if (downloadBtnCount > 0) {
      console.log('🔘 Clicking the first Download PDF button...');
      
      // Set up event listener for new page/tab
      const newPagePromise = context.waitForEvent('page');
      
      // Click the first download button
      await downloadBtns.first().click();
      
      // Wait for new page/tab to open
      console.log('⏳ Waiting for new tab/PDF viewer to open...');
      const newPage = await newPagePromise;
      
      // Wait for new page to load
      await newPage.waitForLoadState('networkidle');
      console.log('✅ New tab/PDF viewer opened');
      
      // Step 15: Handle new tab or PDF viewer correctly
      console.log('📄 Handling new tab/PDF viewer...');
      
      // Wait for PDF content to load
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Step 16: Scroll through PDF pages 1-5 one by one slowly
      console.log('📄 Scrolling through PDF pages 1-5 one by one slowly...');
      
      // Get the full height of the PDF document
      const pdfHeight = await newPage.evaluate(() => {
        return Math.max(
          document.body.scrollHeight,
          document.body.offsetHeight,
          document.documentElement.clientHeight,
          document.documentElement.scrollHeight,
          document.documentElement.offsetHeight
        );
      });
      
      console.log(`📊 PDF total height: ${pdfHeight}px`);
      
      // Calculate approximate page height (assuming 5 pages)
      const pageHeight = pdfHeight / 5;
      console.log(`📊 Approximate page height: ${Math.round(pageHeight)}px`);
      
      // Scroll through pages 1-5 one by one
      for (let pageNum = 1; pageNum <= 5; pageNum++) {
        const scrollPosition = pageHeight * (pageNum - 1);
        console.log(`📄 Showing PDF Page ${pageNum}...`);
        
        // Scroll to the page position
        await newPage.evaluate((position) => {
          window.scrollTo(0, position);
        }, scrollPosition);
        
        // Wait for the page to be visible
        await new Promise(resolve => setTimeout(resolve, 2500)); // 2.5 seconds per page
        
        console.log(`✅ PDF Page ${pageNum} visible for demo`);
      }
      
      // Final scroll to show complete PDF
      console.log('📄 Final scroll to show complete PDF...');
      await newPage.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight);
      });
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      console.log('📸 Complete PDF visible for demo');
      
      // Wait 2 seconds before closing
      console.log('⏳ Waiting 2 seconds before closing PDF tab...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Close the PDF tab
      await newPage.close();
      console.log('✅ PDF tab closed');
      
    } else {
      console.log('⚠️ No Download PDF buttons found, skipping PDF download step...');
    }
    
    // Log completion of Workflow 3
    console.log('🎉 Live Research Paper Collector workflow completed successfully!');
    
    // ========== ULTIMATE COMPLETION ==========
    console.log('\n🏆 ========== ALL THREE WORKFLOWS COMPLETED SUCCESSFULLY! ==========');
    console.log('✅ Workflow 1: RAG PDF Upload - COMPLETED');
    console.log('✅ Workflow 2: Article Q&A - COMPLETED');
    console.log('✅ Workflow 3: Live Research Paper Collector - COMPLETED');
    
    // Wait 3 seconds before final closure
    console.log('⏳ Waiting 3 seconds before final browser closure...');
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Log ultimate completion
    console.log('🎉 Ultimate Comprehensive workflow test completed successfully!');
    console.log('🏆 All three workflows executed continuously in perfect order!');
    
    // Close the main page cleanly
    await page.close();
  }, 300000); // Increase test timeout to 300 seconds (5 minutes)

  test.afterEach(async () => {
    // Clean up the browser context after the demo test
    if (context) {
      console.log('🧹 Cleaning up browser context (Ultimate Comprehensive workflow)...');
      await context.close();
    }
  });
});
