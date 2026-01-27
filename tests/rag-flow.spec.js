import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('RAG PDF Upload & Q&A Workflow', () => {
  test('should complete RAG flow: upload PDF, ask question, get answer with sources', async ({ page }) => {
    // Step a: Open the application
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    
    // Step b: Upload PDF
    console.log('Uploading PDF...');
    const fileInput = page.locator('#pdfFile');
    await expect(fileInput).toBeVisible();
    
    const pdfPath = path.join(__dirname, 'sample.pdf');
    await fileInput.setInputFiles(pdfPath);
    
    // Submit upload form
    const uploadButton = page.locator('#uploadBtn');
    await uploadButton.click();
    
    // Wait for upload to complete
    await page.waitForTimeout(3000);
    
    // Step c: Enter question
    console.log('Entering question...');
    const questionInput = page.locator('#ragQuestion');
    await expect(questionInput).toBeVisible();
    
    await questionInput.fill('What is the main conclusion of this document about climate change?');
    
    // Step d: Click Ask button
    console.log('Submitting question...');
    const askButton = page.locator('#ragQueryBtn');
    await expect(askButton).toBeVisible();
    await askButton.click();
    
    // Step e: Verify AI answer is rendered
    console.log('Verifying AI answer...');
    const resultsSection = page.locator('#resultsSection');
    await expect(resultsSection).toBeVisible({ timeout: 15000 });
    
    const answerBox = page.locator('.answer-box');
    await expect(answerBox).toBeVisible();
    
    // Check that answer contains content
    const answerText = await answerBox.textContent();
    expect(answerText && answerText.trim()).toBeTruthy();
    expect(answerText && answerText.length).toBeGreaterThan(10);
    
    console.log('AI Answer rendered successfully');
    
    // Step f: Verify source highlights are visible
    console.log('Verifying source highlights...');
    const sourceHighlight = page.locator('.source-highlight');
    await expect(sourceHighlight).toBeVisible();
    
    // Check that sources contain content
    const sourceText = await sourceHighlight.textContent();
    expect(sourceText && sourceText.trim()).toBeTruthy();
    
    console.log('Source highlights visible successfully');
    
    // Take screenshot for documentation
    await page.screenshot({ path: 'test-results/rag-flow-success.png', fullPage: true });
    
    console.log('✅ RAG flow test completed successfully');
  });

  test('should handle PDF upload gracefully', async ({ page }) => {
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    
    // Try to upload non-PDF file
    const fileInput = page.locator('#pdfFile');
    await fileInput.setInputFiles(path.join(__dirname, 'sample.txt'));
    
    const uploadButton = page.locator('#uploadBtn');
    await uploadButton.click();
    
    // Wait to see behavior
    await page.waitForTimeout(2000);
    
    // Take screenshot for debugging
    await page.screenshot({ path: 'test-results/rag-upload-error.png', fullPage: true });
  });

  test('should handle empty question submission', async ({ page }) => {
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    
    // Upload PDF first
    const fileInput = page.locator('#pdfFile');
    await fileInput.setInputFiles(path.join(__dirname, 'sample.pdf'));
    
    const uploadButton = page.locator('#uploadBtn');
    await uploadButton.click();
    await page.waitForTimeout(3000);
    
    // Try empty question
    const questionInput = page.locator('#ragQuestion');
    await questionInput.fill('');
    
    const askButton = page.locator('#ragQueryBtn');
    await askButton.click();
    
    await page.waitForTimeout(2000);
    
    // Take screenshot for debugging
    await page.screenshot({ path: 'test-results/rag-empty-question.png', fullPage: true });
  });
});
