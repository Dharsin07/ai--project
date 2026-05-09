import { test, expect } from '@playwright/test';

test.describe('Basic Application Tests', () => {
  test('should load main page', async ({ page }) => {
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    
    // Check that page loads successfully
    await expect(page).toHaveTitle(/Research Assistant/);
    
    // Check for key elements with robust selectors
    await expect(page.locator('.header')).toBeVisible();
    await expect(page.locator('#liveResearchQuery')).toBeVisible();
    await expect(page.locator('#liveResearchBtn')).toBeVisible();
  });

  test('should have RAG upload elements', async ({ page }) => {
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    
    // Check for RAG-specific elements using robust selectors
    await expect(page.locator('input[type="file"]')).toBeVisible();
    await expect(page.locator('#ragQuestion')).toBeVisible();
    await expect(page.locator('#ragQueryBtn')).toBeVisible();
  });

  test('should have answer containers', async ({ page }) => {
    await page.goto('http://localhost:8000');
    await page.waitForLoadState('networkidle');
    
    // Check for answer and source containers (initially hidden)
    const answerBox = page.locator('.answer-box');
    const sourceHighlight = page.locator('.source-highlight');
    
    // Elements should exist in DOM
    await expect(answerBox).toHaveCount(1);
    await expect(sourceHighlight).toHaveCount(1);
  });
});
