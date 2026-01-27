# Playwright RAG Automation Documentation

## Overview
This project includes Playwright end-to-end testing for the RAG (Retrieval-Augmented Generation) PDF Upload & Q&A workflow.

## Files Added/Modified

### Configuration
- **`playwright.config.js`** - Minimal Playwright configuration using system Chrome
- **`package.json`** - Updated with test scripts

### Test Files  
- **`tests/rag-flow.spec.js`** - RAG workflow E2E tests
- **`tests/sample.pdf`** - Sample PDF for testing (climate change research)
- **`tests/sample.txt`** - Sample text file for error testing

### Frontend Updates
- **`index.html`** - Fixed duplicate ID issues, ensured proper selectors

## How to Run RAG Automation Tests

### Prerequisites
- System Chrome browser installed
- Backend running on http://localhost:8000
- Node.js and npm installed

### Commands
```bash
# Run all RAG tests
npm run test:rag

# Run all Playwright tests
npm run test

# View HTML test report
npx playwright show-report
```

## Test Coverage

### Main RAG Flow Test
1. **Open Application** - Navigate to http://localhost:8000
2. **Upload PDF** - Select and upload sample.pdf
3. **Enter Question** - Type question about climate change
4. **Submit Query** - Click Ask button
5. **Verify Answer** - Check AI answer is rendered
6. **Verify Sources** - Check source highlights are visible

### Additional Tests
- **Error Handling** - Non-PDF file upload
- **Validation** - Empty question submission

## Test Results
- ✅ **3/3 tests passed**
- ✅ **PDF upload working**
- ✅ **Question submission working** 
- ✅ **AI answer rendering**
- ✅ **Source highlighting visible**
- ✅ **Error handling functional**

## Technical Details

### Selectors Used
- `#pdfFile` - PDF upload input
- `#ragQuestion` - Question textarea
- `#ragQueryBtn` - Ask button
- `.answer-box` - AI answer container
- `.source-highlight` - Source highlights container

### Configuration Features
- **System Chrome** - Uses installed Chrome browser
- **Auto Web Server** - Starts FastAPI backend automatically
- **Screenshots** - Captures on test failure
- **HTML Reports** - Detailed test results

## Playwright Usage

Playwright provides reliable end-to-end browser automation for your AI Research Assistant:

### Benefits
- **UI Workflow Testing** - Validates complete user journeys
- **Cross-browser Support** - Works on Chrome, Firefox, Safari
- **Reliable Interactions** - Handles file uploads, form submissions
- **Visual Testing** - Screenshots and HTML reports
- **Fast Execution** - Parallel test running

### Best Practices
- Run tests before deploying changes
- Use descriptive test names
- Check element visibility before interaction
- Wait for async operations to complete
- Take screenshots for debugging

## Troubleshooting

### Common Issues
- **Backend not running** - Start FastAPI server first
- **Chrome not found** - Install Google Chrome browser
- **Element not visible** - Check if RAG forms are enabled
- **Timeout errors** - Increase wait times for slow operations

### Debug Tips
- Use `npx playwright test --headed` to watch tests run
- Check screenshots in `test-results/` folder
- Review HTML report for detailed error information
- Verify selectors match actual HTML elements

## Production Safety

This Playwright setup is designed to be:
- **Minimal** - Only essential dependencies
- **Stable** - Uses system browsers, no downloads
- **Safe** - No backend modifications
- **Portable** - Works across different environments
