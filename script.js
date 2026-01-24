// Autonomous Research Assistant - JavaScript Module
// Main application logic for form handling, API simulation, and UI interactions

class ResearchAssistant {
  constructor() {
    this.initElements();
    this.initEventListeners();
    this.initTheme();
  }

  // Initialize DOM elements
  initElements() {
    // Form elements
    this.researchForm = document.getElementById('researchForm');
    this.researchTopic = document.getElementById('researchTopic');
    this.summaryType = document.getElementById('summaryType');
    this.sourceCount = document.getElementById('sourceCount');
    this.sourceValue = document.getElementById('sourceValue');
    this.generateBtn = document.getElementById('generateBtn');
    this.btnText = this.generateBtn.querySelector('.btn-text');
    this.btnLoader = this.generateBtn.querySelector('.btn-loader');

    // RAG elements
    this.ragUploadForm = document.getElementById('ragUploadForm');
    this.pdfFile = document.getElementById('pdfFile');
    this.uploadBtn = document.getElementById('uploadBtn');
    this.uploadBtnText = this.uploadBtn.querySelector('.btn-text');
    this.uploadBtnLoader = this.uploadBtn.querySelector('.btn-loader');
    this.uploadStatus = document.getElementById('uploadStatus');
    
    this.ragQueryForm = document.getElementById('ragQueryForm');
    this.ragQuestion = document.getElementById('ragQuestion');
    this.ragQueryBtn = document.getElementById('ragQueryBtn');
    this.ragQueryBtnText = this.ragQueryBtn.querySelector('.btn-text');
    this.ragQueryBtnLoader = this.ragQueryBtn.querySelector('.btn-loader');

    // Results elements
    this.resultsSection = document.getElementById('resultsSection');
    this.researchTopicDisplay = document.getElementById('researchTopicDisplay');
    this.summaryContent = document.getElementById('summaryContent');
    this.sourcesList = document.getElementById('sourcesList');
    this.copyBtn = document.getElementById('copyBtn');
    this.downloadBtn = document.getElementById('downloadBtn');
    this.clearBtn = document.getElementById('clearBtn');

    // Theme elements
    this.themeToggle = document.getElementById('themeToggle');
    this.themeIcon = this.themeToggle.querySelector('.theme-icon');
    
    // Initialize RAG question as disabled until PDF is uploaded
    this.ragQuestion.disabled = true;
    this.ragQueryBtn.disabled = true;
    this.ragQuestion.placeholder = "Please upload a PDF first...";
  }

  // Initialize event listeners
  initEventListeners() {
    // Form submission
    this.researchForm.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleFormSubmit();
    });

    // RAG upload form
    this.ragUploadForm.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleRAGUpload();
    });

    // RAG query form
    this.ragQueryForm.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleRAGQuery();
    });

    // Source count slider
    this.sourceCount.addEventListener('input', (e) => {
      this.sourceValue.textContent = e.target.value;
    });

    // Theme toggle
    this.themeToggle.addEventListener('click', () => {
      this.toggleTheme();
    });

    // Results actions
    this.copyBtn.addEventListener('click', () => {
      this.copySummary();
    });

    this.downloadBtn.addEventListener('click', () => {
      this.downloadPDF();
    });

    this.clearBtn.addEventListener('click', () => {
      this.clearResults();
    });

    // Input validation
    this.researchTopic.addEventListener('input', () => {
      this.validateInput();
    });
  }

  // Initialize theme
  initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    this.updateThemeIcon(savedTheme);
  }

  // Toggle between light and dark theme
  toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    this.updateThemeIcon(newTheme);
  }

  // Update theme icon
  updateThemeIcon(theme) {
    this.themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
  }

  // Validate form input
  validateInput() {
    const topic = this.researchTopic.value.trim();
    const isValid = topic.length >= 10;
    
    this.generateBtn.disabled = !isValid;
    
    if (topic.length > 0 && topic.length < 10) {
      this.researchTopic.style.borderColor = 'var(--warning)';
    } else if (topic.length >= 10) {
      this.researchTopic.style.borderColor = 'var(--success)';
    } else {
      this.researchTopic.style.borderColor = 'var(--border)';
    }
  }

  // Handle form submission
  async handleFormSubmit() {
    const topic = this.researchTopic.value.trim();
    const summaryType = this.summaryType.value;
    const sourceCount = parseInt(this.sourceCount.value);

    console.log('📝 Form submitted:', { topic, summaryType, sourceCount });

    if (!topic) {
      this.showError('Please enter a research topic');
      return;
    }

    if (topic.length < 10) {
      this.showError('Research topic must be at least 10 characters long');
      return;
    }

    // Show loading state
    this.setLoadingState(true);

    try {
      // Call real API only - NO MOCK FALLBACK
      console.log('🔄 Calling real Gemini API...');
      const response = await this.callRealAPI(topic, summaryType, sourceCount);
      console.log('✅ Real Gemini API call successful');
      
      // Display results
      this.displayResults(response);
      
    } catch (error) {
      console.error('❌ API call failed:', error);
      this.showError('Failed to connect to AI service. Please check if the backend is running on http://localhost:8000');
    } finally {
      this.setLoadingState(false);
    }
  }

  // Set loading state
  setLoadingState(isLoading) {
    if (isLoading) {
      this.generateBtn.disabled = true;
      this.btnText.style.display = 'none';
      this.btnLoader.style.display = 'flex';
      this.researchTopic.disabled = true;
      this.summaryType.disabled = true;
      this.sourceCount.disabled = true;
    } else {
      this.generateBtn.disabled = false;
      this.btnText.style.display = 'inline';
      this.btnLoader.style.display = 'none';
      this.researchTopic.disabled = false;
      this.summaryType.disabled = false;
      this.sourceCount.disabled = false;
    }
  }

  // Simulate API call
  simulateAPICall() {
    return new Promise((resolve) => {
      const delay = 2000 + Math.random() * 2000; // 2-4 seconds
      setTimeout(resolve, delay);
    });
  }

  // Call real API
  async callRealAPI(topic, summaryType, sourceCount) {
    try {
      console.log('🚀 Calling real Gemini API with:', { topic, summaryType, sourceCount });
      
      // First test if backend is reachable
      console.log('🔍 Testing backend connection...');
      const healthResponse = await fetch('http://localhost:8000/health');
      console.log('📡 Health check status:', healthResponse.status);
      
      if (!healthResponse.ok) {
        throw new Error('Backend health check failed');
      }
      
      const healthData = await healthResponse.json();
      console.log('💚 Backend health:', healthData);
      
      // Now call the research endpoint
      const response = await fetch('http://localhost:8000/research', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic: topic,
          summary_type: summaryType,
          source_count: sourceCount
        })
      });

      console.log('📡 Research API Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ API Error Response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      console.log('✅ Real Gemini API Response data:', data);
      console.log('📝 Summary preview:', data.summary ? data.summary.substring(0, 100) + '...' : 'No summary');
      
      return {
        topic: data.topic,
        summary: data.summary,
        sources: data.sources,
        keywords: data.keywords,
        timestamp: data.timestamp
      };
    } catch (error) {
      console.error('❌ Gemini API call failed:', error);
      console.error('🔍 Error details:', error.message);
      
      // Additional debugging info
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        console.error('🌐 Network error - Backend may not be running on http://localhost:8000');
      }
      
      throw error;
    }
  }

  // Extract keywords from summary
  extractKeywords(text) {
    const commonWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'];
    
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 3 && !commonWords.includes(word));
    
    const wordFreq = {};
    words.forEach(word => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });
    
    return Object.entries(wordFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([word]) => word.charAt(0).toUpperCase() + word.slice(1));
  }

  // Display results
  displayResults(response) {
    // Display research topic
    this.researchTopicDisplay.textContent = response.topic;

    // Display summary with highlighted keywords
    let summaryHTML = response.summary;
    response.keywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      summaryHTML = summaryHTML.replace(regex, `<span class="keyword">${keyword}</span>`);
    });
    this.summaryContent.innerHTML = summaryHTML;

    // Display sources
    this.sourcesList.innerHTML = '';
    response.sources.forEach(source => {
      const li = document.createElement('li');
      li.innerHTML = `
        <strong>${source.title}</strong><br>
        <span style="color: var(--text-secondary); font-size: var(--font-size-sm);">${source.authors}</span><br>
        <a href="${source.url}" target="_blank" rel="noopener noreferrer">${source.url}</a>
      `;
      this.sourcesList.appendChild(li);
    });

    // Show results section with animation
    this.resultsSection.style.display = 'block';
    this.resultsSection.classList.add('fade-in');
    
    // Scroll to results
    setTimeout(() => {
      this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }

  // Copy summary to clipboard
  async copySummary() {
    const summaryText = this.summaryContent.textContent;
    
    try {
      await navigator.clipboard.writeText(summaryText);
      this.showSuccess('Answer copied to clipboard!');
    } catch (err) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = summaryText;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      this.showSuccess('Answer copied to clipboard!');
    }
  }

  // Download as PDF (UI only - shows message)
  downloadPDF() {
    this.showInfo('PDF download feature would be implemented with a backend service. For now, you can copy the answer and save it manually.');
  }

  // Clear results
  clearResults() {
    this.resultsSection.classList.add('fade-out');
    
    setTimeout(() => {
      this.resultsSection.style.display = 'none';
      this.resultsSection.classList.remove('fade-in', 'fade-out');
      
      // Reset form
      this.researchForm.reset();
      this.sourceValue.textContent = '5';
      this.validateInput();
      
      // Scroll to top
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 300);
  }

  // Show success message
  showSuccess(message) {
    this.showToast(message, 'success');
  }

  // Show error message
  showError(message) {
    this.showToast(message, 'error');
  }

  // Show info message
  showInfo(message) {
    this.showToast(message, 'info');
  }

  // Show toast notification
  showToast(message, type = 'info') {
    // Remove existing toast if any
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
      existingToast.remove();
    }

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add styles
    toast.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 12px 20px;
      border-radius: 8px;
      color: white;
      font-weight: 500;
      z-index: 10000;
      transform: translateX(100%);
      transition: transform 0.3s ease-in-out;
      max-width: 300px;
      box-shadow: var(--shadow-lg);
    `;

    // Set background color based on type
    const colors = {
      success: 'var(--success)',
      error: 'var(--error)',
      info: 'var(--info)',
      warning: 'var(--warning)'
    };
    toast.style.backgroundColor = colors[type] || colors.info;

    // Add to DOM
    document.body.appendChild(toast);

    // Animate in
    setTimeout(() => {
      toast.style.transform = 'translateX(0)';
    }, 100);

    // Remove after 3 seconds
    setTimeout(() => {
      toast.style.transform = 'translateX(100%)';
      setTimeout(() => {
        if (toast.parentNode) {
          toast.parentNode.removeChild(toast);
        }
      }, 300);
    }, 3000);
  }

  // Generate mock data
  generateMockData() {
    return {
      summaries: {
        short: [
          "[TOPIC] represents a significant area of contemporary research with broad implications across multiple domains. Recent studies indicate that [TOPIC_LOWER] has evolved rapidly, driven by technological advancements and changing societal needs. Key findings suggest that [TOPIC_LOWER] offers substantial benefits while also presenting unique challenges that require careful consideration. The interdisciplinary nature of [TOPIC_LOWER] research continues to foster innovation and collaboration among experts worldwide.",
          
          "The field of [TOPIC] has experienced remarkable growth in recent years, establishing itself as a critical area of study. Research demonstrates that [TOPIC_LOWER] addresses fundamental questions and provides practical solutions to complex problems. Current trends indicate increasing investment in [TOPIC_LOWER] initiatives, with promising outcomes reported across various applications. The convergence of theoretical frameworks and practical implementations has positioned [TOPIC_LOWER] as a transformative force in its respective domain.",
          
          "[TOPIC] has emerged as a focal point of academic and industrial interest, reflecting its growing importance in modern society. Evidence suggests that [TOPIC_LOWER] contributes significantly to technological progress and economic development. Ongoing research efforts continue to uncover new possibilities and applications for [TOPIC_LOWER], while also addressing ethical and regulatory considerations. The interdisciplinary approach to [TOPIC_LOWER] research has yielded valuable insights and breakthrough innovations."
        ],
        detailed: [
          "[TOPIC] has emerged as a transformative field with far-reaching implications across academic, industrial, and societal domains. This comprehensive analysis examines the current state of [TOPIC_LOWER], its historical development, and future prospects. The research landscape reveals a dynamic interplay between theoretical foundations and practical applications, with [TOPIC_LOWER] serving as a catalyst for innovation and progress.\n\nRecent advancements in [TOPIC_LOWER] have been propelled by converging technologies and interdisciplinary collaboration. Studies indicate that [TOPIC_LOWER] addresses critical challenges while creating new opportunities for exploration and discovery. The integration of [TOPIC_LOWER] with emerging technologies has resulted in synergistic effects, amplifying its impact and potential applications.\n\nMethodological approaches to [TOPIC_LOWER] research have evolved significantly, incorporating sophisticated analytical tools and frameworks. Empirical evidence demonstrates that [TOPIC_LOWER] initiatives yield measurable benefits across various metrics, including efficiency, accuracy, and scalability. However, researchers also identify important considerations regarding implementation challenges, ethical implications, and long-term sustainability.\n\nThe global [TOPIC_LOWER] ecosystem continues to expand, with increased investment from both public and private sectors. This growth has fostered a vibrant community of practitioners, researchers, and stakeholders who contribute to the ongoing evolution of the field. International collaboration and knowledge sharing have accelerated progress in [TOPIC_LOWER], leading to breakthrough innovations and best practices.\n\nLooking ahead, [TOPIC_LOWER] is poised to play an increasingly pivotal role in shaping future technological and social landscapes. Emerging trends suggest continued innovation and adoption, with potential applications spanning healthcare, education, environmental sustainability, and economic development. The responsible advancement of [TOPIC_LOWER] will require careful consideration of ethical frameworks, regulatory guidelines, and societal impact assessments.",
          
          "The domain of [TOPIC] represents a paradigm shift in how we approach complex problems and opportunities in the modern era. This extensive review synthesizes current research findings, theoretical frameworks, and practical applications related to [TOPIC_LOWER]. The analysis reveals a multifaceted landscape where [TOPIC_LOWER] intersects with numerous disciplines, creating new possibilities for innovation and discovery.\n\nHistorical context demonstrates that [TOPIC_LOWER] has evolved through distinct phases, each characterized by unique technological capabilities and research priorities. Contemporary [TOPIC_LOWER] initiatives benefit from decades of foundational research, while also leveraging cutting-edge tools and methodologies. This evolutionary trajectory has positioned [TOPIC_LOWER] as a mature yet dynamic field with significant growth potential.\n\nTechnical innovations in [TOPIC_LOWER] have addressed many previous limitations, enabling more sophisticated applications and implementations. Research indicates that [TOPIC_LOWER] systems now demonstrate enhanced performance, reliability, and scalability across diverse use cases. These improvements have facilitated broader adoption and integration of [TOPIC_LOWER] solutions in various industries and sectors.\n\nThe socioeconomic impact of [TOPIC_LOWER] extends beyond technical achievements, influencing workforce development, educational curricula, and policy frameworks. Studies highlight both positive outcomes and potential challenges associated with [TOPIC_LOWER] deployment, emphasizing the need for balanced approaches that maximize benefits while mitigating risks. Stakeholder engagement and public discourse have become increasingly important in shaping the future direction of [TOPIC_LOWER].\n\nFuture research directions in [TOPIC_LOWER] point toward greater integration with artificial intelligence, quantum computing, and other transformative technologies. The convergence of these fields promises to unlock new capabilities and applications for [TOPIC_LOWER], potentially revolutionizing how we address global challenges. Continued investment in research, education, and infrastructure will be essential for realizing the full potential of [TOPIC_LOWER] in the coming decades."
        ]
      },
      sources: [
        {
          title: "Advances in [TOPIC] Research: A Comprehensive Review",
          authors: "Johnson, M., Smith, A., & Williams, R.",
          url: "https://example.com/research1",
          year: 2024
        },
        {
          title: "The Future of [TOPIC]: Trends and Implications",
          authors: "Chen, L., Rodriguez, K., & Thompson, J.",
          url: "https://example.com/research2",
          year: 2024
        },
        {
          title: "[TOPIC] Applications in Modern Society",
          authors: "Anderson, S., Kumar, P., & Martinez, D.",
          url: "https://example.com/research3",
          year: 2023
        },
        {
          title: "Ethical Considerations in [TOPIC] Development",
          authors: "Taylor, E., Brown, M., & Davis, L.",
          url: "https://example.com/research4",
          year: 2024
        },
        {
          title: "Methodological Approaches to [TOPIC] Research",
          authors: "Wilson, R., Garcia, A., & Lee, H.",
          url: "https://example.com/research5",
          year: 2023
        },
        {
          title: "Global Perspectives on [TOPIC] Innovation",
          authors: "White, J., Kim, S., & O'Brien, T.",
          url: "https://example.com/research6",
          year: 2024
        },
        {
          title: "[TOPIC] and Economic Development: An Analysis",
          authors: "Harris, M., Nguyen, C., & Clark, P.",
          url: "https://example.com/research7",
          year: 2023
        },
        {
          title: "Technical Challenges in [TOPIC] Implementation",
          authors: "Martin, D., Singh, R., & Robinson, K.",
          url: "https://example.com/research8",
          year: 2024
        },
        {
          title: "[TOPIC] Education and Workforce Development",
          authors: "Thomas, A., Walker, S., & Hall, L.",
          url: "https://example.com/research9",
          year: 2023
        },
        {
          title: "Regulatory Frameworks for [TOPIC] Technologies",
          authors: "Jackson, B., Perez, M., & Allen, R.",
          url: "https://example.com/research10",
          year: 2024
        }
      ]
    };
  }

  // Handle RAG PDF upload
  async handleRAGUpload() {
    const file = this.pdfFile.files[0];
    
    if (!file) {
      this.showError('Please select a PDF file');
      return;
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      this.showError('Please select a PDF file');
      return;
    }

    this.setUploadLoadingState(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/rag/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      
      this.uploadStatus.style.display = 'block';
      this.uploadStatus.className = 'upload-status success';
      this.uploadStatus.innerHTML = `
        <strong>✅ Upload Successful!</strong><br>
        File: ${data.filename}<br>
        Chunks created: ${data.chunks_created}<br>
        Total characters: ${data.total_characters}
      `;

      this.showSuccess('PDF uploaded successfully! You can now ask questions about it.');
      
      // Enable the question input and button
      this.ragQuestion.disabled = false;
      this.ragQueryBtn.disabled = false;
      
      // Add visual indicator that PDF is ready
      this.ragQuestion.placeholder = "Ask a question about your uploaded PDF...";

    } catch (error) {
      console.error('Upload error:', error);
      this.uploadStatus.style.display = 'block';
      this.uploadStatus.className = 'upload-status error';
      this.uploadStatus.innerHTML = `<strong>❌ Upload Failed:</strong> ${error.message}`;
      this.showError('PDF upload failed');
    } finally {
      this.setUploadLoadingState(false);
    }
  }

  // Handle RAG query
  async handleRAGQuery() {
    const question = this.ragQuestion.value.trim();

    if (!question) {
      this.showError('Please enter a question');
      return;
    }

    this.setRAGQueryLoadingState(true);

    try {
      const response = await fetch('http://localhost:8000/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Query failed');
      }

      const data = await response.json();
      
      // Display RAG results
      this.displayRAGResults(data);

    } catch (error) {
      console.error('RAG query error:', error);
      
      // Show more specific error messages
      if (error.message.includes('No PDF has been uploaded yet')) {
        this.showError('Please upload a PDF document first before asking questions. Use the "Upload Document" section above.');
      } else if (error.message.includes('Question cannot be empty')) {
        this.showError('Please enter a question before submitting.');
      } else {
        this.showError('Query failed: ' + error.message);
      }
    } finally {
      this.setRAGQueryLoadingState(false);
    }
  }

  // Set upload loading state
  setUploadLoadingState(isLoading) {
    if (isLoading) {
      this.uploadBtn.disabled = true;
      this.uploadBtnText.style.display = 'none';
      this.uploadBtnLoader.style.display = 'flex';
      this.pdfFile.disabled = true;
    } else {
      this.uploadBtn.disabled = false;
      this.uploadBtnText.style.display = 'inline';
      this.uploadBtnLoader.style.display = 'none';
      this.pdfFile.disabled = false;
    }
  }

  // Set RAG query loading state
  setRAGQueryLoadingState(isLoading) {
    if (isLoading) {
      this.ragQueryBtn.disabled = true;
      this.ragQueryBtnText.style.display = 'none';
      this.ragQueryBtnLoader.style.display = 'flex';
      this.ragQuestion.disabled = true;
    } else {
      this.ragQueryBtn.disabled = false;
      this.ragQueryBtnText.style.display = 'inline';
      this.ragQueryBtnLoader.style.display = 'none';
      this.ragQuestion.disabled = false;
    }
  }

  // Display RAG results
  displayRAGResults(data) {
    // Display question
    this.researchTopicDisplay.textContent = data.question;

    // Display answer
    this.summaryContent.innerHTML = `<div class="rag-answer">${data.answer}</div>`;

    // Display matched chunks as sources
    this.sourcesList.innerHTML = '';
    data.matched_chunks.forEach((chunk, index) => {
      const li = document.createElement('li');
      li.innerHTML = `
        <strong>Document Chunk ${index + 1}</strong><br>
        <span style="color: var(--text-secondary); font-size: var(--font-size-sm);">${chunk.substring(0, 200)}...</span>
      `;
      this.sourcesList.appendChild(li);
    });

    // Show results section
    this.resultsSection.style.display = 'block';
    this.resultsSection.classList.add('fade-in');
    
    // Scroll to results
    setTimeout(() => {
      this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new ResearchAssistant();
});

// Add fade-out animation class
const style = document.createElement('style');
style.textContent = `
  .fade-out {
    animation: fadeOut 0.3s ease-in-out forwards;
  }
  
  @keyframes fadeOut {
    from {
      opacity: 1;
      transform: translateY(0);
    }
    to {
      opacity: 0;
      transform: translateY(-20px);
    }
  }
`;
document.head.appendChild(style);
