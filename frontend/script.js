class MCIDiagnosticSystem {
  constructor() {
    this.uploadedFiles = {
      face: null,
      eye: null,
      mri: null,
      speech: null,
    }
    this.currentScreen = "upload"
    this.apiBaseUrl = (window.__APP_CONFIG__ && window.__APP_CONFIG__.API_BASE_URL) || "http://127.0.0.1:5000" // Your main.py API endpoint
    this.processingSteps = [
      { id: "step1", duration: 3000, modality: "face" },
      { id: "step2", duration: 2500, modality: "eye" },
      { id: "step3", duration: 4000, modality: "mri" },
      { id: "step4", duration: 3500, modality: "speech" },
      { id: "step5", duration: 2000, modality: "fusion" },
      { id: "step6", duration: 1500, modality: "report" },
    ]
    this.analysisResults = null
    this.init()
  }

  setupTheme() {
    const savedTheme = localStorage.getItem("mci-theme") || "dark"
    this.setTheme(savedTheme)
  }

  setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme)
    localStorage.setItem("mci-theme", theme)

    const toggle = document.getElementById("themeToggle")
    const lightBtn = document.getElementById("lightTheme")
    const darkBtn = document.getElementById("darkTheme")

    if (theme === "light") {
      toggle.classList.add("light")
      lightBtn.classList.add("active")
      darkBtn.classList.remove("active")
    } else {
      toggle.classList.remove("light")
      darkBtn.classList.add("active")
      lightBtn.classList.remove("active")
    }
  }

  init() {
    this.setupTheme()
    this.setupEventListeners()
    this.setupFileUploads()
    this.updateAnalysisButton()
  }

  setupEventListeners() {
    // Theme controls
    document.getElementById("themeToggle").addEventListener("click", this.toggleTheme.bind(this))
    document.getElementById("lightTheme").addEventListener("click", () => this.setTheme("light"))
    document.getElementById("darkTheme").addEventListener("click", () => this.setTheme("dark"))

    // Analysis button
    document.getElementById("startAnalysisBtn").addEventListener("click", this.showDisclaimer.bind(this))

    // Disclaimer modal
    document.getElementById("medicalConsent").addEventListener("change", this.updateDisclaimerButton.bind(this))
    document.getElementById("acceptDisclaimer").addEventListener("click", this.acceptDisclaimer.bind(this))
  }

  setupFileUploads() {
    const fileTypes = ["face", "eye", "mri", "speech"]

    fileTypes.forEach((type) => {
      const input = document.getElementById(`${type}Input`)
      const uploadArea = document.getElementById(`${type}Upload`)

      uploadArea.addEventListener("click", () => input.click())
      input.addEventListener("change", (e) => this.handleFileUpload(e, type))
      uploadArea.addEventListener("dragover", this.handleDragOver.bind(this))
      uploadArea.addEventListener("dragleave", this.handleDragLeave.bind(this))
      uploadArea.addEventListener("drop", (e) => this.handleDrop(e, type))
    })
  }

  handleFileUpload(event, type) {
    const file = event.target.files[0]
    if (!file) return

    if (!this.validateFile(file, type)) {
      this.showError(`Invalid file type for ${type}. Please select the correct file format.`)
      return
    }

    // Check file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      this.showError("File size must be less than 50MB")
      return
    }

    this.uploadedFiles[type] = file
    this.updateFilePreview(type, file)
    this.updateAnalysisButton()
  }

  validateFile(file, type) {
    const imageTypes = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/tiff", "image/bmp"]
    const audioTypes = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/m4a", "audio/aac"]
    const dicomTypes = ["application/dicom"]

    switch (type) {
      case "face":
      case "eye":
        return imageTypes.includes(file.type)
      case "mri":
        return imageTypes.includes(file.type) || dicomTypes.includes(file.type) || file.name.toLowerCase().endsWith('.dcm')
      case "speech":
        return audioTypes.includes(file.type)
      default:
        return false
    }
  }

  updateFilePreview(type, file) {
    const uploadArea = document.getElementById(`${type}Upload`)
    const preview = document.getElementById(`${type}Preview`)

    uploadArea.classList.add("has-file")
    preview.textContent = `✓ ${file.name} (${this.formatFileSize(file.size)})`
    preview.classList.add("show")
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  updateAnalysisButton() {
    const uploadedCount = Object.values(this.uploadedFiles).filter((file) => file !== null).length
    const button = document.getElementById("startAnalysisBtn")
    const requirement = document.getElementById("uploadRequirement")

    if (uploadedCount >= 2) {
      button.disabled = false
      requirement.textContent = `${uploadedCount}/4 modalities uploaded - Ready for analysis`
    } else {
      button.disabled = true
      requirement.textContent = "Please upload at least 2 modalities to begin analysis"
    }
  }

  handleDragOver(e) {
    e.preventDefault()
    e.currentTarget.style.borderColor = "var(--accent-blue)"
    e.currentTarget.style.backgroundColor = "rgba(74, 158, 255, 0.05)"
  }

  handleDragLeave(e) {
    e.preventDefault()
    e.currentTarget.style.borderColor = "var(--border-dashed)"
    e.currentTarget.style.backgroundColor = ""
  }

  handleDrop(e, type) {
    e.preventDefault()
    e.currentTarget.style.borderColor = "var(--border-dashed)"
    e.currentTarget.style.backgroundColor = ""

    const files = e.dataTransfer.files
    if (files.length > 0) {
      const input = document.getElementById(`${type}Input`)
      const dt = new DataTransfer()
      dt.items.add(files[0])
      input.files = dt.files
      input.dispatchEvent(new Event("change", { bubbles: true }))
    }
  }

  showDisclaimer() {
    document.getElementById("disclaimerModal").classList.add("show")
  }

  updateDisclaimerButton() {
    const consent = document.getElementById("medicalConsent").checked
    const button = document.getElementById("acceptDisclaimer")
    button.disabled = !consent
  }

  acceptDisclaimer() {
    this.closeDisclaimer()
    this.startAnalysis()
  }

  closeDisclaimer() {
    document.getElementById("disclaimerModal").classList.remove("show")
  }

  async startAnalysis() {
    this.showScreen("processing")
    try {
      await this.runRealAnalysis()
    } catch (error) {
      console.error("Analysis failed:", error)
      this.showError("Analysis failed. Please try again.")
      this.showScreen("upload")
    }
  }

  async runRealAnalysis() {
    // Create FormData with uploaded files
    const formData = new FormData()
    
    // Add uploaded files to form data
    Object.entries(this.uploadedFiles).forEach(([modality, file]) => {
      if (file) {
        formData.append(modality, file)
      }
    })

    // Start processing animation
    this.startProcessingAnimation()

    try {
      // Call your main.py API
      const response = await fetch(`${this.apiBaseUrl}/api/analyze`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const results = await response.json()
      
      if (results.success) {
        this.analysisResults = results
        // Wait for processing animation to complete
        await this.waitForProcessingComplete()
        this.showResults()
      } else {
        throw new Error(results.error || "Analysis failed")
      }

    } catch (error) {
      console.error("API Error:", error)
      this.showError(`Analysis failed: ${error.message}`)
      this.showScreen("upload")
    }
  }

  async startProcessingAnimation() {
    const uploadedModalities = Object.entries(this.uploadedFiles)
      .filter(([_, file]) => file !== null)
      .map(([modality, _]) => modality)

    let overallProgress = 0
    const progressIncrement = 100 / this.processingSteps.length

    for (let i = 0; i < this.processingSteps.length; i++) {
      const step = this.processingSteps[i]
      const stepElement = document.getElementById(step.id)

      // Skip steps for modalities that weren't uploaded
      if (step.modality !== "fusion" && step.modality !== "report" && 
          !uploadedModalities.includes(step.modality)) {
        stepElement.style.display = "none"
        continue
      }

      // Update step text based on uploaded modalities
      this.updateStepText(step, uploadedModalities)

      // Activate current step
      stepElement.classList.add("active")

      // Update progress
      overallProgress += progressIncrement
      document.getElementById("overallProgress").textContent = `${Math.round(overallProgress)}%`
      document.getElementById("progressFill").style.width = `${overallProgress}%`

      // Wait for step duration
      await new Promise((resolve) => setTimeout(resolve, step.duration))

      // Complete current step
      stepElement.classList.remove("active")
      stepElement.classList.add("completed")
    }
  }

  updateStepText(step, uploadedModalities) {
    const stepElement = document.getElementById(step.id)
    const titleElement = stepElement.querySelector("h4")
    const descElement = stepElement.querySelector("p")

    switch (step.modality) {
      case "face":
        if (uploadedModalities.includes("face")) {
          titleElement.textContent = "Analyzing facial expressions"
          descElement.textContent = "Processing facial emotion patterns with CNN"
        }
        break
      case "eye":
        if (uploadedModalities.includes("eye")) {
          titleElement.textContent = "Evaluating eye tracking data"
          descElement.textContent = "Analyzing attention and cognitive patterns with ResNet18"
        }
        break
      case "mri":
        if (uploadedModalities.includes("mri")) {
          titleElement.textContent = "Processing MRI scan"
          descElement.textContent = "Detecting structural brain changes with MobileNetV2"
        }
        break
      case "speech":
        if (uploadedModalities.includes("speech")) {
          titleElement.textContent = "Analyzing speech patterns"
          descElement.textContent = "Evaluating speech emotion with LightGBM classifier"
        }
        break
      case "fusion":
        titleElement.textContent = "Multimodal fusion"
        descElement.textContent = `Combining ${uploadedModalities.length} modalities with confidence weighting`
        break
      case "report":
        titleElement.textContent = "Generating clinical report"
        descElement.textContent = "Preparing diagnostic recommendations and risk stratification"
        break
    }
  }

  async waitForProcessingComplete() {
    return new Promise(resolve => setTimeout(resolve, 1000))
  }

  showResults() {
    if (!this.analysisResults) {
      this.showError("No analysis results available")
      return
    }

    const analysis = this.analysisResults.analysis
    this.displayResults(analysis)
    this.showScreen("results")
  }

  displayResults(analysis) {
    // Update confidence
    const confidencePercent = Math.round(analysis.confidence * 100)
    document.getElementById("confidenceValue").textContent = `${confidencePercent}%`
    document.getElementById("confidenceFill").style.width = `${confidencePercent}%`

    // Update risk badge with your system's risk classes
    const riskBadge = document.getElementById("riskBadge")
    riskBadge.textContent = analysis.risk_level
    riskBadge.className = `risk-badge ${analysis.risk_class}`

    // Update recommendation with clinical action
    const recommendationCard = document.getElementById("recommendationCard")
    recommendationCard.innerHTML = `
      <h3><span class="warning-indicator">⚕️</span> Clinical Recommendation</h3>
      <p><strong>${analysis.clinical_action}</strong></p>
      <div class="confidence-band">
        <small>Confidence Band: ${analysis.uncertainty_metrics.confidence_band}</small>
      </div>
    `

    // Update analysis details with clinical insights
    const analysisDetails = document.getElementById("analysisDetails")
    analysisDetails.innerHTML = `
      <h3>Analysis Details</h3>
      <div class="analysis-metrics">
        <p><strong>MCI Probability:</strong> ${(analysis.mci_probability * 100).toFixed(1)}%</p>
        <p><strong>Risk Category:</strong> ${analysis.risk_category}/4</p>
        <p><strong>Decision Certainty:</strong> ${analysis.uncertainty_metrics.decision_certainty}</p>
        <p><strong>Fusion Method:</strong> ${analysis.fusion_method}</p>
        <p><strong>Modalities Analyzed:</strong> ${analysis.modalities_used.join(", ")}</p>
      </div>
      <div class="clinical-insights">
        <h4>Clinical Insights:</h4>
        <ul>
          ${Array.isArray(analysis.clinical_insights)
            ? analysis.clinical_insights.map(insight => `<li>${insight}</li>`).join("")
            : typeof analysis.clinical_insights === 'object' && analysis.clinical_insights !== null
              ? Object.entries(analysis.clinical_insights).map(([key, val]) => `<li><strong>${key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}:</strong> ${val}</li>`).join("")
              : `<li>${analysis.clinical_insights || "No clinical insights available."}</li>`}
        </ul>
      </div>
      <div class="individual-results">
        <h4>Individual Modality Results:</h4>
        ${this.renderIndividualResults()}
      </div>
    `
  }

  renderIndividualResults() {
  if (!this.analysisResults.individual_results) return "";

  const results = this.analysisResults.individual_results;
  let html = "<div class='modality-results'>";

  Object.entries(results).forEach(([modality, result]) => {
    if (result.success) {
      const confidence = result.confidence ? (result.confidence * 100).toFixed(1) : "N/A";

      if (modality.toLowerCase() === "mri") {
        const mciProb = result.mci_probability ? (result.mci_probability * 100).toFixed(1) : "N/A";
        html += `
          <div class="modality-result">
            <strong>${modality.toUpperCase()}:</strong> 
            MCI Probability: ${mciProb}% (${confidence}% confidence)
          </div>
        `;
      } else {
        const emotion = result.predicted_emotion || "N/A";
        html += `
          <div class="modality-result">
            <strong>${modality.toUpperCase()}:</strong> 
            ${emotion} (${confidence}% confidence)
          </div>
        `;
      }
    } else {
      html += `
        <div class="modality-result error">
          <strong>${modality.toUpperCase()}:</strong> 
          ${result.error || "Processing failed"}
        </div>
      `;
    }
  });

  html += "</div>";
  return html;
}

  showScreen(screenName) {
    document.querySelectorAll(".screen").forEach((screen) => {
      screen.classList.remove("active")
    })
    document.getElementById(`${screenName}Screen`).classList.add("active")
    this.currentScreen = screenName
  }

  toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme")
    const newTheme = currentTheme === "dark" ? "light" : "dark"
    this.setTheme(newTheme)
  }

  showError(message) {
    // Create and show error notification
    const errorDiv = document.createElement('div')
    errorDiv.className = 'error-notification'
    errorDiv.innerHTML = `
      <div class="error-content">
        <span class="error-icon">⚠️</span>
        <span class="error-message">${message}</span>
        <button class="error-close" onclick="this.parentElement.parentElement.remove()">×</button>
      </div>
    `
    
    document.body.appendChild(errorDiv)
    setTimeout(() => errorDiv.remove(), 5000)
  }

  // Enhanced API health check
  async checkApiHealth() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/health`)
      const health = await response.json()
      console.log("API Health:", health)
      return health.status === 'healthy'
    } catch (error) {
      console.error("API health check failed:", error)
      return false
    }
  }
}

// Global functions
function closeDisclaimer() {
  window.mciSystem.closeDisclaimer()
}

function startNewAnalysis() {
  // Reset all uploaded files
  window.mciSystem.uploadedFiles = {
    face: null,
    eye: null,
    mri: null,
    speech: null
  }
  
  // Clear all file inputs
  document.querySelectorAll('input[type="file"]').forEach(input => {
    input.value = ""
  })
  
  // Clear all previews
  document.querySelectorAll('.upload-area').forEach(area => {
    area.classList.remove('has-file')
  })
  
  document.querySelectorAll('.file-preview').forEach(preview => {
    preview.classList.remove('show')
    preview.textContent = ""
  })
  
  // Reset processing steps
  document.querySelectorAll('.step').forEach(step => {
    step.classList.remove('active', 'completed')
    step.style.display = 'flex'
  })
  
  // Reset progress
  document.getElementById("overallProgress").textContent = "0%"
  document.getElementById("progressFill").style.width = "0%"
  
  // Update analysis button
  window.mciSystem.updateAnalysisButton()
  
  // Show upload screen
  window.mciSystem.showScreen("upload")
}

function printReport() {
  // Generate printable report
  const results = window.mciSystem.analysisResults
  if (!results) return

  const printWindow = window.open('', '_blank')
  const analysis = results.analysis
  
  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>MCI Diagnostic Report</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }
        .section { margin: 20px 0; }
        .risk-badge { 
          display: inline-block; 
          padding: 10px 20px; 
          border-radius: 20px; 
          font-weight: bold;
          ${analysis.risk_class.includes('high') ? 'background: #fee; color: #c00;' : 
            analysis.risk_class.includes('moderate') ? 'background: #ffd; color: #a60;' : 
            'background: #efe; color: #060;'}
        }
        .disclaimer { 
          background: #f9f9f9; 
          padding: 15px; 
          border-left: 4px solid #ff6600; 
          margin-top: 20px;
          font-size: 0.9em;
        }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>MCI Diagnostic System Report</h1>
        <p>Generated on ${new Date().toLocaleDateString()}</p>
      </div>
      
      <div class="section">
        <h2>Risk Assessment</h2>
        <div class="risk-badge">${analysis.risk_level}</div>
        <p><strong>Confidence:</strong> ${Math.round(analysis.confidence * 100)}%</p>
        <p><strong>MCI Probability:</strong> ${(analysis.mci_probability * 100).toFixed(1)}%</p>
      </div>
      
      <div class="section">
        <h2>Clinical Recommendation</h2>
        <p><strong>${analysis.clinical_action}</strong></p>
      </div>
      
      <div class="section">
        <h2>Clinical Insights</h2>
        <ul>
          ${Array.isArray(analysis.clinical_insights)
            ? analysis.clinical_insights.map(insight => `<li>${insight}</li>`).join("")
            : typeof analysis.clinical_insights === 'object' && analysis.clinical_insights !== null
              ? Object.entries(analysis.clinical_insights).map(([key, val]) => `<li><strong>${key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}:</strong> ${val}</li>`).join("")
              : `<li>${analysis.clinical_insights || "No clinical insights available."}</li>`}
        </ul>
      </div>
      
      <div class="section">
        <h2>Analysis Details</h2>
        <p><strong>Modalities Analyzed:</strong> ${analysis.modalities_used.join(", ")}</p>
        <p><strong>Fusion Method:</strong> ${analysis.fusion_method}</p>
        <p><strong>Decision Certainty:</strong> ${analysis.uncertainty_metrics.decision_certainty}</p>
        <p><strong>Confidence Band:</strong> ${analysis.uncertainty_metrics.confidence_band}</p>
      </div>
      
      <div class="disclaimer">
        <h3>Medical Disclaimer</h3>
        <p><strong>This AI system is NOT a substitute for professional medical advice, diagnosis, or treatment.</strong> 
        This screening tool is designed for research and preliminary assessment purposes only. 
        All results should be interpreted by qualified medical professionals.</p>
      </div>
    </body>
    </html>
  `)
  
  printWindow.document.close()
  printWindow.print()
}

// Initialize the application
document.addEventListener("DOMContentLoaded", async () => {
  window.mciSystem = new MCIDiagnosticSystem()
  
  // Check API health on startup
  const isHealthy = await window.mciSystem.checkApiHealth()
  if (!isHealthy) {
    console.warn("Backend API is not responding. Please ensure your Python backend is running on http://localhost:5000")
  }
})
