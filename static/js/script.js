// Tesla Stock Price Prediction - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the input table with 10 rows
    initInputTable();
    
    // Set up event listeners
    setupEventListeners();
});

// Initialize the input table with 10 empty rows
function initInputTable() {
    const tbody = document.querySelector('#inputTable tbody');
    tbody.innerHTML = '';
    
    // Get current date
    const today = new Date();
    
    // Create 10 rows for input data
    for (let i = 9; i >= 0; i--) {
        // Calculate date (today - i days)
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        const dateStr = formatDate(date);
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>
                <input type="date" class="form-control input-date" value="${dateStr}" required>
            </td>
            <td>
                <input type="number" step="0.01" min="0" class="form-control input-price" placeholder="Close Price" required>
            </td>
        `;
        tbody.appendChild(tr);
    }
}

// Format date as YYYY-MM-DD
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// Set up event listeners for forms and buttons
function setupEventListeners() {
    // Upload form submission
    document.getElementById('uploadForm').addEventListener('submit', function(e) {
        e.preventDefault();
        uploadFile();
    });
    
    // Train button click
    document.getElementById('trainButton').addEventListener('click', function() {
        trainModel();
    });
    
    // Predict button click
    document.getElementById('predictButton').addEventListener('click', function() {
        predictPrices();
    });

    // Check model status on page load
    checkModelStatus();
}

// Upload CSV file
function uploadFile() {
    const fileInput = document.getElementById('csvFile');
    const statusDiv = document.getElementById('uploadStatus');
    
    // Check if file is selected
    if (!fileInput.files || fileInput.files.length === 0) {
        showAlert(statusDiv, 'Please select a CSV file to upload.', 'danger');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Check file type
    if (!file.name.endsWith('.csv')) {
        showAlert(statusDiv, 'Only CSV files are allowed.', 'danger');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading status
    showAlert(statusDiv, '<div class="loading-spinner"></div> Uploading file...', 'info');
    
    // Send request to server
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showAlert(statusDiv, `${data.message}. Loaded ${data.rows} rows of data.`, 'success');
        } else {
            showAlert(statusDiv, data.message, 'danger');
        }
    })
    .catch(error => {
        showAlert(statusDiv, `Error: ${error.message}`, 'danger');
    });
}

// Train the model
async function trainModel() {
    try {
        const trainButton = document.getElementById('trainButton');
        trainButton.disabled = true;
        updateStatus('Checking model status...', false);
        
        const response = await fetch('http://127.0.0.1:5000/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (data.pretrained) {
            updateStatus(data.message, false, 'success');
            trainButton.textContent = 'Model Ready';
            return;
        }

        if (response.status === 409) {
            updateStatus(`${data.message} (${data.progress}%)`, false);
            await pollTrainingStatus();
            return;
        }

        if (!response.ok && response.status !== 409) {
            throw new Error(data.message || `HTTP error! status: ${response.status}`);
        }

        await pollTrainingStatus();
        
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error: ' + error.message, true);
    } finally {
        document.getElementById('trainButton').disabled = false;
    }
}

async function pollTrainingStatus() {
    const pollInterval = 1000; // Poll every second
    const maxAttempts = 600;   // Maximum 10 minutes of polling
    let attempts = 0;

    const statusElement = document.getElementById('training-status');
    statusElement.className = 'status progress';

    while (attempts < maxAttempts) {
        try {
            const response = await fetch('http://127.0.0.1:5000/train/status');
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.status}`);
            }

            const data = await response.json();
            updateStatus(`${data.message} (${data.progress}%)`);

            if (!data.is_training) {
                if (data.error) {
                    throw new Error(data.error);
                }
                updateStatus('Training completed successfully!', false, 'success');
                return;
            }

            await new Promise(resolve => setTimeout(resolve, pollInterval));
            attempts++;
        } catch (error) {
            updateStatus('Error checking status: ' + error.message, true);
            throw error;
        }
    }
    throw new Error('Training timeout - exceeded 10 minutes');
}

async function checkModelStatus() {
    try {
        const response = await fetch('http://127.0.0.1:5000/model/status');
        const data = await response.json();
        
        const trainButton = document.getElementById('trainButton');
        if (data.is_trained) {
            trainButton.textContent = 'Model Ready';
            updateStatus(data.message, false, 'success');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
}

function updateStatus(message, isError = false, className = '') {
    const statusElement = document.getElementById('training-status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = className || (isError ? 'status error' : 'status progress');
    }
}

// Predict stock prices
function predictPrices() {
    const statusDiv = document.getElementById('predictStatus');
    const resultsSection = document.getElementById('resultsSection');
    
    // Get input data
    const inputData = getInputData();
    if (!inputData) {
        showAlert(statusDiv, 'Please fill in all date and price fields.', 'danger');
        return;
    }
    
    // Get number of days to predict
    const numDays = parseInt(document.getElementById('predictionDays').value);
    if (isNaN(numDays) || numDays < 1 || numDays > 20) {
        showAlert(statusDiv, 'Please enter a valid number of days (1-20).', 'danger');
        return;
    }
    
    // Show loading status
    showAlert(statusDiv, '<div class="loading-spinner"></div> Making predictions...', 'info');
    
    // Hide results section
    resultsSection.style.display = 'none';
    
    // Send request to server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            input_data: inputData,
            num_days: numDays
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showAlert(statusDiv, 'Predictions generated successfully!', 'success');
            
            // Display results
            displayPredictionResults(data.predictions, data.plot_path);
            
            // Show results section
            resultsSection.style.display = 'block';
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            showAlert(statusDiv, data.message, 'danger');
        }
    })
    .catch(error => {
        showAlert(statusDiv, `Error: ${error.message}`, 'danger');
    });
}

// Get input data from the table
function getInputData() {
    const rows = document.querySelectorAll('#inputTable tbody tr');
    const inputData = [];
    
    for (let row of rows) {
        const dateInput = row.querySelector('.input-date');
        const priceInput = row.querySelector('.input-price');
        
        if (!dateInput.value || !priceInput.value) {
            return null;
        }
        
        inputData.push({
            date: dateInput.value,
            close: parseFloat(priceInput.value)
        });
    }
    
    return inputData;
}

// Display prediction results
function displayPredictionResults(predictions, plotPath) {
    // Display plot
    document.getElementById('predictionPlot').src = plotPath;
    
    // Fill prediction table
    const tbody = document.querySelector('#predictionTable tbody');
    tbody.innerHTML = '';
    
    const dates = predictions.dates;
    const prices = predictions.predictions;
    
    for (let i = 0; i < dates.length; i++) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${dates[i]}</td>
            <td>$${prices[i].toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
    }
}

// Show alert message
function showAlert(element, message, type) {
    element.innerHTML = message;
    element.className = `alert mt-3 alert-${type}`;
    element.classList.remove('d-none');
}