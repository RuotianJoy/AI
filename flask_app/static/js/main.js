document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const mInput = document.getElementById('m');
    const nInput = document.getElementById('n');
    const kInput = document.getElementById('k');
    const jInput = document.getElementById('j');
    const sInput = document.getElementById('s');
    const fInput = document.getElementById('f');
    const algorithmSelect = document.getElementById('algorithm');
    const randomSelect = document.querySelector('input[value="random"]');
    const manualSelect = document.querySelector('input[value="manual"]');
    const sampleInput = document.getElementById('sampleInput');
    const executeBtn = document.getElementById('executeBtn');
    const historyBtn = document.getElementById('historyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const progressBar = document.querySelector('.progress');
    const progressText = document.querySelector('.progress-text');
    const samplesDisplay = document.getElementById('samplesDisplay');
    const resultsDisplay = document.getElementById('resultsDisplay');
    const runId = document.getElementById('runId');

    // State variables
    let currentSamples = [];
    let currentResults = [];
    let currentRunId = null;

    // Event Listeners
    mInput.addEventListener('change', updateNMax);
    nInput.addEventListener('change', () => {
        updateConstraints();
        updateSampleInput();
    });
    jInput.addEventListener('change', updateConstraints);
    sInput.addEventListener('change', updateConstraints);
    randomSelect.addEventListener('change', toggleManualInput);
    manualSelect.addEventListener('change', toggleManualInput);
    executeBtn.addEventListener('click', calculateOptimalGroups);
    historyBtn.addEventListener('click', () => {
        window.location.href = '/history';
    });
    clearBtn.addEventListener('click', clearInputAndResults);

    // Functions
    function updateNMax() {
        const mValue = parseInt(mInput.value);
        nInput.max = Math.min(25, mValue);
        if (parseInt(nInput.value) > mValue) {
            nInput.value = mValue;
        }
        updateSampleInput();
    }

    function updateConstraints() {
        const nValue = parseInt(nInput.value);
        const jValue = parseInt(jInput.value);
        const sValue = parseInt(sInput.value);

        // Ensure j >= s
        if (jValue < sValue) {
            jInput.value = sValue;
        }

        // Ensure j <= n, k <= n
        jInput.max = nValue;
        kInput.max = Math.min(7, nValue);

        // Ensure s <= j, s <= k
        const sMax = Math.min(parseInt(jInput.value), parseInt(kInput.value));
        sInput.max = Math.min(7, sMax);
    }
    
    // Function to auto-populate the sample input with a sequence from 01 to n
    function updateSampleInput() {
        if (manualSelect.checked) {
            const n = parseInt(nInput.value);
            const samples = Array.from({length: n}, (_, i) => 
                (i + 1).toString().padStart(2, '0')
            );
            sampleInput.value = samples.join(', ');
        }
    }

    function toggleManualInput() {
        const manualInputDiv = document.querySelector('.manual-input');
        if (manualSelect.checked) {
            manualInputDiv.style.display = 'block';
            sampleInput.disabled = false;
            updateSampleInput();
        } else {
            manualInputDiv.style.display = 'none';
            sampleInput.disabled = true;
        }
    }

    function validateParameters() {
        const m = parseInt(mInput.value);
        const n = parseInt(nInput.value);
        const k = parseInt(kInput.value);
        const j = parseInt(jInput.value);
        const s = parseInt(sInput.value);

        if (!(3 <= s && s <= j && j <= k && k <= n && n <= m)) {
            alert('Parameters must satisfy: 3 ≤ s ≤ j ≤ k ≤ n ≤ m');
            return false;
        }
        return true;
    }

    function prepareSamples() {
        const m = parseInt(mInput.value);
        const n = parseInt(nInput.value);

        if (randomSelect.checked) {
            // Random selection
            const allSamples = Array.from({length: m}, (_, i) => (i + 1).toString().padStart(2, '0'));
            currentSamples = [];
            while (currentSamples.length < n) {
                const randomIndex = Math.floor(Math.random() * allSamples.length);
                const sample = allSamples[randomIndex];
                if (!currentSamples.includes(sample)) {
                    currentSamples.push(sample);
                }
            }
        } else {
            // Manual input
            const inputText = sampleInput.value.trim();
            if (!inputText) {
                alert('Please enter sample numbers');
                return false;
            }

            try {
                currentSamples = inputText.split(',').map(s => s.trim().padStart(2, '0'));
                
                // Validate samples
                if (currentSamples.length !== n) {
                    alert(`Please enter exactly ${n} samples`);
                    return false;
                }
                
                if (new Set(currentSamples).size !== currentSamples.length) {
                    alert('Sample numbers cannot be duplicated');
                    return false;
                }
                
                if (currentSamples.some(s => !s.match(/^\d+$/) || parseInt(s) < 1 || parseInt(s) > m)) {
                    alert(`Sample numbers must be between 01-${m.toString().padStart(2, '0')}`);
                    return false;
                }
            } catch (e) {
                alert('Please enter valid sample numbers, separated by commas');
                return false;
            }
        }

        // Display selected samples
        currentSamples.sort();
        samplesDisplay.innerHTML = currentSamples.map((sample, i) => 
            `${i + 1}# ${sample}`
        ).join('<br>');

        return true;
    }

    async function calculateOptimalGroups() {
        if (!validateParameters() || !prepareSamples()) {
            return;
        }

        // Get parameters
        const j = parseInt(jInput.value);
        const s = parseInt(sInput.value);
        const k = parseInt(kInput.value);
        const f = parseInt(fInput.value);
        const algorithm = algorithmSelect.value;

        // Disable execute button
        executeBtn.disabled = true;
        executeBtn.textContent = 'Processing...';

        // Clear results
        resultsDisplay.innerHTML = 'Calculation started, please wait...<br>';

        // Create current run ID
        currentRunId = Date.now();

        // Show progress bar as processing
        progressBar.style.width = '50%';
        progressText.textContent = 'Processing...';
        document.querySelector('.progress-section').style.display = 'block';

        try {
            // Send request to backend
            const response = await fetch('/api/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    m: parseInt(mInput.value),
                    n: parseInt(nInput.value),
                    k: k,
                    j: j,
                    s: s,
                    f: f,
                    algorithm: algorithm,
                    samples: currentSamples
                })
            });

            const data = await response.json();

            // Update progress bar to complete
            progressBar.style.width = '100%';
            progressText.textContent = '100%';

            // Enable execute button
            executeBtn.disabled = false;
            executeBtn.textContent = 'Execute';

            if (data.success) {
                // Store results
                currentResults = data.result;
                
                // Display results
                displayResults(currentResults, data.message);
                
                // Show run ID
                runId.textContent = `Run ID: ${currentRunId}`;
            } else {
                // Display error
                resultsDisplay.innerHTML = `<span class="error-message">Error: ${data.message}</span>`;
            }
        } catch (error) {
            console.error('Error:', error);
            
            // Update progress bar to error state
            progressBar.style.width = '100%';
            progressBar.style.backgroundColor = '#ff4d4d';
            progressText.textContent = 'Error';
            
            // Enable execute button
            executeBtn.disabled = false;
            executeBtn.textContent = 'Execute';
            
            // Display error
            resultsDisplay.innerHTML = `<span class="error-message">Error: ${error.message}</span>`;
        }
    }

    function displayResults(results, message) {
        if (!results || results.length === 0) {
            resultsDisplay.innerHTML = 'No valid combinations found.';
            return;
        }

        let html = '';
        
        // Display message (execution time)
        if (message) {
            html += `<div class="result-message">${message}</div>`;
        }
        
        // Display combinations count
        html += `<div class="combinations-count">Found ${results.length} combinations:</div>`;
        
        // Display each combination
        for (let i = 0; i < results.length; i++) {
            const combination = results[i];
            html += `<div class="combination">${i + 1}# ${combination.join(', ')}</div>`;
        }
        
        resultsDisplay.innerHTML = html;
    }

    function clearInputAndResults() {
        // Reset inputs to default values
        mInput.value = '45';
        nInput.value = '7';
        kInput.value = '6';
        jInput.value = '4';
        sInput.value = '4';
        fInput.value = '1';
        
        // Reset sample selection
        randomSelect.checked = true;
        manualSelect.checked = false;
        toggleManualInput();
        
        // Clear displays
        samplesDisplay.innerHTML = '';
        resultsDisplay.innerHTML = '';
        runId.textContent = '';
        
        // Hide progress bar
        document.querySelector('.progress-section').style.display = 'none';
        
        // Reset state
        currentSamples = [];
        currentResults = [];
        currentRunId = null;
    }

    // Initialize constraints
    updateNMax();
    updateConstraints();
    toggleManualInput();
});