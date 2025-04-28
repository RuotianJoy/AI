document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const mInput = document.getElementById('m');
    const nInput = document.getElementById('n');
    const kInput = document.getElementById('k');
    const jInput = document.getElementById('j');
    const sInput = document.getElementById('s');
    const fInput = document.getElementById('f');
    const randomSelect = document.querySelector('input[value="random"]');
    const manualSelect = document.querySelector('input[value="manual"]');
    const sampleInput = document.getElementById('sampleInput');
    const compareBtn = document.getElementById('compareBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const backBtn = document.getElementById('backBtn');
    const runningStatus = document.getElementById('runningStatus');
    const resultsSection = document.getElementById('resultsSection');
    const exportPdfBtn = document.getElementById('exportPdfBtn');

    // Algorithm status elements
    const algorithmStatus = {
        greedy: document.getElementById('greedyStatus'),
        genetic: document.getElementById('geneticStatus'),
        annealing: document.getElementById('annealingStatus')
    };

    // Results display elements
    const algorithmResults = {
        greedy: document.getElementById('greedyResults'),
        genetic: document.getElementById('geneticResults'),
        annealing: document.getElementById('annealingResults')
    };

    // Metrics display elements
    const timeElements = {
        greedy: document.getElementById('greedyTime'),
        genetic: document.getElementById('geneticTime'),
        annealing: document.getElementById('annealingTime')
    };

    const sizeElements = {
        greedy: document.getElementById('greedySize'),
        genetic: document.getElementById('geneticSize'),
        annealing: document.getElementById('annealingSize')
    };

    // Confidence display elements
    const confidenceElements = {
        greedy: document.getElementById('greedyConfidence'),
        genetic: document.getElementById('geneticConfidence'),
        annealing: document.getElementById('annealingConfidence')
    };

    // State variables
    let currentSamples = [];
    let compareResults = {};
    let runningAlgorithms = new Set();
    let abortControllers = {};

    // Event listeners
    mInput.addEventListener('change', updateNMax);
    nInput.addEventListener('change', () => {
        updateConstraints();
        updateSampleInput();
    });
    jInput.addEventListener('change', updateConstraints);
    sInput.addEventListener('change', updateConstraints);
    randomSelect.addEventListener('change', toggleManualInput);
    manualSelect.addEventListener('change', toggleManualInput);
    compareBtn.addEventListener('click', startComparison);
    cancelBtn.addEventListener('click', cancelComparison);
    backBtn.addEventListener('click', () => {
        window.location.href = '/';
    });
    exportPdfBtn.addEventListener('click', exportPdfReport);

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
            // Sort the samples
            currentSamples.sort();
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
                
                // Sort the samples
                currentSamples.sort();
            } catch (e) {
                alert('Please enter valid sample numbers, separated by commas');
                return false;
            }
        }

        return true;
    }

    function startComparison() {
        if (!validateParameters() || !prepareSamples()) {
            return;
        }
        
        // Reset previous results
        resetResults();
        
        // Show running status
        runningStatus.style.display = 'flex';
        resultsSection.style.display = 'block';
        
        // Disable inputs
        toggleInputs(true);
        
        // Get parameters for all algorithms
        const params = {
            m: parseInt(mInput.value),
            n: parseInt(nInput.value),
            k: parseInt(kInput.value),
            j: parseInt(jInput.value),
            s: parseInt(sInput.value),
            f: parseInt(fInput.value),
            samples: currentSamples
        };
        
        // Set of algorithms to run
        const algorithms = ['greedy', 'genetic', 'annealing'];
        
        // Store which algorithms are running
        runningAlgorithms = new Set(algorithms);
        
        // Create controllers for each algorithm (for cancellation)
        abortControllers = {};
        
        // Start each algorithm
        algorithms.forEach(alg => {
            updateAlgorithmStatus(alg, 'running');
            runAlgorithm(alg, params);
        });
    }
    
    async function runAlgorithm(algorithm, params) {
        const controller = new AbortController();
        abortControllers[algorithm] = controller;
        
        try {
            const response = await fetch('/api/compare_calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...params,
                    algorithm: algorithm
                }),
                signal: controller.signal
            });
            
            const result = await response.json();
            
            // Store and display results
            if (result.success) {
                compareResults[algorithm] = {
                    combinations: result.result,
                    metrics: result.metrics,
                    executionTime: result.executionTime || '0'
                };
                
                // Update the display
                updateAlgorithmResult(algorithm);
            } else {
                algorithmResults[algorithm].innerHTML = `<div class="error-message">Error: ${result.message}</div>`;
                updateAlgorithmStatus(algorithm, 'error');
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                algorithmResults[algorithm].innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                updateAlgorithmStatus(algorithm, 'error');
            }
        }
        
        // Mark this algorithm as completed
        if (runningAlgorithms.has(algorithm)) {
            runningAlgorithms.delete(algorithm);
            
            // Update the overall comparison if all algorithms have completed
            if (runningAlgorithms.size === 0) {
                updateComparisonMetrics();
                toggleInputs(false);
            }
        }
    }
    
    function cancelComparison() {
        // Cancel all fetch requests
        Object.values(abortControllers).forEach(controller => {
            try {
                controller.abort();
            } catch (e) {
                console.error('Error aborting fetch:', e);
            }
        });
        
        // Reset running algorithms
        runningAlgorithms.clear();
        
        // Hide running status
        runningStatus.style.display = 'none';
        
        // Reset UI
        resetResults();
        toggleInputs(false);
        
        // Set algorithm statuses to waiting
        Object.keys(algorithmStatus).forEach(alg => {
            updateAlgorithmStatus(alg, 'waiting');
        });
    }
    
    function checkAllCompleted() {
        // Check if all algorithms have completed
        const allCompleted = document.querySelectorAll('.status-completed').length === 3;
        
        if (allCompleted) {
            // Enable inputs and hide running status
            toggleInputs(false);
            runningStatus.style.display = 'none';
            
            // Update the complete results display
            setTimeout(() => {
                document.getElementById('conclusionText').textContent = generateConclusion();
            }, 500);
        }
    }
    
    function updateAlgorithmStatus(algorithm, status) {
        const statusElement = algorithmStatus[algorithm];
        if (!statusElement) return;
        
        // Remove all status classes
        const dot = statusElement.querySelector('.status-dot');
        dot.classList.remove('status-waiting', 'status-running', 'status-completed', 'status-error');
        
        // Add appropriate class
        dot.classList.add(`status-${status}`);
        
        // Update text
        const statusText = statusElement.querySelector('span');
        let statusName = '';
        
        switch (algorithm) {
            case 'greedy': statusName = 'Greedy Algorithm'; break;
            case 'genetic': statusName = 'Genetic Algorithm'; break;
            case 'annealing': statusName = 'Simulated Annealing'; break;
        }
        
        switch (status) {
            case 'waiting': statusText.textContent = `${statusName}: Waiting`; break;
            case 'running': statusText.textContent = `${statusName}: Running`; break;
            case 'completed': statusText.textContent = `${statusName}: Completed`; break;
            case 'error': statusText.textContent = `${statusName}: Error`; break;
        }
        
        // Check if all algorithms have completed
        checkAllCompleted();
    }
    
    function updateAlgorithmResult(algorithm) {
        if (!compareResults[algorithm]) return;
        
        const result = compareResults[algorithm];
        const metrics = result.metrics || {};
        
        // Update time and size displays
        timeElements[algorithm].textContent = `Runtime: ${result.executionTime}s`;
        sizeElements[algorithm].textContent = `Combinations: ${result.combinations.length}`;
        
        // Update confidence meter
        const confidenceValue = metrics.confidence || 0;
        confidenceElements[algorithm].innerHTML = `
            Confidence: ${confidenceValue.toFixed(1)}%
            <div class="confidence-meter">
                <div class="confidence-value" style="width: ${confidenceValue}%"></div>
            </div>
        `;
        
        // Update results display
        if (result.combinations.length > 0) {
            algorithmResults[algorithm].innerHTML = `
                <div class="combinations-count">
                    ${result.combinations.length} combinations found
                </div>
                <div class="combinations-list">
                    ${result.combinations.map((combo, idx) => 
                        `<div class="combination">${idx + 1}. ${combo.join(', ')}</div>`
                    ).join('')}
                </div>
            `;
        } else {
            algorithmResults[algorithm].innerHTML = "<div class='no-results'>No combinations found</div>";
        }
        
        // Mark as completed
        updateAlgorithmStatus(algorithm, 'completed');
        
        // Update the complete comparison if all algorithms have results
        if (Object.keys(compareResults).length === 3) {
            updateComparisonMetrics();
        }
    }
    
    function updateComparisonMetrics() {
        // Get the metrics for all completed algorithms
        const metrics = {
            greedy: compareResults.greedy?.metrics || null,
            genetic: compareResults.genetic?.metrics || null,
            annealing: compareResults.annealing?.metrics || null
        };
        
        // Create table rows for each algorithm
        const tableBody = document.getElementById('metricsTableBody');
        tableBody.innerHTML = '';
        
        // Find the best values for highlighting
        const bestValues = {
            combinations: Infinity,  // Lower is better
            time: Infinity,          // Lower is better
            coverage: 0,             // Higher is better
            avgGroupSize: 0,         // Higher is better
            utilization: 0,          // Higher is better
            confidence: 0            // Higher is better
        };
        
        // Get all result values for comparison
        Object.keys(compareResults).forEach(alg => {
            const result = compareResults[alg];
            const algMetrics = result.metrics || {};
            
            // Update best values
            bestValues.combinations = Math.min(bestValues.combinations, result.combinations.length);
            bestValues.time = Math.min(bestValues.time, parseFloat(result.executionTime));
            bestValues.coverage = Math.max(bestValues.coverage, algMetrics.coverage || 0);
            bestValues.avgGroupSize = Math.max(bestValues.avgGroupSize, algMetrics.avgGroupSize || 0);
            bestValues.utilization = Math.max(bestValues.utilization, algMetrics.utilization || 0);
            bestValues.confidence = Math.max(bestValues.confidence, algMetrics.confidence || 0);
        });
        
        // Create table rows
        Object.keys(metrics).forEach(alg => {
            if (!compareResults[alg]) return;
            
            const result = compareResults[alg];
            const algMetrics = result.metrics || {};
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${getAlgorithmName(alg)}</td>
                <td class="${result.combinations.length === bestValues.combinations ? 'best-metric' : ''}">${result.combinations.length}</td>
                <td class="${parseFloat(result.executionTime) === bestValues.time ? 'best-metric' : ''}">${result.executionTime}</td>
                <td class="${(algMetrics.coverage || 0) === bestValues.coverage ? 'best-metric' : ''}">${algMetrics.coverage?.toFixed(1) || 'N/A'}%</td>
                <td class="${(algMetrics.avgGroupSize || 0) === bestValues.avgGroupSize ? 'best-metric' : ''}">${algMetrics.avgGroupSize?.toFixed(2) || 'N/A'}</td>
                <td class="${(algMetrics.utilization || 0) === bestValues.utilization ? 'best-metric' : ''}">${algMetrics.utilization?.toFixed(1) || 'N/A'}%</td>
                <td class="${(algMetrics.confidence || 0) === bestValues.confidence ? 'best-metric' : ''}">${algMetrics.confidence?.toFixed(1) || 'N/A'}%</td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Generate and display conclusion
        document.getElementById('conclusionText').textContent = generateConclusion();
    }
    
    function generateConclusion() {
        if (Object.keys(compareResults).length < 2) {
            return "Waiting for comparison to complete...";
        }
        
        // Find the "best" algorithm based on confidence
        let bestAlg = null;
        let bestConfidence = -1;
        
        Object.keys(compareResults).forEach(alg => {
            const metrics = compareResults[alg]?.metrics || {};
            if ((metrics.confidence || 0) > bestConfidence) {
                bestConfidence = metrics.confidence || 0;
                bestAlg = alg;
            }
        });
        
        if (!bestAlg) {
            return "Unable to determine the best algorithm due to incomplete data.";
        }
        
        // Create conclusion text
        let conclusion = "";
        
        // Add best algorithm
        conclusion += `The ${getAlgorithmName(bestAlg)} performs best overall with ${bestConfidence.toFixed(1)}% confidence. `;
        
        // Add information about combination counts
        const combinationCounts = {};
        Object.keys(compareResults).forEach(alg => {
            combinationCounts[alg] = compareResults[alg]?.combinations?.length || 0;
        });
        
        // Find algorithm with fewest combinations
        const minCombos = Math.min(...Object.values(combinationCounts));
        const minComboAlg = Object.keys(combinationCounts).find(alg => combinationCounts[alg] === minCombos);
        
        conclusion += `${getAlgorithmName(minComboAlg)} generates the fewest combinations (${minCombos}). `;
        
        // Add information about execution time
        const executionTimes = {};
        Object.keys(compareResults).forEach(alg => {
            executionTimes[alg] = parseFloat(compareResults[alg]?.executionTime || '0');
        });
        
        // Find algorithm with fastest time
        const minTime = Math.min(...Object.values(executionTimes));
        const fastestAlg = Object.keys(executionTimes).find(alg => executionTimes[alg] === minTime);
        
        conclusion += `${getAlgorithmName(fastestAlg)} is the fastest algorithm (${minTime}s).`;
        
        return conclusion;
    }
    
    function calculateUtilization(result) {
        if (!result || !result.combinations || !currentSamples) return 0;
        
        // Get all unique samples used in the combinations
        const usedSamples = new Set();
        result.combinations.forEach(combo => {
            combo.forEach(sample => usedSamples.add(sample));
        });
        
        // Calculate utilization ratio
        return (usedSamples.size / currentSamples.length) * 100;
    }
    
    function resetResults() {
        // Reset results data
        compareResults = {};
        runningAlgorithms.clear();
        
        // Reset UI elements
        Object.keys(algorithmResults).forEach(alg => {
            algorithmResults[alg].innerHTML = '';
            timeElements[alg].textContent = 'Runtime: --';
            sizeElements[alg].textContent = 'Combinations: --';
            confidenceElements[alg].innerHTML = `
                Confidence: --
                <div class="confidence-meter">
                    <div class="confidence-value" style="width: 0%"></div>
                </div>
            `;
        });
        
        // Reset metrics table
        document.getElementById('metricsTableBody').innerHTML = '';
        
        // Reset conclusion
        document.getElementById('conclusionText').textContent = 'Waiting for the comparison to complete...';
    }
    
    function toggleInputs(disabled) {
        // Toggle parameter inputs
        mInput.disabled = disabled;
        nInput.disabled = disabled;
        kInput.disabled = disabled;
        jInput.disabled = disabled;
        sInput.disabled = disabled;
        fInput.disabled = disabled;
        
        // Toggle sample selection
        randomSelect.disabled = disabled;
        manualSelect.disabled = disabled;
        sampleInput.disabled = disabled || !manualSelect.checked;
        
        // Toggle buttons
        compareBtn.disabled = disabled;
        cancelBtn.disabled = !disabled;
    }
    
    function getAlgorithmName(algorithm) {
        switch (algorithm) {
            case 'greedy': return 'Greedy Algorithm';
            case 'genetic': return 'Genetic Algorithm';
            case 'annealing': return 'Simulated Annealing';
            default: return algorithm;
        }
    }

    function exportPdfReport() {
        // Check if we have comparison results
        if (Object.keys(compareResults).length === 0) {
            alert('No comparison results to export. Please run a comparison first.');
            return;
        }

        // Prepare the data for PDF generation
        const data = {
            parameters: {
                m: parseInt(mInput.value),
                n: parseInt(nInput.value),
                k: parseInt(kInput.value),
                j: parseInt(jInput.value),
                s: parseInt(sInput.value),
                f: parseInt(fInput.value)
            },
            samples: currentSamples,
            results: compareResults,
            conclusion: document.getElementById('conclusionText').textContent
        };

        // Send data to backend for PDF generation
        fetch('/api/export_pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to generate PDF');
            }
            return response.blob();
        })
        .then(blob => {
            // Create a URL for the blob
            const url = URL.createObjectURL(blob);
            
            // Create a link and trigger download
            const a = document.createElement('a');
            a.href = url;
            a.download = `algorithm_comparison_${new Date().toISOString().split('T')[0]}.pdf`;
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('Error exporting PDF:', error);
            alert('Failed to export PDF: ' + error.message);
        });
    }
}); 