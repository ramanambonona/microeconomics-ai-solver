feather.replace();

// Navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        const target = this.getAttribute('data-target');
        showSection(target);
        
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(l => {
            l.classList.remove('active-tab');
        });
        this.classList.add('active-tab');
    });
});

// Mobile menu
document.getElementById('mobileMenuButton').addEventListener('click', function() {
    const menu = document.getElementById('mobileMenu');
    menu.classList.toggle('hidden');
});

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('hidden');
        section.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    targetSection.classList.remove('hidden');
    targetSection.classList.add('active');
}

// Example buttons
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const drift = btn.getAttribute('data-drift');
        const diffusion = btn.getAttribute('data-diffusion');
        document.getElementById('drift').value = drift;
        document.getElementById('diffusion').value = diffusion;
        showSection('solver');
    });
});

// Load example buttons
document.querySelectorAll('.load-example').forEach(btn => {
    btn.addEventListener('click', function() {
        const drift = this.getAttribute('data-drift');
        const diffusion = this.getAttribute('data-diffusion');
        const params = this.getAttribute('data-params');
        
        document.getElementById('drift').value = drift;
        document.getElementById('diffusion').value = diffusion;
        document.getElementById('problemDescription').value = params;
        
        showSection('solver');
    });
});

// Solve SDE function
async function solveSDE() {
    const btn = document.getElementById('solveBtn');
    const thinkingDiv = document.getElementById('aiThinking');
    const stepsContainer = document.getElementById('solutionSteps');
    const errorContainer = document.getElementById('errorContainer');
    const finalAnswer = document.getElementById('finalAnswer');
    const simulationResult = document.getElementById('simulationResult');

    // Reset state
    btn.disabled = true;
    thinkingDiv.classList.remove('hidden');
    stepsContainer.innerHTML = '';
    errorContainer.classList.add('hidden');
    finalAnswer.classList.add('hidden');
    simulationResult.classList.add('hidden');

    try {
        const equationType = document.querySelector('input[name="equationType"]:checked').value;
        const drift = document.getElementById('drift').value.trim();
        const diffusion = document.getElementById('diffusion').value.trim();
        const initialCondition = document.getElementById('initialCondition').value.trim();
        const timeVariable = document.getElementById('timeVariable').value.trim() || 't';
        const paramsText = document.getElementById('problemDescription').value.trim();

        if (!drift || !diffusion) {
            throw new Error('Please enter both drift and diffusion coefficients');
        }

        let parameters = {};
        if (paramsText) {
            try {
                parameters = JSON.parse(paramsText);
            } catch (e) {
                throw new Error('Invalid JSON in parameters field');
            }
        }

        const requestData = {
            equation_type: equationType,
            drift: drift,
            diffusion: diffusion,
            initial_condition: initialCondition || undefined,
            variables: {
                [timeVariable]: 'variable',
                'x': 'variable',
                'W': 'variable'
            },
            parameters: parameters
        };

        console.log('Sending request:', requestData);

        const response = await fetch('http://localhost:8000/solve', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Received response:', data);
        
        // Display solution steps
        stepsContainer.innerHTML = '';
        if (data.steps && Array.isArray(data.steps)) {
            data.steps.forEach((step, idx) => {
                if (step.title && step.content) {
                    addStep(idx + 1, step.title, step.content);
                }
            });
        }
        
        // Display final solution
        if (data.final_solution) {
            document.getElementById('finalFormula').innerHTML = `\\[ ${data.final_solution} \\]`;
            finalAnswer.classList.remove('hidden');
        }

    } catch (error) {
        console.error('Error:', error);
        errorContainer.classList.remove('hidden');
        errorContainer.innerHTML = `
            <div class="error-message">
                <h4 class="font-bold mb-2 flex items-center">
                    <i data-feather="alert-triangle" class="w-4 h-4 mr-2"></i>
                    Solving Error
                </h4>
                <p>${error.message}</p>
                <p class="text-sm mt-2">Please check your equation syntax and try again.</p>
            </div>
        `;
        feather.replace();
    } finally {
        btn.disabled = false;
        thinkingDiv.classList.add('hidden');
        feather.replace();
        // Re-trigger MathJax rendering
        if (typeof MathJax !== 'undefined') {
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
}

// Simulate SDE function
async function simulateSDE() {
    const btn = document.getElementById('simulateBtn');
    const thinkingDiv = document.getElementById('aiThinking');
    const errorContainer = document.getElementById('errorContainer');
    const simulationResult = document.getElementById('simulationResult');
    const finalAnswer = document.getElementById('finalAnswer');

    // Reset state
    btn.disabled = true;
    thinkingDiv.classList.remove('hidden');
    errorContainer.classList.add('hidden');
    simulationResult.classList.add('hidden');
    finalAnswer.classList.add('hidden');

    try {
        const equationType = document.querySelector('input[name="equationType"]:checked').value;
        const drift = document.getElementById('drift').value.trim();
        const diffusion = document.getElementById('diffusion').value.trim();
        const initialCondition = document.getElementById('initialCondition').value.trim();
        const timeStart = parseFloat(document.getElementById('timeStart').value);
        const timeEnd = parseFloat(document.getElementById('timeEnd').value);
        const numTrajectories = parseInt(document.getElementById('numTrajectories').value);
        const paramsText = document.getElementById('problemDescription').value.trim();

        if (!drift || !diffusion) {
            throw new Error('Please enter both drift and diffusion coefficients');
        }

        if (timeStart >= timeEnd) {
            throw new Error('Time end must be greater than time start');
        }

        let parameters = {};
        if (paramsText) {
            try {
                parameters = JSON.parse(paramsText);
            } catch (e) {
                throw new Error('Invalid JSON in parameters field');
            }
        }

        const requestData = {
            equation_type: equationType,
            drift: drift,
            diffusion: diffusion,
            initial_condition: initialCondition,
            parameters: parameters,
            time_span: [timeStart, timeEnd],
            num_points: 100,
            num_trajectories: numTrajectories
        };

        console.log('Sending simulation request:', requestData);

        const response = await fetch('http://localhost:8000/simulate', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Received simulation response:', data);
        
        // Display simulation plot using Plotly
        displaySimulationPlot(data);

    } catch (error) {
        console.error('Error:', error);
        errorContainer.classList.remove('hidden');
        errorContainer.innerHTML = `
            <div class="error-message">
                <h4 class="font-bold mb-2 flex items-center">
                    <i data-feather="alert-triangle" class="w-4 h-4 mr-2"></i>
                    Simulation Error
                </h4>
                <p>${error.message}</p>
                <p class="text-sm mt-2">Please check your parameters and try again.</p>
            </div>
        `;
        feather.replace();
    } finally {
        btn.disabled = false;
        thinkingDiv.classList.add('hidden');
        feather.replace();
    }
}

function displaySimulationPlot(data) {
    const plotDiv = document.getElementById('simulationPlot');
    const simulationResult = document.getElementById('simulationResult');
    
    // Clear previous plot
    plotDiv.innerHTML = '';
    
    // Create traces for each trajectory
    const traces = [];
    data.trajectories.forEach((trajectory, index) => {
        traces.push({
            x: data.time_points,
            y: trajectory,
            type: 'scatter',
            mode: 'lines',
            name: `Trajectory ${index + 1}`,
            line: { color: 'black', width: 1.5 },
            opacity: 0.7
        });
    });
    
    // Layout with Palatino font and custom styling
    const layout = {
        font: {
            family: 'Palatino, serif',
            size: 14,
            color: 'black'
        },
        title: {
            text: 'SDE Simulation',
            font: { size: 18, family: 'Palatino, serif' }
        },
        xaxis: {
            title: 'Time (t)',
            showgrid: true,
            gridcolor: 'lightgray',
            gridwidth: 1,
            zeroline: false
        },
        yaxis: {
            title: 'X(t)',
            showgrid: true,
            gridcolor: 'lightgray',
            gridwidth: 1,
            zeroline: false
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        showlegend: true,
        annotations: [{
            x: 0.02,
            y: 0.98,
            xref: 'paper',
            yref: 'paper',
            text: 'SDE Solver by Ramanambonona Ambinintsoa, PhD',
            showarrow: false,
            font: { family: 'Palatino, serif', size: 12 },
            bgcolor: 'white',
            bordercolor: 'black',
            borderwidth: 1,
            borderpad: 4,
            opacity: 0.8
        }]
    };
    
    // Create plot
    Plotly.newPlot(plotDiv, traces, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
        displaylogo: false
    });
    
    simulationResult.classList.remove('hidden');
}

function downloadPlot() {
    const plotDiv = document.getElementById('simulationPlot');
    Plotly.downloadImage(plotDiv, {
        format: 'png',
        filename: 'sde_simulation',
        width: 1000,
        height: 600
    });
}

// Helper functions
function addStep(number, title, content) {
    const stepsContainer = document.getElementById('solutionSteps');
    const step = document.createElement('div');
    step.className = 'solution-step p-4 bg-white rounded-lg shadow-sm';
    
    step.innerHTML = `
        <div class="flex items-center justify-between mb-3">
            <div class="flex items-center">
                <span class="bg-teal-600 text-white rounded-full w-7 h-7 flex items-center justify-center text-sm font-bold mr-3 flex-shrink-0">${number}</span>
                <span class="font-semibold text-gray-800">${title}</span>
            </div>
            <button class="expand-btn" onclick="toggleStep(this)">
                Expand
            </button>
        </div>
        <div class="formula-box collapsed">
            \\[ ${content} \\]
        </div>
    `;
    
    stepsContainer.appendChild(step);
    
    // Animation
    step.style.opacity = '0';
    step.style.transform = 'translateY(20px)';
    setTimeout(() => {
        step.style.transition = 'all 0.4s ease';
        step.style.opacity = '1';
        step.style.transform = 'translateY(0)';
    }, 100 * number);
}

function toggleStep(button) {
    const formulaBox = button.closest('.solution-step').querySelector('.formula-box');
    const isCollapsed = formulaBox.classList.contains('collapsed');
    
    formulaBox.classList.toggle('collapsed');
    button.textContent = isCollapsed ? 'Collapse' : 'Expand';
}

function toggleAllBoxes() {
    const toggleBtn = document.getElementById('toggleAllText');
    const boxes = document.querySelectorAll('.formula-box');
    const expandBtns = document.querySelectorAll('.expand-btn');
    
    const allExpanded = Array.from(boxes).every(box => !box.classList.contains('collapsed'));
    
    boxes.forEach(box => {
        if (allExpanded) {
            box.classList.add('collapsed');
        } else {
            box.classList.remove('collapsed');
        }
    });
    
    expandBtns.forEach(btn => {
        btn.textContent = allExpanded ? 'Expand' : 'Collapse';
    });
    
    toggleBtn.textContent = allExpanded ? 'Expand All' : 'Collapse All';
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Set active section
    showSection('solver');
    
    // Set active nav link
    document.querySelector('.nav-link[data-target="solver"]').classList.add('active-tab');
});