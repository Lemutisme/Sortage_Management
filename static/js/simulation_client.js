// File: static/js/simulation_client.js
/**
 * Real-time simulation client for ShortageSim web interface
 * Handles WebSocket communication with Flask-SocketIO backend
 */

class ShortageSimClient {
    constructor() {
        this.socket = io();
        this.sessionId = null;
        this.simulationRunning = false;
        this.currentPeriod = 0;
        this.simulationResults = null;
        this.resultsChart = null;
        
        this.setupSocketHandlers();
        this.setupUIHandlers();
    }
    
    setupSocketHandlers() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ Connected to ShortageSim server');
            this.showNotification('Connected to server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.showNotification('Disconnected from server', 'warning');
        });
        
        // Session management
        this.socket.on('session_created', (data) => {
            this.sessionId = data.session_id;
            console.log('Session created:', this.sessionId);
        });
        
        // Simulation events
        this.socket.on('simulation_started', (data) => {
            console.log('Simulation started:', data.session_id);
            this.simulationRunning = true;
            this.updateButtonStates();
        });
        
        this.socket.on('simulation_status', (data) => {
            if (data.session_id === this.sessionId) {
                this.updateStatus(data.status, data.detail, data.progress);
            }
        });
        
        this.socket.on('period_completed', (data) => {
            if (data.session_id === this.sessionId) {
                this.currentPeriod = data.period;
                this.addTimelineItem(data.period, data.data);
                this.updateChart(data.period, data.data);
            }
        });
        
        this.socket.on('simulation_results', (data) => {
            if (data.session_id === this.sessionId) {
                this.simulationResults = data.results;
                this.displayFinalResults(data.results);
                this.simulationRunning = false;
                this.updateButtonStates();
            }
        });
        
        this.socket.on('simulation_stopped', (data) => {
            if (data.session_id === this.sessionId) {
                this.simulationRunning = false;
                this.updateButtonStates();
                this.showNotification('Simulation stopped', 'warning');
            }
        });
        
        this.socket.on('simulation_error', (data) => {
            if (data.session_id === this.sessionId) {
                this.simulationRunning = false;
                this.updateButtonStates();
                this.showNotification('Simulation Error: ' + data.error, 'error');
                this.updateStatus('error', 'Simulation failed: ' + data.error, 0);
            }
        });
    }
    
    setupUIHandlers() {
        // Load default configuration on startup
        this.loadDefaultConfig();
        
        // Simulation type change handler
        document.getElementById('simulationType').addEventListener('change', (e) => {
            const monteCarloRuns = document.getElementById('monteCarloRuns');
            if (e.target.value === 'monte_carlo') {
                monteCarloRuns.style.display = 'block';
            } else {
                monteCarloRuns.style.display = 'none';
            }
        });
    }
    
    async loadDefaultConfig() {
        try {
            const response = await fetch('/api/config/defaults');
            const data = await response.json();
            
            // Populate form with default values
            const config = data.config;
            const prompts = data.prompts;
            
            // Basic config
            document.getElementById('nManufacturers').value = config.n_manufacturers;
            document.getElementById('nPeriods').value = config.n_periods;
            document.getElementById('disruptionProb').value = config.disruption_probability;
            document.getElementById('disruptionMagnitude').value = config.disruption_magnitude;
            document.getElementById('unitProfit').value = config.unit_profit;
            document.getElementById('capacityCost').value = config.capacity_cost;
            document.getElementById('holdingCost').value = config.holding_cost;
            document.getElementById('stockoutPenalty').value = config.stockout_penalty;
            document.getElementById('llmModel').value = config.llm_model;
            document.getElementById('temperature').value = config.temperature;
            document.getElementById('maxRetries').value = config.max_retries;
            
            // Prompts
            document.getElementById('manufacturerSystemPrompt').value = prompts.manufacturer_system;
            document.getElementById('buyerSystemPrompt').value = prompts.buyer_system;
            document.getElementById('fdaSystemPrompt').value = prompts.fda_system;
            
        } catch (error) {
            console.error('Failed to load default config:', error);
            this.showNotification('Failed to load default configuration', 'error');
        }
    }
    
    runSimulation() {
        if (this.simulationRunning || !this.sessionId) {
            return;
        }
        
        // Collect configuration from form
        const config = this.getConfiguration();
        
        // Validate configuration
        if (!this.validateConfiguration(config)) {
            return;
        }
        
        // Clear previous results
        this.clearPreviousResults();
        
        // Send simulation request
        this.socket.emit('run_simulation', {
            session_id: this.sessionId,
            config: config
        });
        
        this.simulationRunning = true;
        this.updateButtonStates();
    }
    
    stopSimulation() {
        if (!this.simulationRunning || !this.sessionId) {
            return;
        }
        
        this.socket.emit('stop_simulation', {
            session_id: this.sessionId
        });
    }
    
    resetSimulation() {
        this.stopSimulation();
        
        // Clear UI
        this.clearPreviousResults();
        this.updateStatus('idle', 'Ready to run simulation', 0);
        
        // Reset variables
        this.currentPeriod = 0;
        this.simulationResults = null;
        this.simulationRunning = false;
        this.updateButtonStates();
        
        this.showNotification('Simulation reset', 'success');
    }
    
    getConfiguration() {
        return {
            n_manufacturers: parseInt(document.getElementById('nManufacturers').value),
            n_periods: parseInt(document.getElementById('nPeriods').value),
            disruption_probability: parseFloat(document.getElementById('disruptionProb').value),
            disruption_magnitude: parseFloat(document.getElementById('disruptionMagnitude').value),
            force_disruption: document.getElementById('forceDisruption').checked,
            unit_profit: parseFloat(document.getElementById('unitProfit').value),
            capacity_cost: parseFloat(document.getElementById('capacityCost').value),
            holding_cost: parseFloat(document.getElementById('holdingCost').value),
            stockout_penalty: parseFloat(document.getElementById('stockoutPenalty').value),
            llm_model: document.getElementById('llmModel').value,
            temperature: parseFloat(document.getElementById('temperature').value),
            api_key: document.getElementById('apiKey').value || null,
            max_retries: parseInt(document.getElementById('maxRetries').value),
            simulation_type: document.getElementById('simulationType').value,
            num_runs: parseInt(document.getElementById('numRuns').value) || 1,
            random_seed: parseInt(document.getElementById('randomSeed').value) || null,
            log_level: document.getElementById('logLevel').value,
            prompts: {
                manufacturer_system: document.getElementById('manufacturerSystemPrompt').value,
                buyer_system: document.getElementById('buyerSystemPrompt').value,
                fda_system: document.getElementById('fdaSystemPrompt').value
            }
        };
    }
    
    validateConfiguration(config) {
        const errors = [];
        
        if (config.n_manufacturers < 2 || config.n_manufacturers > 10) {
            errors.push('Number of manufacturers must be between 2 and 10');
        }
        
        if (config.n_periods < 1 || config.n_periods > 12) {
            errors.push('Number of periods must be between 1 and 12');
        }
        
        if (config.disruption_probability < 0 || config.disruption_probability > 1) {
            errors.push('Disruption probability must be between 0 and 1');
        }
        
        if (errors.length > 0) {
            this.showNotification('Configuration errors: ' + errors.join(', '), 'error');
            return false;
        }
        
        return true;
    }
    
    clearPreviousResults() {
        // Clear timeline
        document.getElementById('timeline').innerHTML = `
            <div class="timeline-item">
                <div class="period-header">
                    <div class="period-number">Starting simulation...</div>
                </div>
                <p>Initializing agents and market environment...</p>
            </div>
        `;
        
        // Clear chart
        if (this.resultsChart) {
            this.resultsChart.destroy();
            this.resultsChart = null;
        }
        
        // Hide results summary
        document.getElementById('resultsSummary').style.display = 'none';
    }
    
    updateStatus(status, detail, progress) {
        const statusElement = document.getElementById('simulationStatus');
        const statusText = document.getElementById('statusText');
        const statusDetail = document.getElementById('statusDetail');
        const progressFill = document.getElementById('progressFill');
        
        // Update status class
        statusElement.className = `simulation-status status-${status}`;
        
        // Update text
        statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        statusDetail.textContent = detail;
        
        // Update progress
        progressFill.style.width = progress + '%';
    }
    
    updateButtonStates() {
        const runButton = document.getElementById('runButton');
        const stopButton = document.getElementById('stopButton');
        const runButtonText = document.getElementById('runButtonText');
        
        if (this.simulationRunning) {
            runButton.disabled = true;
            stopButton.disabled = false;
            runButtonText.innerHTML = '<div class="loading-spinner"></div> Running...';
        } else {
            runButton.disabled = false;
            stopButton.disabled = true;
            runButtonText.innerHTML = '<i class="fas fa-play"></i> Run Simulation';
        }
    }
    
    addTimelineItem(period, data) {
        const timeline = document.getElementById('timeline');
        const timelineItem = document.createElement('div');
        timelineItem.className = 'timeline-item';
        
        const shortageClass = data.shortage_percentage > 0.1 ? 'shortage' : 'supply';
        
        timelineItem.innerHTML = `
            <div class="period-header">
                <div class="period-number">Period ${period + 1}</div>
                <div class="period-metrics">
                    <div class="metric ${shortageClass}">
                        Shortage: ${(data.shortage_percentage * 100).toFixed(1)}%
                    </div>
                    <div class="metric supply">
                        Supply: ${data.total_supply.toFixed(2)}
                    </div>
                    <div class="metric">
                        Demand: ${data.total_demand.toFixed(2)}
                    </div>
                </div>
            </div>
            
            ${data.fda_announcement ? `<div class="fda-announcement">${data.fda_announcement}</div>` : ''}
            
            <div class="agent-decisions">
                <div class="agent-card fda">
                    <div class="agent-title">🏛️ FDA Regulator</div>
                    <div class="agent-decision">
                        <strong>Decision:</strong> ${data.agent_decisions.fda.announcement_type || 'none'}
                    </div>
                    <div class="agent-reasoning">
                        "${data.agent_decisions.fda.reasoning || 'No action taken'}"
                    </div>
                </div>
                
                <div class="agent-card manufacturer">
                    <div class="agent-title">🏭 Manufacturers</div>
                    ${data.agent_decisions.manufacturers.slice(0, 2).map(mfr => `
                        <div class="agent-decision">
                            <strong>Mfr ${mfr.id}:</strong> ${(mfr.investment * 100).toFixed(0)}% investment
                        </div>
                    `).join('')}
                    <div class="agent-reasoning">
                        "${data.agent_decisions.manufacturers[0]?.reasoning || 'No reasoning available'}"
                    </div>
                </div>
                
                <div class="agent-card buyer">
                    <div class="agent-title">🏥 Healthcare Buyers</div>
                    <div class="agent-decision">
                        <strong>Order:</strong> ${data.agent_decisions.buyer.demand_quantity?.toFixed(2) || '1.00'} units
                    </div>
                    <div class="agent-reasoning">
                        "${data.agent_decisions.buyer.reasoning || 'Standard procurement'}"
                    </div>
                </div>
            </div>
        `;
        
        timeline.appendChild(timelineItem);
        timeline.scrollTop = timeline.scrollHeight;
    }
    
    updateChart(period, data) {
        if (!this.resultsChart) {
            const ctx = document.getElementById('resultsChart').getContext('2d');
            this.resultsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Supply',
                            data: [],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            fill: true
                        },
                        {
                            label: 'Demand',
                            data: [],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            fill: false
                        },
                        {
                            label: 'Shortage %',
                            data: [],
                            borderColor: '#f39c12',
                            backgroundColor: 'rgba(243, 156, 18, 0.1)',
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Quantity'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Shortage %'
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true
                        },
                        title: {
                            display: true,
                            text: 'Supply Chain Dynamics Over Time'
                        }
                    }
                }
            });
        }
        
        // Update chart data with real simulation data
        this.resultsChart.data.labels.push(`Period ${period + 1}`);
        this.resultsChart.data.datasets[0].data.push(data.total_supply);
        this.resultsChart.data.datasets[1].data.push(data.total_demand);
        this.resultsChart.data.datasets[2].data.push(data.shortage_percentage * 100);
        
        this.resultsChart.update();
    }
    
    displayFinalResults(results) {
        if (!results) return;
        
        const metrics = results.summary_metrics;
        
        // Update summary metrics
        document.getElementById('peakShortage').textContent = (metrics.peak_shortage_percentage * 100).toFixed(1) + '%';
        document.getElementById('totalCost').textContent = results.buyer_total_cost.toFixed(1);
        document.getElementById('resolutionTime').textContent = (metrics.time_to_resolution || 'Not resolved') + ' periods';
        document.getElementById('fdaInterventions').textContent = results.fda_announcements?.length || 0;
        
        // Show results summary
        document.getElementById('resultsSummary').style.display = 'block';
        
        this.showNotification('Simulation completed successfully!', 'success');
    }
    
    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        const notificationText = document.getElementById('notificationText');
        
        notificationText.textContent = message;
        notification.className = `notification ${type} show`;
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 4000);
    }
    
    async exportResults(format) {
        if (!this.simulationResults || !this.sessionId) {
            this.showNotification('No results to export', 'warning');
            return;
        }
        
        try {
            const response = await fetch(`/api/results/${this.sessionId}/download/${format}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `shortagesim_results_${this.sessionId}.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showNotification(`Results exported as ${format.toUpperCase()}`, 'success');
            } else {
                throw new Error('Export failed');
            }
        } catch (error) {
            this.showNotification(`Export failed: ${error.message}`, 'error');
        }
    }
    
    generateReport() {
        this.showNotification('PDF report generation coming soon...', 'warning');
    }
}

// Global functions for HTML event handlers
let simClient;

function switchTab(tabName) {
    // Remove active class from all tabs and content
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab and content
    event.target.classList.add('active');
    document.getElementById(tabName + '-tab').classList.add('active');
}

function toggleCollapsible(element) {
    element.classList.toggle('open');
    const content = element.nextElementSibling;
    content.classList.toggle('open');
}

function runSimulation() {
    simClient.runSimulation();
}

function stopSimulation() {
    simClient.stopSimulation();
}

function resetSimulation() {
    simClient.resetSimulation();
}

function exportResults(format) {
    simClient.exportResults(format);
}

function generateReport() {
    simClient.generateReport();
}

// Initialize client when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    simClient = new ShortageSimClient();
    console.log('🚀 ShortageSim web client initialized');
});