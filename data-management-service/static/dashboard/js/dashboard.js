// Data Reconciliation Dashboard JavaScript

// API URL
const API_URL = '/api';

// Chart objects
let matchPercentageChart = null;
let issuesBySeverityChart = null;
let issuesByFieldChart = null;
let configPerformanceChart = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Load initial data
    loadDashboardData();
    
    // Set up event listeners
    document.getElementById('timeRange').addEventListener('change', loadDashboardData);
    document.getElementById('newConfigBtn').addEventListener('click', showNewConfigModal);
    
    // Refresh data every 5 minutes
    setInterval(loadDashboardData, 5 * 60 * 1000);
});

// Load dashboard data
async function loadDashboardData() {
    const days = document.getElementById('timeRange').value;
    
    try {
        // Load summary
        const summary = await fetchData(`${API_URL}/reconciliation/dashboard/summary`);
        updateSummaryStats(summary);
        
        // Load match percentage time series
        const matchPercentage = await fetchData(`${API_URL}/reconciliation/dashboard/time-series/match-percentage?days=${days}`);
        updateMatchPercentageChart(matchPercentage);
        
        // Load issues by severity
        const issuesBySeverity = await fetchData(`${API_URL}/reconciliation/dashboard/issues-by-severity?days=${days}`);
        updateIssuesBySeverityChart(issuesBySeverity);
        
        // Load issues by field
        const issuesByField = await fetchData(`${API_URL}/reconciliation/dashboard/issues-by-field?days=${days}`);
        updateIssuesByFieldChart(issuesByField);
        
        // Load config performance
        const configPerformance = await fetchData(`${API_URL}/reconciliation/dashboard/config-performance?days=${days}`);
        updateConfigPerformanceChart(configPerformance);
        
        // Load configurations
        const configs = await fetchData(`${API_URL}/reconciliation/configs`);
        updateConfigsTable(configs);
        
        // Load tasks
        const tasks = await fetchData(`${API_URL}/reconciliation/tasks?limit=10`);
        updateTasksTable(tasks);
        
        // Load results
        const results = await fetchData(`${API_URL}/reconciliation/results?limit=10`);
        updateResultsTable(results);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data. Please try again later.');
    }
}

// Fetch data from API
async function fetchData(url) {
    const response = await fetch(url);
    
    if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
}

// Update summary statistics
function updateSummaryStats(summary) {
    const container = document.getElementById('summaryStats');
    
    container.innerHTML = `
        <div class="col-md-3 col-sm-6">
            <div class="stat-card">
                <div class="stat-value">${summary.total_configs}</div>
                <div class="stat-label">Total Configurations</div>
            </div>
        </div>
        <div class="col-md-3 col-sm-6">
            <div class="stat-card">
                <div class="stat-value">${summary.total_tasks_24h}</div>
                <div class="stat-label">Tasks (24h)</div>
            </div>
        </div>
        <div class="col-md-3 col-sm-6">
            <div class="stat-card">
                <div class="stat-value">${summary.match_percentage_24h.toFixed(2)}%</div>
                <div class="stat-label">Match Percentage (24h)</div>
            </div>
        </div>
        <div class="col-md-3 col-sm-6">
            <div class="stat-card">
                <div class="stat-value">${summary.total_issues_24h}</div>
                <div class="stat-label">Issues (24h)</div>
            </div>
        </div>
    `;
}

// Update match percentage chart
function updateMatchPercentageChart(data) {
    const ctx = document.getElementById('matchPercentageChart').getContext('2d');
    
    // Prepare data
    const labels = data.data.map(point => {
        const date = new Date(point.timestamp);
        return date.toLocaleDateString();
    });
    
    const values = data.data.map(point => point.value);
    
    // Create or update chart
    if (matchPercentageChart) {
        matchPercentageChart.data.labels = labels;
        matchPercentageChart.data.datasets[0].data = values;
        matchPercentageChart.update();
    } else {
        matchPercentageChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Match Percentage',
                    data: values,
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: Math.max(0, Math.min(...values) - 5),
                        max: Math.min(100, Math.max(...values) + 5),
                        title: {
                            display: true,
                            text: 'Percentage'
                        }
                    }
                }
            }
        });
    }
}

// Update issues by severity chart
function updateIssuesBySeverityChart(data) {
    const ctx = document.getElementById('issuesBySeverityChart').getContext('2d');
    
    // Prepare data
    const labels = data.map(item => item.severity);
    const values = data.map(item => item.count);
    const colors = {
        'ERROR': '#dc3545',
        'WARNING': '#ffc107',
        'INFO': '#0dcaf0'
    };
    
    // Create or update chart
    if (issuesBySeverityChart) {
        issuesBySeverityChart.data.labels = labels;
        issuesBySeverityChart.data.datasets[0].data = values;
        issuesBySeverityChart.update();
    } else {
        issuesBySeverityChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: labels.map(label => colors[label] || '#6c757d')
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    }
}

// Update issues by field chart
function updateIssuesByFieldChart(data) {
    const ctx = document.getElementById('issuesByFieldChart').getContext('2d');
    
    // Prepare data
    const labels = data.map(item => item.field);
    const errorValues = data.map(item => item.error_count);
    const warningValues = data.map(item => item.warning_count);
    const infoValues = data.map(item => item.info_count);
    
    // Create or update chart
    if (issuesByFieldChart) {
        issuesByFieldChart.data.labels = labels;
        issuesByFieldChart.data.datasets[0].data = errorValues;
        issuesByFieldChart.data.datasets[1].data = warningValues;
        issuesByFieldChart.data.datasets[2].data = infoValues;
        issuesByFieldChart.update();
    } else {
        issuesByFieldChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Error',
                        data: errorValues,
                        backgroundColor: '#dc3545'
                    },
                    {
                        label: 'Warning',
                        data: warningValues,
                        backgroundColor: '#ffc107'
                    },
                    {
                        label: 'Info',
                        data: infoValues,
                        backgroundColor: '#0dcaf0'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Update config performance chart
function updateConfigPerformanceChart(data) {
    const ctx = document.getElementById('configPerformanceChart').getContext('2d');
    
    // Prepare data
    const labels = data.map(item => item.name);
    const values = data.map(item => item.match_percentage);
    const issueValues = data.map(item => item.issue_count);
    
    // Create or update chart
    if (configPerformanceChart) {
        configPerformanceChart.data.labels = labels;
        configPerformanceChart.data.datasets[0].data = values;
        configPerformanceChart.data.datasets[1].data = issueValues;
        configPerformanceChart.update();
    } else {
        configPerformanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Match Percentage',
                        data: values,
                        backgroundColor: '#0d6efd',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Issue Count',
                        data: issueValues,
                        backgroundColor: '#dc3545',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        beginAtZero: false,
                        min: Math.max(0, Math.min(...values) - 5),
                        max: Math.min(100, Math.max(...values) + 5),
                        title: {
                            display: true,
                            text: 'Match Percentage'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Issue Count'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }
}

// Update configurations table
function updateConfigsTable(configs) {
    const tbody = document.getElementById('configsTable');
    
    tbody.innerHTML = '';
    
    configs.forEach(config => {
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>${config.name}</td>
            <td>${config.reconciliation_type}</td>
            <td>${config.primary_source.source_id}</td>
            <td>${config.secondary_source ? config.secondary_source.source_id : 'N/A'}</td>
            <td>
                <span class="badge ${config.enabled ? 'bg-success' : 'bg-secondary'}">
                    ${config.enabled ? 'Enabled' : 'Disabled'}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="viewConfig('${config.config_id}')">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-success" onclick="scheduleTask('${config.config_id}')">
                    <i class="bi bi-play"></i>
                </button>
                <button class="btn btn-sm ${config.enabled ? 'btn-warning' : 'btn-info'}" 
                        onclick="${config.enabled ? 'disableConfig' : 'enableConfig'}('${config.config_id}')">
                    <i class="bi ${config.enabled ? 'bi-pause' : 'bi-check'}"></i>
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// Update tasks table
function updateTasksTable(tasks) {
    const tbody = document.getElementById('tasksTable');
    
    tbody.innerHTML = '';
    
    tasks.forEach(task => {
        const row = document.createElement('tr');
        
        const statusClass = {
            'PENDING': 'bg-secondary',
            'RUNNING': 'bg-primary',
            'COMPLETED': 'bg-success',
            'FAILED': 'bg-danger'
        }[task.status] || 'bg-secondary';
        
        row.innerHTML = `
            <td>${task.task_id.substring(0, 8)}...</td>
            <td>${task.config_id.substring(0, 8)}...</td>
            <td>${new Date(task.scheduled_time).toLocaleString()}</td>
            <td>
                <span class="badge ${statusClass}">
                    ${task.status}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="viewTask('${task.task_id}')">
                    <i class="bi bi-eye"></i>
                </button>
                ${task.status === 'PENDING' ? `
                <button class="btn btn-sm btn-success" onclick="runTask('${task.task_id}')">
                    <i class="bi bi-play"></i>
                </button>
                ` : ''}
                ${task.result_id ? `
                <button class="btn btn-sm btn-info" onclick="viewResult('${task.result_id}')">
                    <i class="bi bi-clipboard-check"></i>
                </button>
                ` : ''}
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// Update results table
function updateResultsTable(results) {
    const tbody = document.getElementById('resultsTable');
    
    tbody.innerHTML = '';
    
    results.forEach(result => {
        const row = document.createElement('tr');
        
        const matchPercentage = result.total_records > 0 
            ? (result.matched_records / result.total_records * 100).toFixed(2)
            : 'N/A';
        
        const statusClass = {
            'RUNNING': 'bg-primary',
            'COMPLETED': 'bg-success',
            'FAILED': 'bg-danger'
        }[result.status] || 'bg-secondary';
        
        row.innerHTML = `
            <td>${result.result_id.substring(0, 8)}...</td>
            <td>${result.config_id.substring(0, 8)}...</td>
            <td>${new Date(result.start_time).toLocaleString()}</td>
            <td>${matchPercentage}%</td>
            <td>${result.issues ? result.issues.length : 0}</td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="viewResult('${result.result_id}')">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-info" onclick="downloadReport('${result.result_id}')">
                    <i class="bi bi-download"></i>
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// Show error message
function showError(message) {
    alert(message);
}

// Placeholder functions for actions
function showNewConfigModal() {
    alert('New configuration modal would open here');
}

function viewConfig(configId) {
    alert(`View configuration: ${configId}`);
}

function scheduleTask(configId) {
    alert(`Schedule task for configuration: ${configId}`);
}

function enableConfig(configId) {
    alert(`Enable configuration: ${configId}`);
}

function disableConfig(configId) {
    alert(`Disable configuration: ${configId}`);
}

function viewTask(taskId) {
    alert(`View task: ${taskId}`);
}

function runTask(taskId) {
    alert(`Run task: ${taskId}`);
}

function viewResult(resultId) {
    alert(`View result: ${resultId}`);
}

function downloadReport(resultId) {
    alert(`Download report for result: ${resultId}`);
}
