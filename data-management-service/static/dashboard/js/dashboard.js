// Data Reconciliation Dashboard JavaScript

// API URL
const API_URL = '';

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

// Modal elements
let configModal = null;
let taskModal = null;
let resultModal = null;

// Show loading indicator
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></div>';
    }
}

// Show new configuration modal
function showNewConfigModal() {
    // Create modal if it doesn't exist
    if (!configModal) {
        const modalHtml = `
            <div class="modal fade" id="configModal" tabindex="-1" aria-labelledby="configModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="configModalLabel">New Reconciliation Configuration</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="configForm">
                                <div class="mb-3">
                                    <label for="configName" class="form-label">Name</label>
                                    <input type="text" class="form-control" id="configName" required>
                                </div>
                                <div class="mb-3">
                                    <label for="configDescription" class="form-label">Description</label>
                                    <textarea class="form-control" id="configDescription" rows="2"></textarea>
                                </div>
                                <div class="mb-3">
                                    <label for="reconciliationType" class="form-label">Reconciliation Type</label>
                                    <select class="form-select" id="reconciliationType" required>
                                        <option value="cross_source">Cross Source</option>
                                        <option value="temporal">Temporal</option>
                                        <option value="derived">Derived</option>
                                        <option value="custom">Custom</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="primarySource" class="form-label">Primary Source</label>
                                    <div class="input-group">
                                        <input type="text" class="form-control" id="primarySourceId" placeholder="Source ID" required>
                                        <select class="form-select" id="primarySourceType" required>
                                            <option value="ohlcv">OHLCV</option>
                                            <option value="tick">Tick</option>
                                            <option value="alternative">Alternative</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label for="secondarySource" class="form-label">Secondary Source</label>
                                    <div class="input-group">
                                        <input type="text" class="form-control" id="secondarySourceId" placeholder="Source ID">
                                        <select class="form-select" id="secondarySourceType">
                                            <option value="">None</option>
                                            <option value="ohlcv">OHLCV</option>
                                            <option value="tick">Tick</option>
                                            <option value="alternative">Alternative</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <label for="schedule" class="form-label">Schedule (Cron Expression)</label>
                                    <input type="text" class="form-control" id="schedule" placeholder="e.g., 0 0 * * *">
                                </div>
                                <div class="form-check mb-3">
                                    <input class="form-check-input" type="checkbox" id="enabled" checked>
                                    <label class="form-check-label" for="enabled">Enabled</label>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="saveConfigBtn">Save</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal to document
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Get modal element
        configModal = new bootstrap.Modal(document.getElementById('configModal'));

        // Add event listener for save button
        document.getElementById('saveConfigBtn').addEventListener('click', saveConfig);
    }

    // Reset form
    document.getElementById('configForm').reset();

    // Show modal
    configModal.show();
}

// Save configuration
async function saveConfig() {
    try {
        // Get form values
        const name = document.getElementById('configName').value;
        const description = document.getElementById('configDescription').value;
        const reconciliationType = document.getElementById('reconciliationType').value;
        const primarySourceId = document.getElementById('primarySourceId').value;
        const primarySourceType = document.getElementById('primarySourceType').value;
        const secondarySourceId = document.getElementById('secondarySourceId').value;
        const secondarySourceType = document.getElementById('secondarySourceType').value;
        const schedule = document.getElementById('schedule').value;
        const enabled = document.getElementById('enabled').checked;

        // Create request body
        const requestBody = {
            name,
            description,
            reconciliation_type: reconciliationType,
            primary_source: {
                source_id: primarySourceId,
                source_type: primarySourceType,
                query_params: {},
                filters: {}
            },
            schedule,
            enabled
        };

        // Add secondary source if provided
        if (secondarySourceId && secondarySourceType) {
            requestBody.secondary_source = {
                source_id: secondarySourceId,
                source_type: secondarySourceType,
                query_params: {},
                filters: {}
            };
        }

        // Send request
        const response = await fetch(`${API_URL}/reconciliation/configs`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Failed to create configuration: ${response.status} ${response.statusText}`);
        }

        // Hide modal
        configModal.hide();

        // Reload dashboard data
        loadDashboardData();

        // Show success message
        showSuccess('Configuration created successfully');
    } catch (error) {
        console.error('Error creating configuration:', error);
        showError(`Failed to create configuration: ${error.message}`);
    }
}

// View configuration details
async function viewConfig(configId) {
    try {
        // Show loading
        showLoading('configDetails');

        // Fetch configuration
        const config = await fetchData(`${API_URL}/reconciliation/configs/${configId}`);

        // Create modal if it doesn't exist
        if (!configModal) {
            const modalHtml = `
                <div class="modal fade" id="configModal" tabindex="-1" aria-labelledby="configModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-lg">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="configModalLabel">Configuration Details</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body" id="configDetails">
                                <!-- Configuration details will be loaded here -->
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Add modal to document
            document.body.insertAdjacentHTML('beforeend', modalHtml);

            // Get modal element
            configModal = new bootstrap.Modal(document.getElementById('configModal'));
        }

        // Update modal content
        const configDetails = document.getElementById('configDetails');
        configDetails.innerHTML = `
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">ID:</div>
                <div class="col-md-9">${config.config_id}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Name:</div>
                <div class="col-md-9">${config.name}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Description:</div>
                <div class="col-md-9">${config.description || 'N/A'}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Type:</div>
                <div class="col-md-9">${config.reconciliation_type}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Primary Source:</div>
                <div class="col-md-9">${config.primary_source.source_id} (${config.primary_source.source_type})</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Secondary Source:</div>
                <div class="col-md-9">${config.secondary_source ? `${config.secondary_source.source_id} (${config.secondary_source.source_type})` : 'N/A'}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Schedule:</div>
                <div class="col-md-9">${config.schedule || 'N/A'}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Status:</div>
                <div class="col-md-9">
                    <span class="badge ${config.enabled ? 'bg-success' : 'bg-secondary'}">
                        ${config.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                </div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Created:</div>
                <div class="col-md-9">${new Date(config.created_at).toLocaleString()}</div>
            </div>
        `;

        // Show modal
        configModal.show();
    } catch (error) {
        console.error('Error viewing configuration:', error);
        showError(`Failed to view configuration: ${error.message}`);
    }
}

// Schedule a reconciliation task
async function scheduleTask(configId) {
    try {
        // Send request
        const response = await fetch(`${API_URL}/reconciliation/tasks`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                config_id: configId,
                scheduled_time: new Date().toISOString()
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to schedule task: ${response.status} ${response.statusText}`);
        }

        // Reload dashboard data
        loadDashboardData();

        // Show success message
        showSuccess('Task scheduled successfully');
    } catch (error) {
        console.error('Error scheduling task:', error);
        showError(`Failed to schedule task: ${error.message}`);
    }
}

// Enable configuration
async function enableConfig(configId) {
    try {
        // Send request
        const response = await fetch(`${API_URL}/reconciliation/configs/${configId}/enable`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`Failed to enable configuration: ${response.status} ${response.statusText}`);
        }

        // Reload dashboard data
        loadDashboardData();

        // Show success message
        showSuccess('Configuration enabled successfully');
    } catch (error) {
        console.error('Error enabling configuration:', error);
        showError(`Failed to enable configuration: ${error.message}`);
    }
}

// Disable configuration
async function disableConfig(configId) {
    try {
        // Send request
        const response = await fetch(`${API_URL}/reconciliation/configs/${configId}/disable`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`Failed to disable configuration: ${response.status} ${response.statusText}`);
        }

        // Reload dashboard data
        loadDashboardData();

        // Show success message
        showSuccess('Configuration disabled successfully');
    } catch (error) {
        console.error('Error disabling configuration:', error);
        showError(`Failed to disable configuration: ${error.message}`);
    }
}

// View task details
async function viewTask(taskId) {
    try {
        // Show loading
        showLoading('taskDetails');

        // Fetch task
        const task = await fetchData(`${API_URL}/reconciliation/tasks/${taskId}`);

        // Create modal if it doesn't exist
        if (!taskModal) {
            const modalHtml = `
                <div class="modal fade" id="taskModal" tabindex="-1" aria-labelledby="taskModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-lg">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="taskModalLabel">Task Details</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body" id="taskDetails">
                                <!-- Task details will be loaded here -->
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                <button type="button" class="btn btn-success" id="runTaskBtn">Run Task</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Add modal to document
            document.body.insertAdjacentHTML('beforeend', modalHtml);

            // Get modal element
            taskModal = new bootstrap.Modal(document.getElementById('taskModal'));

            // Add event listener for run button
            document.getElementById('runTaskBtn').addEventListener('click', function() {
                const taskId = this.dataset.taskId;
                if (taskId) {
                    runTask(taskId);
                }
            });
        }

        // Update modal content
        const taskDetails = document.getElementById('taskDetails');
        taskDetails.innerHTML = `
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">ID:</div>
                <div class="col-md-9">${task.task_id}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Configuration:</div>
                <div class="col-md-9">${task.config_id}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Status:</div>
                <div class="col-md-9">
                    <span class="badge ${task.status === 'COMPLETED' ? 'bg-success' : task.status === 'FAILED' ? 'bg-danger' : task.status === 'RUNNING' ? 'bg-primary' : 'bg-secondary'}">
                        ${task.status}
                    </span>
                </div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Scheduled Time:</div>
                <div class="col-md-9">${new Date(task.scheduled_time).toLocaleString()}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Start Time:</div>
                <div class="col-md-9">${task.start_time ? new Date(task.start_time).toLocaleString() : 'N/A'}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">End Time:</div>
                <div class="col-md-9">${task.end_time ? new Date(task.end_time).toLocaleString() : 'N/A'}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Result:</div>
                <div class="col-md-9">${task.result_id ? `<a href="#" onclick="viewResult('${task.result_id}'); return false;">${task.result_id}</a>` : 'N/A'}</div>
            </div>
        `;

        // Update run button
        const runTaskBtn = document.getElementById('runTaskBtn');
        runTaskBtn.dataset.taskId = task.task_id;
        runTaskBtn.style.display = task.status === 'PENDING' ? 'block' : 'none';

        // Show modal
        taskModal.show();
    } catch (error) {
        console.error('Error viewing task:', error);
        showError(`Failed to view task: ${error.message}`);
    }
}

// Run a reconciliation task
async function runTask(taskId) {
    try {
        // Send request
        const response = await fetch(`${API_URL}/reconciliation/tasks/${taskId}/run`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`Failed to run task: ${response.status} ${response.statusText}`);
        }

        // Get result ID
        const result = await response.json();

        // Hide task modal if open
        if (taskModal) {
            taskModal.hide();
        }

        // Reload dashboard data
        loadDashboardData();

        // Show success message
        showSuccess('Task started successfully');

        // View result
        if (result.result_id) {
            viewResult(result.result_id);
        }
    } catch (error) {
        console.error('Error running task:', error);
        showError(`Failed to run task: ${error.message}`);
    }
}

// View result details
async function viewResult(resultId) {
    try {
        // Show loading
        showLoading('resultDetails');

        // Fetch result
        const result = await fetchData(`${API_URL}/reconciliation/results/${resultId}`);

        // Create modal if it doesn't exist
        if (!resultModal) {
            const modalHtml = `
                <div class="modal fade" id="resultModal" tabindex="-1" aria-labelledby="resultModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-lg">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="resultModalLabel">Result Details</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body" id="resultDetails">
                                <!-- Result details will be loaded here -->
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                <button type="button" class="btn btn-info" id="downloadReportBtn">Download Report</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Add modal to document
            document.body.insertAdjacentHTML('beforeend', modalHtml);

            // Get modal element
            resultModal = new bootstrap.Modal(document.getElementById('resultModal'));

            // Add event listener for download button
            document.getElementById('downloadReportBtn').addEventListener('click', function() {
                const resultId = this.dataset.resultId;
                if (resultId) {
                    downloadReport(resultId);
                }
            });
        }

        // Calculate match percentage
        const matchPercentage = result.total_records > 0
            ? (result.matched_records / result.total_records * 100).toFixed(2)
            : 'N/A';

        // Update modal content
        const resultDetails = document.getElementById('resultDetails');
        resultDetails.innerHTML = `
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">ID:</div>
                <div class="col-md-9">${result.result_id}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Configuration:</div>
                <div class="col-md-9">${result.config_id}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Status:</div>
                <div class="col-md-9">
                    <span class="badge ${result.status === 'COMPLETED' ? 'bg-success' : result.status === 'FAILED' ? 'bg-danger' : result.status === 'RUNNING' ? 'bg-primary' : 'bg-secondary'}">
                        ${result.status}
                    </span>
                </div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Start Time:</div>
                <div class="col-md-9">${new Date(result.start_time).toLocaleString()}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">End Time:</div>
                <div class="col-md-9">${result.end_time ? new Date(result.end_time).toLocaleString() : 'N/A'}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Total Records:</div>
                <div class="col-md-9">${result.total_records}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Matched Records:</div>
                <div class="col-md-9">${result.matched_records}</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Match Percentage:</div>
                <div class="col-md-9">${matchPercentage}%</div>
            </div>
            <div class="row mb-3">
                <div class="col-md-3 fw-bold">Issues:</div>
                <div class="col-md-9">${result.issues ? result.issues.length : 0}</div>
            </div>

            ${result.issues && result.issues.length > 0 ? `
                <h6 class="mt-4">Issues</h6>
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>Field</th>
                                <th>Severity</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${result.issues.map(issue => `
                                <tr>
                                    <td>${issue.field}</td>
                                    <td>
                                        <span class="badge ${issue.severity === 'ERROR' ? 'bg-danger' : issue.severity === 'WARNING' ? 'bg-warning' : 'bg-info'}">
                                            ${issue.severity}
                                        </span>
                                    </td>
                                    <td>${issue.description}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : ''}
        `;

        // Update download button
        document.getElementById('downloadReportBtn').dataset.resultId = result.result_id;

        // Show modal
        resultModal.show();
    } catch (error) {
        console.error('Error viewing result:', error);
        showError(`Failed to view result: ${error.message}`);
    }
}

// Download reconciliation report
function downloadReport(resultId) {
    try {
        // Create URL
        const url = `${API_URL}/reconciliation/results/${resultId}/report`;

        // Open in new tab
        window.open(url, '_blank');
    } catch (error) {
        console.error('Error downloading report:', error);
        showError(`Failed to download report: ${error.message}`);
    }
}

// Show success message
function showSuccess(message) {
    // Create toast container if it doesn't exist
    if (!document.getElementById('toastContainer')) {
        const toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }

    // Create toast
    const toastId = `toast-${Date.now()}`;
    const toastHtml = `
        <div class="toast" id="${toastId}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-success text-white">
                <strong class="me-auto">Success</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;

    // Add toast to container
    document.getElementById('toastContainer').insertAdjacentHTML('beforeend', toastHtml);

    // Show toast
    const toast = new bootstrap.Toast(document.getElementById(toastId));
    toast.show();

    // Remove toast after it's hidden
    document.getElementById(toastId).addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

// Show error message
function showError(message) {
    // Create toast container if it doesn't exist
    if (!document.getElementById('toastContainer')) {
        const toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }

    // Create toast
    const toastId = `toast-${Date.now()}`;
    const toastHtml = `
        <div class="toast" id="${toastId}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-danger text-white">
                <strong class="me-auto">Error</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;

    // Add toast to container
    document.getElementById('toastContainer').insertAdjacentHTML('beforeend', toastHtml);

    // Show toast
    const toast = new bootstrap.Toast(document.getElementById(toastId));
    toast.show();

    // Remove toast after it's hidden
    document.getElementById(toastId).addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}
