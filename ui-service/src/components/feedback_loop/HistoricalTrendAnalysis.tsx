import React from 'react';
import { Card, Chart } from '../ui-library';

interface HistoricalTrendAnalysisProps {
  parameterName: string;
  analytics: any | null;
}

const HistoricalTrendAnalysis: React.FC<HistoricalTrendAnalysisProps> = ({
  parameterName,
  analytics
}) => {
  if (!analytics) {
    return (
      <Card title={`${parameterName} Historical Trend Analysis`}>
        <div className="no-data">
          <p>No historical trend data available for this parameter.</p>
        </div>
      </Card>
    );
  }

  // Extract data from analytics
  const trendWindows = analytics.trend_windows || [];
  const correlationAnalysis = analytics.correlation_analysis || {};
  const lookbackDays = analytics.lookback_days || 90;
  const windowSize = analytics.window_size || 7;

  // Group trend data by parameter value
  const valueGroups = trendWindows.reduce((groups, trend) => {
    const valueKey = trend.parameter_value?.toString() || 'unknown';
    if (!groups[valueKey]) {
      groups[valueKey] = [];
    }
    groups[valueKey].push(trend);
    return groups;
  }, {});

  // Prepare data for trend visualization
  // This is a simplified example - in a real application, you'd have time series data
  const trendChartData = {
    labels: Object.keys(valueGroups),
    datasets: [
      {
        label: 'Win Rate',
        data: Object.values(valueGroups).map(
          (trends: any[]) => trends[0]?.latest_win_rate || 0
        ),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      },
      {
        label: 'Profit Factor',
        data: Object.values(valueGroups).map(
          (trends: any[]) => {
            const val = trends[0]?.latest_profit_factor;
            return val && val !== Infinity ? Math.min(val, 5) : 0;
          }
        ),
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      },
      {
        label: 'Avg Profit',
        data: Object.values(valueGroups).map(
          (trends: any[]) => trends[0]?.latest_avg_profit || 0
        ),
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1
      }
    ]
  };

  return (
    <Card title={`${parameterName} Historical Trend Analysis`}>
      <div className="historical-trend-analysis">
        <div className="trend-summary">
          <div className="summary-header">
            <h4>Performance Trends (Last {lookbackDays} Days)</h4>
            <div className="period-info">
              Window Size: {windowSize} days
            </div>
          </div>
          
          <div className="trend-table-container">
            <table className="trend-table">
              <thead>
                <tr>
                  <th>Parameter Value</th>
                  <th>Win Rate Trend</th>
                  <th>Profit Factor Trend</th>
                  <th>Latest Win Rate</th>
                  <th>Latest Profit Factor</th>
                  <th>Sample Size</th>
                </tr>
              </thead>
              <tbody>
                {trendWindows.map((trend, index) => (
                  <tr key={index}>
                    <td>{trend.parameter_value?.toString() || 'N/A'}</td>
                    <td>
                      <span className={`trend-indicator ${trend.win_rate_trend}`}>
                        {trend.win_rate_trend === 'improving' ? '↑ Improving' :
                          trend.win_rate_trend === 'degrading' ? '↓ Degrading' :
                          '↔ Stable'}
                      </span>
                    </td>
                    <td>
                      <span className={`trend-indicator ${trend.profit_factor_trend}`}>
                        {trend.profit_factor_trend === 'improving' ? '↑ Improving' :
                          trend.profit_factor_trend === 'degrading' ? '↓ Degrading' :
                          '↔ Stable'}
                      </span>
                    </td>
                    <td>
                      {trend.latest_win_rate != null ? 
                        `${(trend.latest_win_rate * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td>
                      {trend.latest_profit_factor != null ?
                        trend.latest_profit_factor === Infinity ?
                          '∞' : trend.latest_profit_factor.toFixed(2)
                        : 'N/A'}
                    </td>
                    <td>{trend.sample_size || 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="trend-visualization">
          <div className="chart-container">
            <h4>Performance Metrics by Parameter Value</h4>
            <Chart 
              type="bar"
              data={trendChartData}
              options={{
                responsive: true,
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Value'
                    }
                  },
                  x: {
                    title: {
                      display: true,
                      text: 'Parameter Value'
                    }
                  }
                }
              }}
            />
          </div>
        </div>
        
        {/* Correlation Analysis */}
        <div className="correlation-analysis">
          <h4>Parameter-Performance Correlation</h4>
          
          {correlationAnalysis.status === 'completed' ? (
            <div className="correlation-results">
              <div className={`correlation-value ${Math.abs(correlationAnalysis.correlation || 0) > 0.7 ? 'strong' : Math.abs(correlationAnalysis.correlation || 0) > 0.3 ? 'moderate' : 'weak'}`}>
                Correlation: {(correlationAnalysis.correlation * 100).toFixed(1)}%
                {correlationAnalysis.correlation > 0 ? ' (Positive)' : ' (Negative)'}
              </div>
              
              <div className="significance">
                {correlationAnalysis.is_significant ? 
                  <span className="significant">✓ Statistically Significant (p-value: {correlationAnalysis.p_value.toFixed(4)})</span> :
                  <span className="not-significant">✗ Not Statistically Significant</span>
                }
              </div>
              
              <div className="correlation-interpretation">
                {Math.abs(correlationAnalysis.correlation || 0) > 0.7 ? (
                  <div className="interpretation">
                    <strong>Strong relationship:</strong> Parameter value changes strongly {correlationAnalysis.correlation > 0 ? 'increase' : 'decrease'} performance metrics
                  </div>
                ) : Math.abs(correlationAnalysis.correlation || 0) > 0.3 ? (
                  <div className="interpretation">
                    <strong>Moderate relationship:</strong> Parameter value changes moderately {correlationAnalysis.correlation > 0 ? 'increase' : 'decrease'} performance metrics
                  </div>
                ) : (
                  <div className="interpretation">
                    <strong>Weak relationship:</strong> Parameter value changes have limited impact on performance metrics
                  </div>
                )}
              </div>
            </div>
          ) : correlationAnalysis.status === 'non_numeric_parameters' ? (
            <div className="no-correlation">
              Cannot calculate correlation for non-numeric parameter values.
            </div>
          ) : (
            <div className="no-correlation">
              Insufficient data to calculate correlation.
            </div>
          )}
        </div>
      </div>
      
      <style jsx>{`
        .historical-trend-analysis {
          padding: 10px;
        }
        
        .summary-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }
        
        .period-info {
          background-color: #f0f0f0;
          padding: 5px 10px;
          border-radius: 4px;
          font-size: 14px;
        }
        
        .trend-table-container {
          margin-bottom: 20px;
          overflow-x: auto;
        }
        
        .trend-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }
        
        .trend-table th, 
        .trend-table td {
          padding: 8px 12px;
          border: 1px solid #ddd;
          text-align: left;
        }
        
        .trend-table th {
          background-color: #f5f5f5;
          font-weight: 600;
        }
        
        .trend-indicator {
          font-weight: 600;
        }
        
        .trend-indicator.improving {
          color: #4caf50;
        }
        
        .trend-indicator.degrading {
          color: #f44336;
        }
        
        .trend-indicator.stable,
        .trend-indicator.insufficient_data {
          color: #ff9800;
        }
        
        .chart-container {
          height: 300px;
          margin-bottom: 20px;
        }
        
        .chart-container h4 {
          margin-bottom: 10px;
          text-align: center;
        }
        
        .correlation-analysis {
          border-top: 1px solid #eee;
          padding-top: 15px;
        }
        
        .correlation-results {
          background-color: #f9f9f9;
          border-radius: 4px;
          padding: 15px;
          margin-top: 10px;
        }
        
        .correlation-value {
          font-size: 18px;
          font-weight: bold;
          margin-bottom: 8px;
        }
        
        .correlation-value.strong {
          color: #4caf50;
        }
        
        .correlation-value.moderate {
          color: #ff9800;
        }
        
        .correlation-value.weak {
          color: #f44336;
        }
        
        .significance {
          margin-bottom: 8px;
        }
        
        .significant {
          color: #4caf50;
          font-weight: 500;
        }
        
        .not-significant {
          color: #f44336;
          font-weight: 500;
        }
        
        .correlation-interpretation {
          margin-top: 10px;
          font-size: 14px;
        }
        
        .no-correlation {
          background-color: #f9f9f9;
          border-radius: 4px;
          padding: 15px;
          margin-top: 10px;
          color: #666;
          font-style: italic;
          text-align: center;
        }
        
        .no-data {
          display: flex;
          justify-content: center;
          align-items: center;
          height: 150px;
          color: #666;
          font-style: italic;
        }
      `}</style>
    </Card>
  );
};

export default HistoricalTrendAnalysis;
