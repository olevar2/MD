import React from 'react';
import { Card, Chart } from '../ui-library';

interface ParameterCrossRegimeAnalysisProps {
  parameterName: string;
  analytics: any | null;
}

const ParameterCrossRegimeAnalysis: React.FC<ParameterCrossRegimeAnalysisProps> = ({
  parameterName,
  analytics
}) => {
  if (!analytics) {
    return (
      <Card title={`${parameterName} Regime Analysis`}>
        <div className="no-data">
          <p>No cross-regime analysis data available for this parameter.</p>
        </div>
      </Card>
    );
  }

  // Extract data for visualization
  const regimes = analytics.regimes_analyzed || [];
  const optimalByRegime = analytics.optimal_by_regime || {};
  const sensitivityByRegime = analytics.sensitivity_by_regime || {};

  // Prepare data for regime comparison chart
  const regimeComparisonData = {
    labels: regimes,
    datasets: [
      {
        label: 'Win Rate',
        data: regimes.map(regime => {
          if (optimalByRegime[regime]) {
            return optimalByRegime[regime].win_rate;
          }
          return 0;
        }),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      },
      {
        label: 'Profit Factor',
        data: regimes.map(regime => {
          const profitFactor = optimalByRegime[regime]?.profit_factor;
          // Cap extremely high profit factors for visualization
          return profitFactor && profitFactor !== Infinity ? Math.min(profitFactor, 5) : 0;
        }),
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      },
      {
        label: 'Parameter Sensitivity',
        data: regimes.map(regime => sensitivityByRegime[regime] || 0),
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1
      }
    ]
  };

  // Prepare data for optimal value comparison
  const optimalValueData = {
    labels: regimes,
    datasets: [
      {
        label: 'Optimal Value',
        data: regimes.map(regime => {
          if (optimalByRegime[regime]) {
            const value = optimalByRegime[regime].value;
            // Convert to number if possible for visualization
            return typeof value === 'string' ? parseFloat(value) || 0 : value;
          }
          return 0;
        }),
        backgroundColor: 'rgba(153, 102, 255, 0.5)',
        borderColor: 'rgba(153, 102, 255, 1)',
        borderWidth: 1
      }
    ]
  };

  // Check if we have ANOVA results
  const hasAnovaResults = analytics.anova_results && Object.keys(analytics.anova_results).length > 0;

  return (
    <Card title={`${parameterName} Regime Analysis`}>
      <div className="cross-regime-analysis">
        <div className="regime-metrics-summary">
          <div className="summary-header">
            <h4>Cross-Regime Performance</h4>
            <div className="regimes-count">
              {regimes.length} Regimes Analyzed
            </div>
          </div>
          
          <div className="regime-comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Regime</th>
                  <th>Optimal Value</th>
                  <th>Win Rate</th>
                  <th>Profit Factor</th>
                  <th>Sensitivity</th>
                </tr>
              </thead>
              <tbody>
                {regimes.map(regime => (
                  <tr key={regime}>
                    <td>{regime}</td>
                    <td>
                      {optimalByRegime[regime] ? optimalByRegime[regime].value.toString() : 'N/A'}
                    </td>
                    <td>
                      {optimalByRegime[regime] 
                        ? `${(optimalByRegime[regime].win_rate * 100).toFixed(1)}%` 
                        : 'N/A'}
                    </td>
                    <td>
                      {optimalByRegime[regime]
                        ? optimalByRegime[regime].profit_factor === Infinity 
                          ? '∞' 
                          : optimalByRegime[regime].profit_factor.toFixed(2)
                        : 'N/A'}
                    </td>
                    <td>
                      {sensitivityByRegime[regime] 
                        ? `${(sensitivityByRegime[regime] * 100).toFixed(1)}%` 
                        : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="regime-charts">
          {/* Performance metrics by regime chart */}
          <div className="chart-container">
            <h4>Performance Metrics by Regime</h4>
            <Chart 
              type="bar"
              data={regimeComparisonData}
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
                      text: 'Market Regime'
                    }
                  }
                }
              }}
            />
          </div>
          
          {/* Optimal value by regime chart */}
          <div className="chart-container">
            <h4>Optimal Parameter Value by Regime</h4>
            <Chart 
              type="bar"
              data={optimalValueData}
              options={{
                responsive: true,
                scales: {
                  y: {
                    beginAtZero: false,
                    title: {
                      display: true,
                      text: 'Optimal Value'
                    }
                  },
                  x: {
                    title: {
                      display: true,
                      text: 'Market Regime'
                    }
                  }
                }
              }}
            />
          </div>
        </div>
        
        {/* Statistical significance */}
        {hasAnovaResults && (
          <div className="statistical-significance">
            <h4>Statistical Significance Analysis</h4>
            <div className="significant-values">
              <h5>Parameter Values with Significant Regime Differences:</h5>
              <ul>
                {Object.entries(analytics.anova_results).map(([value, result]: [string, any]) => (
                  <li key={value} className={result.is_significant ? 'significant' : 'not-significant'}>
                    Value {value}: 
                    <span className="significance-marker">
                      {result.is_significant ? '✓ Significant' : '✗ Not Significant'}
                    </span>
                    {result.is_significant && (
                      <span className="p-value">(p-value: {result.p_value.toFixed(4)})</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
      
      <style jsx>{`
        .cross-regime-analysis {
          padding: 10px;
        }
        
        .summary-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }
        
        .regimes-count {
          background-color: #f0f0f0;
          padding: 5px 10px;
          border-radius: 4px;
          font-size: 14px;
        }
        
        .regime-comparison-table {
          margin-bottom: 20px;
          overflow-x: auto;
        }
        
        table {
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }
        
        th, td {
          padding: 8px 12px;
          border: 1px solid #ddd;
          text-align: left;
        }
        
        th {
          background-color: #f5f5f5;
          font-weight: 600;
        }
        
        .regime-charts {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          margin-bottom: 20px;
        }
        
        .chart-container {
          height: 250px;
        }
        
        .chart-container h4 {
          margin-bottom: 10px;
          text-align: center;
        }
        
        .statistical-significance {
          border-top: 1px solid #eee;
          padding-top: 15px;
        }
        
        .significant-values {
          margin-top: 10px;
        }
        
        .significant-values ul {
          padding-left: 20px;
        }
        
        .significant-values li {
          margin-bottom: 5px;
        }
        
        .significance-marker {
          margin-left: 8px;
          font-weight: 600;
        }
        
        .significant .significance-marker {
          color: #4caf50;
        }
        
        .not-significant .significance-marker {
          color: #f44336;
        }
        
        .p-value {
          margin-left: 8px;
          font-size: 13px;
          color: #666;
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

export default ParameterCrossRegimeAnalysis;
