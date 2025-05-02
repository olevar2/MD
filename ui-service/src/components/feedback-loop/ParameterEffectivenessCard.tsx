import React from 'react';
import { Card, Chart } from '../ui-library';

interface ParameterEffectivenessCardProps {
  parameterName: string;
  effectivenessScore: number;
  optimalValue: number | string;
  confidenceScore: number;
  analytics: any | null;
}

const ParameterEffectivenessCard: React.FC<ParameterEffectivenessCardProps> = ({
  parameterName,
  effectivenessScore,
  optimalValue,
  confidenceScore,
  analytics
}) => {
  // Format metrics for display
  const effectivenessPercentage = (effectivenessScore * 100).toFixed(1);
  const confidencePercentage = (confidenceScore * 100).toFixed(1);

  // Effectiveness level classification
  let effectivenessLevel = 'low';
  if (effectivenessScore >= 0.7) {
    effectivenessLevel = 'high';
  } else if (effectivenessScore >= 0.4) {
    effectivenessLevel = 'medium';
  }

  // Determine color based on effectiveness
  const getEffectivenessColor = () => {
    if (effectivenessScore >= 0.7) return '#4caf50';  // Green for high
    if (effectivenessScore >= 0.4) return '#ff9800';  // Orange for medium
    return '#f44336';  // Red for low
  };

  // Sample data for the effectiveness gauge chart
  const gaugeChartData = {
    labels: ['Effectiveness'],
    datasets: [
      {
        data: [effectivenessScore],
        backgroundColor: getEffectivenessColor(),
        circumference: 180,
        rotation: 270,
        cutout: '70%'
      }
    ]
  };

  // Sample data for sensitivity visualization if analytics is available
  const sensitivityData = analytics?.sensitivityByValue ? {
    labels: Object.keys(analytics.sensitivityByValue).map(key => `Value: ${key}`),
    datasets: [
      {
        label: 'Sensitivity',
        data: Object.values(analytics.sensitivityByValue),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }
    ]
  } : null;

  return (
    <Card title={`${parameterName} Effectiveness`}>
      <div className="parameter-effectiveness">
        <div className="effectiveness-header">
          <div className="gauge-chart">
            {/* This would be replaced with an actual Chart component */}
            <Chart 
              type="doughnut"
              data={gaugeChartData}
              options={{
                responsive: true,
                plugins: {
                  tooltip: {
                    enabled: false
                  },
                  legend: {
                    display: false
                  }
                }
              }}
            />
            <div className="gauge-value">{effectivenessPercentage}%</div>
          </div>
          
          <div className="effectiveness-summary">
            <div className="summary-item">
              <div className="label">Effectiveness Level:</div>
              <div className={`value ${effectivenessLevel}`}>
                {effectivenessLevel.charAt(0).toUpperCase() + effectivenessLevel.slice(1)}
              </div>
            </div>
            
            <div className="summary-item">
              <div className="label">Optimal Value:</div>
              <div className="value">{optimalValue.toString()}</div>
            </div>
            
            <div className="summary-item">
              <div className="label">Confidence Score:</div>
              <div className={`value ${confidenceScore >= 0.7 ? 'high' : confidenceScore >= 0.4 ? 'medium' : 'low'}`}>
                {confidencePercentage}%
              </div>
            </div>
          </div>
        </div>
        
        {/* Additional analytics if available */}
        {analytics && (
          <div className="effectiveness-analytics">
            <h4>Sensitivity Analysis</h4>
            
            <div className="analytics-metrics">
              <div className="metric">
                <div className="metric-label">Sensitivity Score</div>
                <div className="metric-value">{(analytics.sensitivityScore * 100).toFixed(1)}%</div>
              </div>
              
              {analytics.stabilityByValue && (
                <div className="metric">
                  <div className="metric-label">Stability</div>
                  <div className="metric-value">
                    {Object.entries(analytics.stabilityByValue).map(([value, stability]) => (
                      <div key={value} className="stability-item">
                        Value {value}: <span className={stability >= 0.7 ? 'high' : stability >= 0.4 ? 'medium' : 'low'}>
                          {(Number(stability) * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            {sensitivityData && (
              <div className="sensitivity-chart">
                <Chart
                  type="bar"
                  data={sensitivityData}
                  options={{
                    responsive: true,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                          display: true,
                          text: 'Sensitivity'
                        }
                      }
                    }
                  }}
                />
              </div>
            )}
          </div>
        )}
      </div>
      
      <style jsx>{`
        .parameter-effectiveness {
          padding: 10px;
        }
        
        .effectiveness-header {
          display: flex;
          align-items: center;
          margin-bottom: 20px;
        }
        
        .gauge-chart {
          position: relative;
          width: 140px;
          height: 100px;
          margin-right: 20px;
        }
        
        .gauge-value {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          text-align: center;
          font-size: 20px;
          font-weight: bold;
        }
        
        .effectiveness-summary {
          flex: 1;
        }
        
        .summary-item {
          display: flex;
          margin-bottom: 10px;
          align-items: center;
        }
        
        .label {
          width: 150px;
          font-weight: 500;
        }
        
        .value {
          font-weight: bold;
        }
        
        .value.high {
          color: #4caf50;
        }
        
        .value.medium {
          color: #ff9800;
        }
        
        .value.low {
          color: #f44336;
        }
        
        .effectiveness-analytics {
          border-top: 1px solid #eee;
          padding-top: 15px;
        }
        
        .analytics-metrics {
          display: flex;
          gap: 20px;
          margin-bottom: 15px;
        }
        
        .metric {
          flex: 1;
        }
        
        .metric-label {
          font-weight: 500;
          margin-bottom: 5px;
        }
        
        .stability-item {
          margin-bottom: 5px;
        }
        
        .sensitivity-chart {
          height: 200px;
        }
      `}</style>
    </Card>
  );
};

export default ParameterEffectivenessCard;
