import React, { useState, useEffect } from 'react';
import { Card, DataTable, StatusIndicator, Chart } from '../ui-library';
import ParameterEffectivenessCard from './ParameterEffectivenessCard';
import ParameterCrossRegimeAnalysis from './ParameterCrossRegimeAnalysis';
import HistoricalTrendAnalysis from './HistoricalTrendAnalysis';
import ParameterInsightsPanel from './ParameterInsightsPanel';
import { fetchFeedbackLoopMetrics, fetchParameterAnalytics } from '../../services/feedbackLoopService';

interface FeedbackLoopMetrics {
  strategyId: string;
  totalFeedbackCount: number;
  processedFeedbackCount: number;
  adaptationCount: number;
  successRate: number;
  parameterMetrics: Record<string, {
    name: string;
    effectivenessScore: number;
    optimalValue: number | string;
    significanceLevel: number;
    confidenceScore: number;
  }>;
}

interface FeedbackLoopDashboardProps {
  strategyId: string;
}

const FeedbackLoopDashboard: React.FC<FeedbackLoopDashboardProps> = ({ strategyId }) => {
  const [metrics, setMetrics] = useState<FeedbackLoopMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedParameter, setSelectedParameter] = useState<string | null>(null);
  const [parameterAnalytics, setParameterAnalytics] = useState<any | null>(null);

  useEffect(() => {
    const loadMetrics = async () => {
      setLoading(true);
      try {
        const data = await fetchFeedbackLoopMetrics(strategyId);
        setMetrics(data);
        
        // Select first parameter by default if available
        if (data && data.parameterMetrics) {
          const firstParam = Object.keys(data.parameterMetrics)[0];
          if (firstParam) {
            setSelectedParameter(firstParam);
          }
        }
      } catch (err) {
        setError('Failed to load feedback loop metrics');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadMetrics();
  }, [strategyId]);

  useEffect(() => {
    if (selectedParameter) {
      const loadParameterAnalytics = async () => {
        try {
          const data = await fetchParameterAnalytics(strategyId, selectedParameter);
          setParameterAnalytics(data);
        } catch (err) {
          console.error('Failed to load parameter analytics', err);
        }
      };
      
      loadParameterAnalytics();
    }
  }, [strategyId, selectedParameter]);

  // Handle parameter selection
  const handleParameterSelect = (paramName: string) => {
    setSelectedParameter(paramName);
  };

  if (loading) return <div>Loading feedback loop metrics...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!metrics) return <div>No feedback loop metrics available</div>;

  // Extract parameter data for display in the table
  const parameterTableData = Object.entries(metrics.parameterMetrics).map(([key, param]) => ({
    id: key,
    name: param.name,
    effectivenessScore: `${(param.effectivenessScore * 100).toFixed(1)}%`,
    optimalValue: param.optimalValue,
    significanceLevel: `${(param.significanceLevel * 100).toFixed(1)}%`,
    confidenceScore: `${(param.confidenceScore * 100).toFixed(1)}%`,
  }));

  // Calculate summary metrics
  const adaptationSuccessRate = metrics.processedFeedbackCount > 0 
    ? ((metrics.successRate) * 100).toFixed(1)
    : '0.0';
  
  return (
    <div className="feedback-loop-dashboard">
      <h1>Feedback Loop Dashboard - Strategy {strategyId}</h1>
      
      {/* Summary Metrics */}
      <div className="metrics-summary">
        <Card title="Feedback Loop Summary">
          <div className="metrics-grid">
            <div className="metric-item">
              <h3>Total Feedback</h3>
              <div className="metric-value">{metrics.totalFeedbackCount}</div>
            </div>
            <div className="metric-item">
              <h3>Processed Feedback</h3>
              <div className="metric-value">{metrics.processedFeedbackCount}</div>
            </div>
            <div className="metric-item">
              <h3>Adaptations</h3>
              <div className="metric-value">{metrics.adaptationCount}</div>
            </div>
            <div className="metric-item">
              <h3>Success Rate</h3>
              <div className="metric-value">
                <StatusIndicator 
                  status={parseFloat(adaptationSuccessRate) > 70 ? 'success' : parseFloat(adaptationSuccessRate) > 50 ? 'warning' : 'error'} 
                  label={`${adaptationSuccessRate}%`} 
                />
              </div>
            </div>
          </div>
        </Card>
      </div>
      
      {/* Parameter Effectiveness Table */}
      <Card title="Parameter Effectiveness">
        <DataTable
          data={parameterTableData}
          columns={[
            { id: 'name', header: 'Parameter', cell: (row) => row.name },
            { id: 'effectivenessScore', header: 'Effectiveness', cell: (row) => row.effectivenessScore },
            { id: 'optimalValue', header: 'Optimal Value', cell: (row) => row.optimalValue.toString() },
            { id: 'confidenceScore', header: 'Confidence', cell: (row) => <StatusIndicator 
              status={parseFloat(row.confidenceScore) > 80 ? 'success' : parseFloat(row.confidenceScore) > 60 ? 'warning' : 'error'} 
              label={row.confidenceScore} 
            /> },
            { id: 'actions', header: '', cell: (row) => (
              <button 
                onClick={() => handleParameterSelect(row.id)}
                className={selectedParameter === row.id ? 'selected' : ''}
              >
                Analyze
              </button>
            )}
          ]}
          onRowClick={(row) => handleParameterSelect(row.id)}
        />
      </Card>
      
      {/* Parameter Analysis Section */}
      {selectedParameter && metrics.parameterMetrics[selectedParameter] && (
        <div className="parameter-analysis">
          <h2>{metrics.parameterMetrics[selectedParameter].name} Analysis</h2>
          
          <div className="analysis-grid">
            {/* Parameter effectiveness detailed card */}
            <ParameterEffectivenessCard
              parameterName={metrics.parameterMetrics[selectedParameter].name}
              effectivenessScore={metrics.parameterMetrics[selectedParameter].effectivenessScore}
              optimalValue={metrics.parameterMetrics[selectedParameter].optimalValue}
              confidenceScore={metrics.parameterMetrics[selectedParameter].confidenceScore}
              analytics={parameterAnalytics?.sensitivity || null}
            />
            
            {/* Parameter cross-regime analysis */}
            <ParameterCrossRegimeAnalysis
              parameterName={metrics.parameterMetrics[selectedParameter].name}
              analytics={parameterAnalytics?.crossRegime || null}
            />
          </div>
          
          {/* Historical trend analysis */}
          <HistoricalTrendAnalysis
            parameterName={metrics.parameterMetrics[selectedParameter].name}
            analytics={parameterAnalytics?.historicalTrends || null}
          />
          
          {/* Parameter insights panel */}
          <ParameterInsightsPanel
            parameterName={metrics.parameterMetrics[selectedParameter].name}
            insights={parameterAnalytics?.insights || null}
            recommendations={parameterAnalytics?.recommendations || null}
          />
        </div>
      )}
      
      <style jsx>{`
        .feedback-loop-dashboard {
          padding: 20px;
        }
        
        .metrics-summary {
          margin-bottom: 24px;
        }
        
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
        }
        
        .metric-item {
          text-align: center;
        }
        
        .metric-value {
          font-size: 24px;
          font-weight: bold;
          margin-top: 8px;
        }
        
        .parameter-analysis {
          margin-top: 24px;
        }
        
        .analysis-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 20px;
          margin-bottom: 20px;
        }
        
        button {
          padding: 6px 12px;
          background-color: #f0f0f0;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
        }
        
        button.selected {
          background-color: #e0e0ff;
          border-color: #aaaaff;
        }
      `}</style>
    </div>
  );
};

export default FeedbackLoopDashboard;
