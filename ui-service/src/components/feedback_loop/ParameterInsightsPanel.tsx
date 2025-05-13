import React, { useState } from 'react';
import { Card } from '../ui-library';

interface ParameterInsightsPanelProps {
  parameterName: string;
  insights: any[] | null;
  recommendations: any[] | null;
}

const ParameterInsightsPanel: React.FC<ParameterInsightsPanelProps> = ({
  parameterName,
  insights,
  recommendations
}) => {
  const [activeTab, setActiveTab] = useState<'insights' | 'recommendations'>('insights');
  
  if (!insights && !recommendations) {
    return (
      <Card title={`${parameterName} Insights & Recommendations`}>
        <div className="no-data">
          <p>No insights or recommendations available for this parameter.</p>
        </div>
      </Card>
    );
  }
  
  // Group insights by type for better organization
  const insightsByType = insights ? insights.reduce((groups, insight) => {
    const type = insight.type || 'general';
    if (!groups[type]) {
      groups[type] = [];
    }
    groups[type].push(insight);
    return groups;
  }, {}) : {};
  
  // Helper to get icon for insight type
  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'high_sensitivity':
        return '🔍';
      case 'low_sensitivity':
        return '🔧';
      case 'unstable_values':
        return '⚠️';
      case 'regime_specific_optima':
        return '🌐';
      case 'regime_sensitive_values':
        return '📊';
      case 'improving_values':
        return '📈';
      case 'degrading_values':
        return '📉';
      case 'significant_correlation':
        return '🔗';
      default:
        return '💡';
    }
  };
  
  // Helper to get icon for recommendation type
  const getRecommendationIcon = (type: string) => {
    switch (type) {
      case 'fine_tuning':
        return '🎯';
      case 'regime_adaptation':
        return '🔄';
      case 'increase_allocation':
        return '⬆️';
      case 'decrease_allocation':
        return '⬇️';
      default:
        return '✅';
    }
  };

  return (
    <Card title={`${parameterName} Insights & Recommendations`}>
      <div className="parameter-insights-panel">
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'insights' ? 'active' : ''}`}
            onClick={() => setActiveTab('insights')}
          >
            Insights
            {insights && insights.length > 0 && (
              <span className="badge">{insights.length}</span>
            )}
          </button>
          <button 
            className={`tab ${activeTab === 'recommendations' ? 'active' : ''}`}
            onClick={() => setActiveTab('recommendations')}
          >
            Recommendations
            {recommendations && recommendations.length > 0 && (
              <span className="badge">{recommendations.length}</span>
            )}
          </button>
        </div>
        
        <div className="tab-content">
          {activeTab === 'insights' && (
            <div className="insights">
              {insights && insights.length > 0 ? (
                <>
                  {Object.entries(insightsByType).map(([type, typeInsights]) => (
                    <div key={type} className="insight-group">
                      <h4 className="insight-group-title">
                        {type.split('_').map(word => 
                          word.charAt(0).toUpperCase() + word.slice(1)
                        ).join(' ')} Insights
                      </h4>
                      
                      <div className="insight-list">
                        {typeInsights.map((insight, index) => (
                          <div key={index} className="insight-item">
                            <div className="insight-icon">
                              {getInsightIcon(insight.type)}
                            </div>
                            <div className="insight-content">
                              <div className="insight-message">
                                {insight.message}
                              </div>
                              
                              {/* Display additional data if available */}
                              {insight.type === 'regime_specific_optima' && insight.regime_optima && (
                                <div className="insight-details">
                                  <h5>Optimal Values by Regime:</h5>
                                  <ul className="regime-values">
                                    {Object.entries(insight.regime_optima).map(([regime, value]) => (
                                      <li key={regime}>
                                        <strong>{regime}:</strong> {value?.toString()}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              
                              {insight.type === 'significant_correlation' && (
                                <div className="insight-details correlation-details">
                                  <div className="correlation-value">
                                    Correlation: {insight.correlation.toFixed(2)}
                                  </div>
                                  <div className="correlation-significance">
                                    p-value: {insight.p_value.toFixed(4)}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </>
              ) : (
                <div className="no-items-message">
                  No insights available for this parameter.
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'recommendations' && (
            <div className="recommendations">
              {recommendations && recommendations.length > 0 ? (
                <div className="recommendation-list">
                  {recommendations.map((recommendation, index) => (
                    <div key={index} className="recommendation-item">
                      <div className="recommendation-icon">
                        {getRecommendationIcon(recommendation.type)}
                      </div>
                      <div className="recommendation-content">
                        <div className="recommendation-message">
                          {recommendation.message}
                        </div>
                        
                        {recommendation.details && (
                          <div className="recommendation-details">
                            {recommendation.details}
                          </div>
                        )}
                        
                        <div className="recommendation-action">
                          <button className="action-button">Apply</button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-items-message">
                  No recommendations available for this parameter.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      <style jsx>{`
        .parameter-insights-panel {
          padding: 10px;
        }
        
        .tabs {
          display: flex;
          border-bottom: 1px solid #ddd;
          margin-bottom: 15px;
        }
        
        .tab {
          padding: 10px 15px;
          background: none;
          border: none;
          border-bottom: 2px solid transparent;
          margin-right: 10px;
          cursor: pointer;
          font-weight: 500;
          font-size: 16px;
          position: relative;
        }
        
        .tab.active {
          border-bottom-color: #4285f4;
          color: #4285f4;
        }
        
        .badge {
          background-color: #4285f4;
          color: white;
          border-radius: 50%;
          min-width: 18px;
          height: 18px;
          padding: 0 4px;
          font-size: 12px;
          line-height: 18px;
          text-align: center;
          margin-left: 5px;
          display: inline-block;
        }
        
        .tab-content {
          padding: 5px;
        }
        
        .insight-group {
          margin-bottom: 20px;
        }
        
        .insight-group-title {
          margin-bottom: 10px;
          color: #555;
          font-size: 16px;
          border-bottom: 1px solid #eee;
          padding-bottom: 5px;
        }
        
        .insight-list, .recommendation-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        
        .insight-item, .recommendation-item {
          display: flex;
          background-color: #f9f9f9;
          border-radius: 4px;
          padding: 12px;
        }
        
        .insight-icon, .recommendation-icon {
          font-size: 24px;
          margin-right: 15px;
          min-width: 24px;
        }
        
        .insight-content, .recommendation-content {
          flex: 1;
        }
        
        .insight-message, .recommendation-message {
          margin-bottom: 8px;
          line-height: 1.4;
        }
        
        .insight-details, .recommendation-details {
          background-color: #f0f0f0;
          padding: 10px;
          border-radius: 4px;
          margin-top: 8px;
          font-size: 14px;
        }
        
        .insight-details h5 {
          margin-top: 0;
          margin-bottom: 5px;
        }
        
        .regime-values {
          margin: 0;
          padding-left: 20px;
        }
        
        .regime-values li {
          margin-bottom: 3px;
        }
        
        .correlation-details {
          display: flex;
          justify-content: space-between;
        }
        
        .correlation-value {
          font-weight: 500;
        }
        
        .recommendation-action {
          margin-top: 12px;
          text-align: right;
        }
        
        .action-button {
          background-color: #4285f4;
          color: white;
          border: none;
          padding: 6px 12px;
          border-radius: 4px;
          cursor: pointer;
          transition: background-color 0.3s;
        }
        
        .action-button:hover {
          background-color: #3367d6;
        }
        
        .no-items-message {
          text-align: center;
          padding: 20px;
          color: #666;
          font-style: italic;
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

export default ParameterInsightsPanel;
