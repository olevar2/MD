// Multi-asset detail component for displaying asset-specific information
import { useState, useEffect } from 'react';
import { Card, Tabs, Tab, Badge, Spinner } from 'react-bootstrap';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { formatCurrency, formatNumber, formatPercentage } from '../utils/formatters';
import { getAssetColor } from '../utils/assetHelpers';
import { useAssetInfo } from '../hooks/useAssetInfo';

// Asset-specific display components
import ForexDetails from './asset-detail/ForexDetails';
import CryptoDetails from './asset-detail/CryptoDetails';
import StockDetails from './asset-detail/StockDetails';
import CommodityDetails from './asset-detail/CommodityDetails';
import IndexDetails from './asset-detail/IndexDetails';
import DefaultDetails from './asset-detail/DefaultDetails';

const AssetDetailView = ({ symbol, timeframe }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const { assetInfo, isLoading, error } = useAssetInfo(symbol);
  
  if (isLoading) {
    return <div className="text-center p-4"><Spinner animation="border" /></div>;
  }
  
  if (error || !assetInfo) {
    return <div className="alert alert-danger">Error loading asset details</div>;
  }
  
  // Determine which detail component to use based on asset class
  const renderAssetSpecificDetails = () => {
    switch (assetInfo.asset_class) {
      case 'forex':
        return <ForexDetails symbol={symbol} assetInfo={assetInfo} />;
      case 'crypto':
        return <CryptoDetails symbol={symbol} assetInfo={assetInfo} />;
      case 'stocks':
        return <StockDetails symbol={symbol} assetInfo={assetInfo} />;
      case 'commodities':
        return <CommodityDetails symbol={symbol} assetInfo={assetInfo} />;
      case 'indices':
        return <IndexDetails symbol={symbol} assetInfo={assetInfo} />;
      default:
        return <DefaultDetails symbol={symbol} assetInfo={assetInfo} />;
    }
  };

  return (
    <div className="asset-detail-container">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h2>
          {assetInfo.display_name || symbol} 
          <Badge 
            bg={getAssetColor(assetInfo.asset_class)} 
            className="ms-2"
          >
            {assetInfo.asset_class}
          </Badge>
        </h2>
      </div>

      <Tabs
        activeKey={activeTab}
        onSelect={(k) => setActiveTab(k)}
        className="mb-3"
      >
        <Tab eventKey="overview" title="Overview">
          <Card>
            <Card.Body>
              {renderAssetSpecificDetails()}
            </Card.Body>
          </Card>
        </Tab>
        
        <Tab eventKey="analysis" title="Analysis">
          <Card>
            <Card.Body>
              <AssetAnalysisTab symbol={symbol} assetInfo={assetInfo} timeframe={timeframe} />
            </Card.Body>
          </Card>
        </Tab>
        
        <Tab eventKey="correlations" title="Correlations">
          <Card>
            <Card.Body>
              <AssetCorrelationTab symbol={symbol} assetInfo={assetInfo} />
            </Card.Body>
          </Card>
        </Tab>
        
        <Tab eventKey="performance" title="Performance">
          <Card>
            <Card.Body>
              <AssetPerformanceTab symbol={symbol} assetInfo={assetInfo} />
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>
    </div>
  );
};

const AssetAnalysisTab = ({ symbol, assetInfo, timeframe }) => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch analysis data for the asset
    const fetchAnalysisData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/v1/multi-asset/analysis/${symbol}?timeframe=${timeframe}`);
        if (!response.ok) {
          throw new Error('Failed to fetch analysis data');
        }
        const data = await response.json();
        setAnalysisData(data);
      } catch (error) {
        console.error('Error fetching analysis data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchAnalysisData();
  }, [symbol, timeframe]);
  
  if (loading) {
    return <div className="text-center"><Spinner animation="border" /></div>;
  }
  
  if (!analysisData) {
    return <div className="alert alert-warning">No analysis data available</div>;
  }

  // Render analysis content based on asset class
  return (
    <div>
      <h4>Technical Analysis</h4>
      
      {/* Asset-specific analysis parameters */}
      <div className="analysis-parameters">
        <h5>Key Parameters</h5>
        <div className="row">
          <div className="col-md-6">
            {assetInfo.asset_class === 'forex' && (
              <div className="mb-3">
                <div className="small text-muted">Pip Value</div>
                <div className="fw-bold">{formatNumber(assetInfo.trading_parameters?.pip_value || 0.0001, 6)}</div>
              </div>
            )}
            
            {assetInfo.asset_class === 'crypto' && (
              <div className="mb-3">
                <div className="small text-muted">24h Volume</div>
                <div className="fw-bold">{formatCurrency(analysisData.market_data?.volume_24h || 0)}</div>
              </div>
            )}
            
            {assetInfo.asset_class === 'stocks' && (
              <div className="mb-3">
                <div className="small text-muted">Market Cap</div>
                <div className="fw-bold">{formatCurrency(analysisData.market_data?.market_cap || 0)}</div>
              </div>
            )}
          </div>
          <div className="col-md-6">
            <div className="mb-3">
              <div className="small text-muted">Volatility</div>
              <div className="fw-bold">{formatPercentage(analysisData.volatility?.current || 0)}</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Signals display */}
      <div className="signals-display mt-4">
        <h5>Current Signals</h5>
        <div className="row">
          {analysisData.signals?.map((signal, index) => (
            <div key={index} className="col-md-4 mb-3">
              <Card className="h-100">
                <Card.Body>
                  <Card.Title>{signal.name}</Card.Title>
                  <div className={`signal-value ${signal.direction}`}>
                    {signal.direction === 'buy' ? '⬆️ Buy' : signal.direction === 'sell' ? '⬇️ Sell' : '➖ Neutral'}
                  </div>
                  <div className="signal-strength small text-muted">
                    Strength: {formatPercentage(signal.strength)}
                  </div>
                </Card.Body>
              </Card>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const AssetCorrelationTab = ({ symbol, assetInfo }) => {
  const [correlationData, setCorrelationData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch correlation data
    const fetchCorrelationData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/v1/correlations/symbol/${symbol}?threshold=0.5&limit=10`);
        if (!response.ok) {
          throw new Error('Failed to fetch correlation data');
        }
        const data = await response.json();
        setCorrelationData(data);
      } catch (error) {
        console.error('Error fetching correlation data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchCorrelationData();
  }, [symbol]);
  
  if (loading) {
    return <div className="text-center"><Spinner animation="border" /></div>;
  }
  
  if (!correlationData || !correlationData.correlations || correlationData.correlations.length === 0) {
    return <div className="alert alert-warning">No correlation data available</div>;
  }

  // Render correlation data
  return (
    <div>
      <h4>Correlations with {assetInfo.display_name || symbol}</h4>
      
      <div className="correlation-list mt-4">
        <h5>Highest Correlations</h5>
        <table className="table table-striped">
          <thead>
            <tr>
              <th>Asset</th>
              <th>Asset Class</th>
              <th>Correlation</th>
              <th>Relationship</th>
            </tr>
          </thead>
          <tbody>
            {correlationData.correlations.map((corr, index) => (
              <tr key={index}>
                <td>{corr.display_name || corr.symbol}</td>
                <td>
                  <Badge bg={getAssetColor(corr.asset_class)}>
                    {corr.asset_class}
                  </Badge>
                </td>
                <td>{formatNumber(corr.correlation, 2)}</td>
                <td>
                  {corr.correlation > 0.7 ? (
                    <span className="text-success">Strong positive</span>
                  ) : corr.correlation > 0.3 ? (
                    <span className="text-success">Positive</span>
                  ) : corr.correlation > -0.3 ? (
                    <span className="text-secondary">Weak</span>
                  ) : corr.correlation > -0.7 ? (
                    <span className="text-danger">Negative</span>
                  ) : (
                    <span className="text-danger">Strong negative</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="cross-asset-correlations mt-4">
        <h5>Cross-Asset Correlations</h5>
        {/* Chart would go here */}
      </div>
    </div>
  );
};

const AssetPerformanceTab = ({ symbol, assetInfo }) => {
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch performance data
    const fetchPerformanceData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/v1/multi-asset/performance/${symbol}`);
        if (!response.ok) {
          throw new Error('Failed to fetch performance data');
        }
        const data = await response.json();
        setPerformanceData(data);
      } catch (error) {
        console.error('Error fetching performance data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchPerformanceData();
  }, [symbol]);

  if (loading) {
    return <div className="text-center"><Spinner animation="border" /></div>;
  }
  
  if (!performanceData) {
    return <div className="alert alert-warning">No performance data available</div>;
  }

  // Render performance data
  return (
    <div>
      <h4>Performance</h4>
      
      <div className="row mb-4">
        <div className="col-md-4">
          <Card>
            <Card.Body>
              <Card.Title>Daily Return</Card.Title>
              <div className={performanceData.daily_return >= 0 ? 'text-success' : 'text-danger'}>
                {formatPercentage(performanceData.daily_return || 0)}
              </div>
            </Card.Body>
          </Card>
        </div>
        <div className="col-md-4">
          <Card>
            <Card.Body>
              <Card.Title>Weekly Return</Card.Title>
              <div className={performanceData.weekly_return >= 0 ? 'text-success' : 'text-danger'}>
                {formatPercentage(performanceData.weekly_return || 0)}
              </div>
            </Card.Body>
          </Card>
        </div>
        <div className="col-md-4">
          <Card>
            <Card.Body>
              <Card.Title>Monthly Return</Card.Title>
              <div className={performanceData.monthly_return >= 0 ? 'text-success' : 'text-danger'}>
                {formatPercentage(performanceData.monthly_return || 0)}
              </div>
            </Card.Body>
          </Card>
        </div>
      </div>

      <div className="performance-chart">
        <h5>Price History</h5>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performanceData.price_history || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke={getAssetColor(assetInfo.asset_class) || '#8884d8'} 
              activeDot={{ r: 8 }} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AssetDetailView;
