import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { Card, DataTable, Chart } from '../../components/ui-library';
import SignalConfidenceChart from '../../components/visualization/SignalConfidenceChart';
import RegimeIndicator from '../../components/dashboard/RegimeIndicator';
import useResponsiveLayout from '../../hooks/useResponsiveLayout';

interface Position {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  currentPrice: number;
  size: number;
  pnl: number;
  pnlPercentage: number;
  openTime: string;
  stopLoss: number;
  takeProfit: number;
  regime: string;
  signals: {
    source: string;
    confidence: number;
    direction: string;
  }[];
}

const PositionsMonitorPage: React.FC = () => {
  const router = useRouter();
  const { isMobile, isTablet } = useResponsiveLayout();
  
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<number>(10000); // 10 seconds
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  
  // Fetch positions data
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        setIsLoading(true);
        // In production, replace with actual API call
        const response = await fetch('/api/positions/active');
        const data = await response.json();
        
        if (data.success) {
          setPositions(data.positions);
          // Select first position by default if available and none selected
          if (data.positions.length > 0 && !selectedPosition) {
            setSelectedPosition(data.positions[0]);
          }
        } else {
          setError(data.message || 'Failed to load positions');
        }
      } catch (err) {
        console.error('Error fetching positions:', err);
        setError('Error fetching positions data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchPositions();
    
    // Set up auto-refresh
    let intervalId: NodeJS.Timeout;
    if (autoRefresh) {
      intervalId = setInterval(fetchPositions, refreshInterval);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefresh, refreshInterval, selectedPosition]);
  
  // Handle position selection
  const handlePositionSelect = (position: Position) => {
    setSelectedPosition(position);
  };
  
  // Handle position close
  const handlePositionClose = async (positionId: string) => {
    try {
      // In production, replace with actual API call
      const response = await fetch(`/api/positions/${positionId}/close`, {
        method: 'POST',
      });
      const data = await response.json();
      
      if (data.success) {
        // Remove the closed position from the list
        setPositions(positions.filter(p => p.id !== positionId));
        if (selectedPosition?.id === positionId) {
          setSelectedPosition(null);
        }
      } else {
        alert(`Failed to close position: ${data.message}`);
      }
    } catch (err) {
      console.error('Error closing position:', err);
      alert('Error closing position. Please try again.');
    }
  };
  
  // Toggle auto-refresh
  const toggleAutoRefresh = () => {
    setAutoRefresh(prev => !prev);
  };
  
  // Update refresh interval
  const handleRefreshIntervalChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setRefreshInterval(parseInt(e.target.value, 10));
  };
  
  // Calculate total P&L
  const totalPnL = positions.reduce((sum, position) => sum + position.pnl, 0);
  const pnlClass = totalPnL >= 0 ? 'positive' : 'negative';
  
  // Prepare P&L chart data
  const pnlChartData = {
    labels: positions.map(p => p.symbol),
    datasets: [
      {
        label: 'P&L',
        data: positions.map(p => p.pnl),
        backgroundColor: positions.map(p => p.pnl >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'),
        borderColor: positions.map(p => p.pnl >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'),
        borderWidth: 1
      }
    ]
  };
  
  return (
    <>
      <Head>
        <title>Positions Monitor | Forex Trading Platform</title>
        <meta name="description" content="Real-time position monitoring with P&L visualization" />
      </Head>
      
      <div className={`positions-monitor-container ${isMobile ? 'mobile' : ''}`}>
        <div className="dashboard-header">
          <h1>Positions Monitor</h1>
          
          <div className="controls">
            <div className="refresh-controls">
              <label>
                <input 
                  type="checkbox" 
                  checked={autoRefresh} 
                  onChange={toggleAutoRefresh}
                />
                Auto-refresh
              </label>
              
              <select 
                value={refreshInterval} 
                onChange={handleRefreshIntervalChange}
                disabled={!autoRefresh}
              >
                <option value="5000">5s</option>
                <option value="10000">10s</option>
                <option value="30000">30s</option>
                <option value="60000">1m</option>
              </select>
            </div>
            
            <div className={`total-pnl ${pnlClass}`}>
              Total P&L: {totalPnL.toFixed(2)}
            </div>
          </div>
        </div>
        
        {isLoading && !positions.length ? (
          <div className="loading">Loading positions data...</div>
        ) : error ? (
          <div className="error-message">{error}</div>
        ) : (
          <div className={`dashboard-content ${isMobile ? 'stacked' : 'side-by-side'}`}>
            <div className="positions-list">
              <Card title="Active Positions">
                {positions.length === 0 ? (
                  <div className="no-positions">
                    No active positions found
                  </div>
                ) : (
                  <DataTable
                    data={positions}
                    columns={[
                      { 
                        id: 'symbol', 
                        header: 'Symbol', 
                        cell: (row) => row.symbol 
                      },
                      { 
                        id: 'direction', 
                        header: 'Dir', 
                        cell: (row) => (
                          <span className={row.direction === 'long' ? 'direction-long' : 'direction-short'}>
                            {row.direction === 'long' ? '↑' : '↓'}
                          </span>
                        ) 
                      },
                      { 
                        id: 'size', 
                        header: 'Size', 
                        cell: (row) => row.size 
                      },
                      { 
                        id: 'entryPrice', 
                        header: 'Entry', 
                        cell: (row) => row.entryPrice.toFixed(5) 
                      },
                      { 
                        id: 'currentPrice', 
                        header: 'Current', 
                        cell: (row) => row.currentPrice.toFixed(5) 
                      },
                      { 
                        id: 'pnl', 
                        header: 'P&L', 
                        cell: (row) => (
                          <span className={row.pnl >= 0 ? 'positive' : 'negative'}>
                            {row.pnl.toFixed(2)} ({row.pnlPercentage.toFixed(2)}%)
                          </span>
                        )
                      },
                      { 
                        id: 'regime', 
                        header: 'Regime', 
                        cell: (row) => <RegimeIndicator regime={row.regime} size="small" /> 
                      },
                      { 
                        id: 'actions', 
                        header: '', 
                        cell: (row) => (
                          <div className="position-actions">
                            <button 
                              className="view-btn"
                              onClick={() => handlePositionSelect(row)}
                            >
                              View
                            </button>
                            <button 
                              className="close-btn"
                              onClick={() => handlePositionClose(row.id)}
                            >
                              Close
                            </button>
                          </div>
                        ) 
                      }
                    ]}
                    onRowClick={(row) => handlePositionSelect(row)}
                    selectedRowId={selectedPosition?.id}
                  />
                )}
              </Card>
              
              {!isMobile && (
                <Card title="P&L Distribution">
                  <div className="pnl-chart">
                    <Chart
                      type="bar"
                      data={pnlChartData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            beginAtZero: true,
                            title: {
                              display: true,
                              text: 'Profit/Loss'
                            }
                          }
                        },
                        plugins: {
                          legend: {
                            display: false
                          }
                        }
                      }}
                    />
                  </div>
                </Card>
              )}
            </div>
            
            {selectedPosition && (
              <div className="position-details">
                <Card title={`${selectedPosition.symbol} Position Details`}>
                  <div className="detail-header">
                    <div className="detail-title">
                      <h3>{selectedPosition.symbol}</h3>
                      <span className={`direction ${selectedPosition.direction}`}>
                        {selectedPosition.direction === 'long' ? 'LONG' : 'SHORT'}
                      </span>
                    </div>
                    
                    <RegimeIndicator regime={selectedPosition.regime} />
                  </div>
                  
                  <div className="detail-grid">
                    <div className="detail-item">
                      <div className="label">Entry Price</div>
                      <div className="value">{selectedPosition.entryPrice.toFixed(5)}</div>
                    </div>
                    <div className="detail-item">
                      <div className="label">Current Price</div>
                      <div className="value">{selectedPosition.currentPrice.toFixed(5)}</div>
                    </div>
                    <div className="detail-item">
                      <div className="label">Size</div>
                      <div className="value">{selectedPosition.size}</div>
                    </div>
                    <div className="detail-item">
                      <div className="label">Open Time</div>
                      <div className="value">{new Date(selectedPosition.openTime).toLocaleString()}</div>
                    </div>
                    <div className="detail-item">
                      <div className="label">Stop Loss</div>
                      <div className="value">{selectedPosition.stopLoss.toFixed(5)}</div>
                    </div>
                    <div className="detail-item">
                      <div className="label">Take Profit</div>
                      <div className="value">{selectedPosition.takeProfit.toFixed(5)}</div>
                    </div>
                    <div className="detail-item full-width">
                      <div className="label">P&L</div>
                      <div className={`value large ${selectedPosition.pnl >= 0 ? 'positive' : 'negative'}`}>
                        {selectedPosition.pnl.toFixed(2)} ({selectedPosition.pnlPercentage.toFixed(2)}%)
                      </div>
                    </div>
                  </div>
                  
                  <div className="signal-confidence-section">
                    <h4>Signal Confidence</h4>
                    <SignalConfidenceChart signals={selectedPosition.signals} />
                  </div>
                  
                  <div className="position-actions">
                    <button 
                      className="close-position-btn"
                      onClick={() => handlePositionClose(selectedPosition.id)}
                    >
                      Close Position
                    </button>
                    <button 
                      className="modify-btn"
                      onClick={() => router.push(`/positions/edit/${selectedPosition.id}`)}
                    >
                      Modify
                    </button>
                  </div>
                </Card>
              </div>
            )}
          </div>
        )}
        
        {isMobile && positions.length > 0 && (
          <Card title="P&L Distribution" className="mobile-pnl-chart">
            <div className="pnl-chart">
              <Chart
                type="bar"
                data={pnlChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true
                    }
                  },
                  plugins: {
                    legend: {
                      display: false
                    }
                  }
                }}
              />
            </div>
          </Card>
        )}
      </div>
      
      <style jsx>{`
        .positions-monitor-container {
          padding: 20px;
          max-width: 1400px;
          margin: 0 auto;
        }
        
        .positions-monitor-container.mobile {
          padding: 10px;
        }
        
        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }
        
        .controls {
          display: flex;
          align-items: center;
          gap: 20px;
        }
        
        .refresh-controls {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        
        .total-pnl {
          font-size: 18px;
          font-weight: bold;
          padding: 8px 15px;
          border-radius: 4px;
        }
        
        .total-pnl.positive {
          background-color: rgba(75, 192, 192, 0.2);
          color: rgb(0, 128, 0);
        }
        
        .total-pnl.negative {
          background-color: rgba(255, 99, 132, 0.2);
          color: rgb(220, 53, 69);
        }
        
        .dashboard-content {
          display: flex;
          gap: 20px;
        }
        
        .dashboard-content.side-by-side {
          flex-direction: row;
        }
        
        .dashboard-content.stacked {
          flex-direction: column;
        }
        
        .positions-list {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        
        .position-details {
          width: 400px;
        }
        
        .detail-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
        }
        
        .detail-title {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        
        .detail-title h3 {
          margin: 0;
        }
        
        .direction {
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 14px;
          font-weight: bold;
        }
        
        .direction.long {
          background-color: rgba(75, 192, 192, 0.2);
          color: rgb(0, 128, 0);
        }
        
        .direction.short {
          background-color: rgba(255, 99, 132, 0.2);
          color: rgb(220, 53, 69);
        }
        
        .detail-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 15px;
          margin-bottom: 20px;
        }
        
        .full-width {
          grid-column: 1 / -1;
        }
        
        .detail-item {
          border: 1px solid #eee;
          border-radius: 4px;
          padding: 10px;
        }
        
        .label {
          font-size: 12px;
          color: #666;
          margin-bottom: 4px;
        }
        
        .value {
          font-size: 16px;
          font-weight: 500;
        }
        
        .value.large {
          font-size: 20px;
          font-weight: bold;
        }
        
        .positive {
          color: rgb(0, 128, 0);
        }
        
        .negative {
          color: rgb(220, 53, 69);
        }
        
        .direction-long {
          color: rgb(0, 128, 0);
          font-weight: bold;
        }
        
        .direction-short {
          color: rgb(220, 53, 69);
          font-weight: bold;
        }
        
        .signal-confidence-section {
          margin-bottom: 20px;
        }
        
        .position-actions {
          display: flex;
          gap: 10px;
        }
        
        .close-position-btn {
          background-color: #dc3545;
          color: white;
          border: none;
          padding: 10px 15px;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .modify-btn {
          background-color: #6c757d;
          color: white;
          border: none;
          padding: 10px 15px;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .view-btn {
          background-color: #007bff;
          color: white;
          border: none;
          padding: 5px 10px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
        }
        
        .close-btn {
          background-color: #dc3545;
          color: white;
          border: none;
          padding: 5px 10px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
        }
        
        .pnl-chart {
          height: 300px;
        }
        
        .no-positions {
          padding: 20px;
          text-align: center;
          color: #666;
        }
        
        .loading {
          padding: 20px;
          text-align: center;
          color: #666;
        }
        
        .error-message {
          padding: 20px;
          text-align: center;
          color: #dc3545;
        }
        
        .mobile-pnl-chart {
          margin-top: 20px;
        }
        
        @media (max-width: 768px) {
          .position-details {
            width: 100%;
          }
          
          .dashboard-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 15px;
          }
          
          .controls {
            width: 100%;
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
          }
          
          .pnl-chart {
            height: 200px;
          }
        }
      `}</style>
    </>
  );
};

export default PositionsMonitorPage;
