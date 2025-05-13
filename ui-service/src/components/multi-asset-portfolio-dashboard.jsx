// Multi-asset portfolio dashboard component
import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Table, Spinner, Badge, Nav } from 'react-bootstrap';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { formatCurrency, formatPercentage, formatNumber } from '../utils/formatters';
import { getAssetColor } from '../utils/assetHelpers';

const MultiAssetPortfolioDashboard = ({ accountId }) => {
  const [portfolioData, setPortfolioData] = useState(null);
  const [selectedView, setSelectedView] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPortfolioData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/v1/multi-asset/portfolio/${accountId}/summary`);
        if (!response.ok) {
          throw new Error('Failed to fetch portfolio data');
        }
        const data = await response.json();
        setPortfolioData(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching portfolio data:', err);
        setError('Failed to load portfolio data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolioData();
  }, [accountId]);

  if (loading) {
    return (
      <div className="text-center p-5">
        <Spinner animation="border" />
        <p className="mt-2">Loading portfolio data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-danger" role="alert">
        {error}
      </div>
    );
  }

  if (!portfolioData) {
    return (
      <div className="alert alert-warning" role="alert">
        No portfolio data available.
      </div>
    );
  }

  // Transform data for pie chart
  const allocationData = Object.entries(portfolioData.by_asset_class || {}).map(([asset_class, data]) => ({
    name: asset_class,
    value: data.value,
    color: getAssetColor(asset_class)
  }));

  const performanceData = Object.entries(portfolioData.by_asset_class || {}).map(([asset_class, data]) => ({
    name: asset_class,
    value: data.profit_loss,
    color: getAssetColor(asset_class)
  }));

  return (
    <div className="multi-asset-dashboard">
      <h2 className="mb-4">Portfolio Dashboard</h2>
      
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={3} className="border-end">
              <div className="text-muted small">Total Portfolio Value</div>
              <div className="h3 mb-0">{formatCurrency(portfolioData.total_value || 0)}</div>
            </Col>
            <Col md={3} className="border-end">
              <div className="text-muted small">Unrealized P/L</div>
              <div className={`h4 mb-0 ${portfolioData.unrealized_pl >= 0 ? 'text-success' : 'text-danger'}`}>
                {formatCurrency(portfolioData.unrealized_pl || 0)} 
                ({formatPercentage(portfolioData.unrealized_pl_pct || 0)})
              </div>
            </Col>
            <Col md={3} className="border-end">
              <div className="text-muted small">Today's Change</div>
              <div className={`h5 mb-0 ${portfolioData.daily_change >= 0 ? 'text-success' : 'text-danger'}`}>
                {formatCurrency(portfolioData.daily_change || 0)} 
                ({formatPercentage(portfolioData.daily_change_pct || 0)})
              </div>
            </Col>
            <Col md={3}>
              <div className="text-muted small">Open Positions</div>
              <div className="h4 mb-0">{portfolioData.open_positions_count || 0}</div>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      <Nav variant="tabs" className="mb-3" activeKey={selectedView} onSelect={setSelectedView}>
        <Nav.Item>
          <Nav.Link eventKey="overview">Overview</Nav.Link>
        </Nav.Item>
        <Nav.Item>
          <Nav.Link eventKey="breakdown">Asset Breakdown</Nav.Link>
        </Nav.Item>
        <Nav.Item>
          <Nav.Link eventKey="performance">Performance</Nav.Link>
        </Nav.Item>
        <Nav.Item>
          <Nav.Link eventKey="risk">Risk Analysis</Nav.Link>
        </Nav.Item>
      </Nav>

      {selectedView === 'overview' && (
        <Row>
          <Col md={5}>
            <Card className="h-100">
              <Card.Header>Asset Allocation</Card.Header>
              <Card.Body>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={allocationData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {allocationData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => formatCurrency(value)} />
                  </PieChart>
                </ResponsiveContainer>
              </Card.Body>
            </Card>
          </Col>
          <Col md={7}>
            <Card className="h-100">
              <Card.Header>Asset Class Summary</Card.Header>
              <Card.Body>
                <Table responsive>
                  <thead>
                    <tr>
                      <th>Asset Class</th>
                      <th>Positions</th>
                      <th>Value</th>
                      <th>Allocation</th>
                      <th>P/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(portfolioData.by_asset_class || {}).map(([assetClass, data]) => (
                      <tr key={assetClass}>
                        <td>
                          <Badge bg={getAssetColor(assetClass)} className="me-2">{assetClass}</Badge>
                        </td>
                        <td>{data.count}</td>
                        <td>{formatCurrency(data.value)}</td>
                        <td>{formatPercentage(data.allocation_pct)}</td>
                        <td className={data.profit_loss >= 0 ? 'text-success' : 'text-danger'}>
                          {formatCurrency(data.profit_loss)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}

      {selectedView === 'breakdown' && (
        <Row>
          <Col md={12}>
            <Card>
              <Card.Header>Detailed Asset Breakdown</Card.Header>
              <Card.Body>
                <Table responsive>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Asset Class</th>
                      <th>Quantity</th>
                      <th>Entry Price</th>
                      <th>Current Price</th>
                      <th>Value</th>
                      <th>P/L</th>
                      <th>P/L %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(portfolioData.positions || []).map((position) => (
                      <tr key={position.id}>
                        <td>{position.symbol}</td>
                        <td>
                          <Badge bg={getAssetColor(position.asset_class)}>{position.asset_class}</Badge>
                        </td>
                        <td>{formatNumber(position.quantity)}</td>
                        <td>{formatCurrency(position.entry_price)}</td>
                        <td>{formatCurrency(position.current_price)}</td>
                        <td>{formatCurrency(position.current_value)}</td>
                        <td className={position.unrealized_pl >= 0 ? 'text-success' : 'text-danger'}>
                          {formatCurrency(position.unrealized_pl)}
                        </td>
                        <td className={position.unrealized_pl_pct >= 0 ? 'text-success' : 'text-danger'}>
                          {formatPercentage(position.unrealized_pl_pct)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}

      {selectedView === 'performance' && (
        <Row>
          <Col md={6}>
            <Card className="mb-4">
              <Card.Header>Performance by Asset Class</Card.Header>
              <Card.Body>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip formatter={(value) => formatCurrency(value)} />
                    <Legend />
                    <Bar dataKey="value" name="P/L">
                      {performanceData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.value >= 0 ? '#28a745' : '#dc3545'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6}>
            <Card className="mb-4">
              <Card.Header>Performance Metrics</Card.Header>
              <Card.Body>
                <Table borderless>
                  <tbody>
                    <tr>
                      <td>Daily Return</td>
                      <td className={`text-end ${portfolioData.daily_return >= 0 ? 'text-success' : 'text-danger'}`}>
                        {formatPercentage(portfolioData.daily_return || 0)}
                      </td>
                    </tr>
                    <tr>
                      <td>Weekly Return</td>
                      <td className={`text-end ${portfolioData.weekly_return >= 0 ? 'text-success' : 'text-danger'}`}>
                        {formatPercentage(portfolioData.weekly_return || 0)}
                      </td>
                    </tr>
                    <tr>
                      <td>Monthly Return</td>
                      <td className={`text-end ${portfolioData.monthly_return >= 0 ? 'text-success' : 'text-danger'}`}>
                        {formatPercentage(portfolioData.monthly_return || 0)}
                      </td>
                    </tr>
                    <tr>
                      <td>YTD Return</td>
                      <td className={`text-end ${portfolioData.ytd_return >= 0 ? 'text-success' : 'text-danger'}`}>
                        {formatPercentage(portfolioData.ytd_return || 0)}
                      </td>
                    </tr>
                    <tr>
                      <td>Best Performing Asset</td>
                      <td className="text-end">{portfolioData.best_performing_asset || 'N/A'}</td>
                    </tr>
                    <tr>
                      <td>Worst Performing Asset</td>
                      <td className="text-end">{portfolioData.worst_performing_asset || 'N/A'}</td>
                    </tr>
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}

      {selectedView === 'risk' && (
        <Row>
          <Col md={6}>
            <Card className="mb-4">
              <Card.Header>Risk Metrics</Card.Header>
              <Card.Body>
                <Table borderless>
                  <tbody>
                    <tr>
                      <td>Value at Risk (95%)</td>
                      <td className="text-end">{formatCurrency(portfolioData.cross_asset_risk?.value_at_risk?.var_95 || 0)}</td>
                    </tr>
                    <tr>
                      <td>Value at Risk (99%)</td>
                      <td className="text-end">{formatCurrency(portfolioData.cross_asset_risk?.value_at_risk?.var_99 || 0)}</td>
                    </tr>
                    <tr>
                      <td>Concentration Score</td>
                      <td className="text-end">
                        {formatNumber(portfolioData.cross_asset_risk?.concentration_risk?.concentration_score || 0, 2)}
                      </td>
                    </tr>
                    <tr>
                      <td>Cross-Asset Correlation</td>
                      <td className="text-end">
                        {formatNumber(portfolioData.cross_asset_risk?.cross_correlation || 0, 2)}
                      </td>
                    </tr>
                    <tr>
                      <td>Diversification Score</td>
                      <td className="text-end">
                        {formatNumber(portfolioData.cross_asset_risk?.diversification_score || 0, 2)}
                      </td>
                    </tr>
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6}>
            <Card className="mb-4">
              <Card.Header>Risk Analysis</Card.Header>
              <Card.Body>
                <div className="mb-3">
                  <h6>Most Concentrated Asset Class</h6>
                  <p>{portfolioData.cross_asset_risk?.concentration_risk?.max_concentrated_class || 'None'}</p>
                </div>
                <div className="mb-3">
                  <h6>Risk Assessment</h6>
                  <p>{portfolioData.risk_assessment || 'Moderate risk with balanced asset allocation.'}</p>
                </div>
                <div>
                  <h6>Recommendation</h6>
                  <p>{portfolioData.risk_recommendation || 'Consider rebalancing to optimize risk-adjusted returns.'}</p>
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default MultiAssetPortfolioDashboard;
