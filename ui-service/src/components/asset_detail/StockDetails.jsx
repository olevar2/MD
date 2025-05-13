// Stock-specific detail component
import React from 'react';
import { Badge, Row, Col, Table, Card } from 'react-bootstrap';
import { formatNumber, formatCurrency, formatPercentage } from '../../utils/formatters';

const StockDetails = ({ symbol, assetInfo }) => {
  const {
    trading_parameters = {},
    market_data = {},
    metadata = {}
  } = assetInfo;
  
  return (
    <div className="stock-details">
      <Row className="mb-4">
        <Col md={6}>
          <h4>
            {assetInfo.display_name || symbol} 
            <Badge bg="success" className="ms-2">Stocks</Badge>
          </h4>
          <p className="text-secondary">{metadata.company_name || ''}</p>
          <p className="text-muted small">{metadata.exchange || ''} • {metadata.sector || ''}</p>
        </Col>
        
        <Col md={6} className="text-end">
          <div className="current-price display-6">
            {formatCurrency(market_data.current_price || 0)}
          </div>
          <div className={`daily-change ${(market_data.daily_change_pct || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
            {formatCurrency(market_data.daily_change || 0)} 
            ({formatPercentage(market_data.daily_change_pct || 0)})
          </div>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={6}>
          <Card className="h-100">
            <Card.Header>Company Information</Card.Header>
            <Card.Body>
              <Table borderless size="sm">
                <tbody>
                  <tr>
                    <td>Market Cap</td>
                    <td className="text-end">{formatCurrency(market_data.market_cap || 0)}</td>
                  </tr>
                  <tr>
                    <td>P/E Ratio</td>
                    <td className="text-end">{formatNumber(metadata.pe_ratio || 0, 2)}</td>
                  </tr>
                  <tr>
                    <td>EPS (TTM)</td>
                    <td className="text-end">{formatCurrency(metadata.eps || 0)}</td>
                  </tr>
                  <tr>
                    <td>Dividend Yield</td>
                    <td className="text-end">{formatPercentage(metadata.dividend_yield || 0)}</td>
                  </tr>
                  <tr>
                    <td>52 Week Range</td>
                    <td className="text-end">
                      {formatCurrency(market_data.yearly_low || 0)} - {formatCurrency(market_data.yearly_high || 0)}
                    </td>
                  </tr>
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={6}>
          <Card className="h-100">
            <Card.Header>Market Details</Card.Header>
            <Card.Body>
              <Table borderless size="sm">
                <tbody>
                  <tr>
                    <td>Daily Range</td>
                    <td className="text-end">
                      {formatCurrency(market_data.daily_low || 0)} - 
                      {formatCurrency(market_data.daily_high || 0)}
                    </td>
                  </tr>
                  <tr>
                    <td>Volume</td>
                    <td className="text-end">{formatNumber(market_data.volume || 0, 0)}</td>
                  </tr>
                  <tr>
                    <td>Average Volume</td>
                    <td className="text-end">{formatNumber(market_data.avg_volume || 0, 0)}</td>
                  </tr>
                  <tr>
                    <td>Previous Close</td>
                    <td className="text-end">{formatCurrency(market_data.prev_close || 0)}</td>
                  </tr>
                  <tr>
                    <td>Open</td>
                    <td className="text-end">{formatCurrency(market_data.open || 0)}</td>
                  </tr>
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>Technical and Fundamental Analysis</Card.Header>
            <Card.Body>
              <Row>
                <Col md={4}>
                  <h6>Technical Indicators</h6>
                  <Table borderless size="sm">
                    <tbody>
                      <tr>
                        <td>RSI (14)</td>
                        <td className="text-end">
                          <span className={
                            market_data.rsi_14 > 70 ? 'text-danger' : 
                            market_data.rsi_14 < 30 ? 'text-success' : 
                            'text-muted'
                          }>
                            {formatNumber(market_data.rsi_14 || 50, 2)}
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td>MACD</td>
                        <td className={`text-end ${(market_data.macd || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                          {formatNumber(market_data.macd || 0, 2)}
                        </td>
                      </tr>
                      <tr>
                        <td>20-Day SMA</td>
                        <td className="text-end">{formatCurrency(market_data.sma_20 || 0)}</td>
                      </tr>
                      <tr>
                        <td>50-Day SMA</td>
                        <td className="text-end">{formatCurrency(market_data.sma_50 || 0)}</td>
                      </tr>
                    </tbody>
                  </Table>
                </Col>
                <Col md={4}>
                  <h6>Key Financial Metrics</h6>
                  <Table borderless size="sm">
                    <tbody>
                      <tr>
                        <td>Profit Margin</td>
                        <td className="text-end">{formatPercentage(metadata.profit_margin || 0)}</td>
                      </tr>
                      <tr>
                        <td>Return on Equity</td>
                        <td className="text-end">{formatPercentage(metadata.roe || 0)}</td>
                      </tr>
                      <tr>
                        <td>Debt to Equity</td>
                        <td className="text-end">{formatNumber(metadata.debt_to_equity || 0, 2)}</td>
                      </tr>
                      <tr>
                        <td>Current Ratio</td>
                        <td className="text-end">{formatNumber(metadata.current_ratio || 0, 2)}</td>
                      </tr>
                    </tbody>
                  </Table>
                </Col>
                <Col md={4}>
                  <h6>Analyst Ratings</h6>
                  <div className="rating-summary mb-2">
                    <div className="fw-bold">Consensus: {metadata.analyst_consensus || 'N/A'}</div>
                    <div className="small text-muted">Based on {metadata.analyst_count || 0} analysts</div>
                  </div>
                  <div className="rating-breakdown">
                    <div className="d-flex justify-content-between mb-1">
                      <span>Buy</span>
                      <span>{formatPercentage(metadata.buy_ratings_pct || 0)}</span>
                    </div>
                    <div className="d-flex justify-content-between mb-1">
                      <span>Hold</span>
                      <span>{formatPercentage(metadata.hold_ratings_pct || 0)}</span>
                    </div>
                    <div className="d-flex justify-content-between mb-1">
                      <span>Sell</span>
                      <span>{formatPercentage(metadata.sell_ratings_pct || 0)}</span>
                    </div>
                  </div>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default StockDetails;
