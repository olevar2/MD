// Forex-specific detail component
import React from 'react';
import { Badge, Row, Col, Table, Card } from 'react-bootstrap';
import { formatNumber, formatCurrency, formatPercentage } from '../../utils/formatters';

const ForexDetails = ({ symbol, assetInfo }) => {
  const {
    base_currency = '',
    quote_currency = '',
    trading_parameters = {},
    market_data = {}
  } = assetInfo;
  
  return (
    <div className="forex-details">
      <Row className="mb-4">
        <Col md={6}>
          <h4>
            {base_currency}/{quote_currency} 
            <Badge bg="primary" className="ms-2">Forex</Badge>
          </h4>
          <p className="text-secondary">{assetInfo.description || `Exchange rate for ${base_currency} to ${quote_currency}`}</p>
        </Col>
        
        <Col md={6} className="text-end">
          <div className="current-price display-6">
            {formatNumber(market_data.current_price || 0, 
              trading_parameters.price_precision || 4)}
          </div>
          <div className={`daily-change ${(market_data.daily_change_pct || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
            {formatNumber(market_data.daily_change || 0, trading_parameters.price_precision || 4)} 
            ({formatPercentage(market_data.daily_change_pct || 0)})
          </div>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={6}>
          <Card className="h-100">
            <Card.Header>Key Information</Card.Header>
            <Card.Body>
              <Table borderless size="sm">
                <tbody>
                  <tr>
                    <td>Pip Value</td>
                    <td className="text-end">{formatNumber(trading_parameters.pip_value || 0.0001, 6)}</td>
                  </tr>
                  <tr>
                    <td>Standard Lot Size</td>
                    <td className="text-end">100,000</td>
                  </tr>
                  <tr>
                    <td>Margin Requirement</td>
                    <td className="text-end">{formatPercentage(trading_parameters.margin_rate || 0.03)}</td>
                  </tr>
                  <tr>
                    <td>Spread (Avg)</td>
                    <td className="text-end">{formatNumber(trading_parameters.avg_spread || 1.2, 1)} pips</td>
                  </tr>
                  <tr>
                    <td>Trading Hours</td>
                    <td className="text-end">24/5</td>
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
                      {formatNumber(market_data.daily_low || 0, trading_parameters.price_precision || 4)} - 
                      {formatNumber(market_data.daily_high || 0, trading_parameters.price_precision || 4)}
                    </td>
                  </tr>
                  <tr>
                    <td>Daily Range (Pips)</td>
                    <td className="text-end">
                      {formatNumber((market_data.daily_high - market_data.daily_low) / 
                        (trading_parameters.pip_value || 0.0001), 1)} pips
                    </td>
                  </tr>
                  <tr>
                    <td>Weekly Range</td>
                    <td className="text-end">
                      {formatNumber(market_data.weekly_low || 0, trading_parameters.price_precision || 4)} - 
                      {formatNumber(market_data.weekly_high || 0, trading_parameters.price_precision || 4)}
                    </td>
                  </tr>
                  <tr>
                    <td>Current Session</td>
                    <td className="text-end">{market_data.current_session || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td>Volatility (Daily)</td>
                    <td className="text-end">{formatPercentage(market_data.volatility || 0)}</td>
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
            <Card.Header>Technical Levels</Card.Header>
            <Card.Body>
              <Row>
                <Col md={4}>
                  <h6>Support Levels</h6>
                  <ul className="list-unstyled">
                    {(market_data.support_levels || []).map((level, index) => (
                      <li key={`support-${index}`} className="mb-1">
                        <span className="text-primary">S{index + 1}:</span> {formatNumber(level, trading_parameters.price_precision || 4)}
                      </li>
                    ))}
                  </ul>
                </Col>
                <Col md={4}>
                  <h6>Resistance Levels</h6>
                  <ul className="list-unstyled">
                    {(market_data.resistance_levels || []).map((level, index) => (
                      <li key={`resistance-${index}`} className="mb-1">
                        <span className="text-danger">R{index + 1}:</span> {formatNumber(level, trading_parameters.price_precision || 4)}
                      </li>
                    ))}
                  </ul>
                </Col>
                <Col md={4}>
                  <h6>Pivots</h6>
                  <ul className="list-unstyled">
                    <li className="mb-1">
                      <span className="text-secondary">PP:</span> {formatNumber(market_data.pivot_point || 0, trading_parameters.price_precision || 4)}
                    </li>
                    <li className="mb-1">
                      <span className="text-info">Daily Pivot:</span> {formatNumber(market_data.daily_pivot || 0, trading_parameters.price_precision || 4)}
                    </li>
                  </ul>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ForexDetails;
