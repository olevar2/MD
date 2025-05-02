// Cryptocurrency-specific detail component
import React from 'react';
import { Badge, Row, Col, Table, Card } from 'react-bootstrap';
import { formatNumber, formatCurrency, formatPercentage } from '../../utils/formatters';

const CryptoDetails = ({ symbol, assetInfo }) => {
  const {
    base_currency = '',
    quote_currency = '',
    trading_parameters = {},
    market_data = {}
  } = assetInfo;
  
  return (
    <div className="crypto-details">
      <Row className="mb-4">
        <Col md={6}>
          <h4>
            {base_currency}/{quote_currency} 
            <Badge bg="warning" className="ms-2">Crypto</Badge>
          </h4>
          <p className="text-secondary">{assetInfo.description || `${base_currency} crypto asset`}</p>
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
            <Card.Header>Key Information</Card.Header>
            <Card.Body>
              <Table borderless size="sm">
                <tbody>
                  <tr>
                    <td>Market Cap</td>
                    <td className="text-end">{formatCurrency(market_data.market_cap || 0)}</td>
                  </tr>
                  <tr>
                    <td>24h Volume</td>
                    <td className="text-end">{formatCurrency(market_data.volume_24h || 0)}</td>
                  </tr>
                  <tr>
                    <td>Circulating Supply</td>
                    <td className="text-end">{formatNumber(market_data.circulating_supply || 0, 0)}</td>
                  </tr>
                  <tr>
                    <td>Max Supply</td>
                    <td className="text-end">{market_data.max_supply ? formatNumber(market_data.max_supply, 0) : 'Unlimited'}</td>
                  </tr>
                  <tr>
                    <td>Trading Hours</td>
                    <td className="text-end">24/7</td>
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
                    <td>24h Change</td>
                    <td className={`text-end ${(market_data.daily_change_pct || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                      {formatPercentage(market_data.daily_change_pct || 0)}
                    </td>
                  </tr>
                  <tr>
                    <td>7d Change</td>
                    <td className={`text-end ${(market_data.weekly_change_pct || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                      {formatPercentage(market_data.weekly_change_pct || 0)}
                    </td>
                  </tr>
                  <tr>
                    <td>30d Change</td>
                    <td className={`text-end ${(market_data.monthly_change_pct || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                      {formatPercentage(market_data.monthly_change_pct || 0)}
                    </td>
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
            <Card.Header>Technical Analysis</Card.Header>
            <Card.Body>
              <Row>
                <Col md={4}>
                  <h6>Volume Analysis</h6>
                  <Table borderless size="sm">
                    <tbody>
                      <tr>
                        <td>Volume Change 24h</td>
                        <td className={`text-end ${(market_data.volume_change_24h || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                          {formatPercentage(market_data.volume_change_24h || 0)}
                        </td>
                      </tr>
                      <tr>
                        <td>Buy Volume %</td>
                        <td className="text-end">
                          {formatPercentage(market_data.buy_volume_pct || 0.5)}
                        </td>
                      </tr>
                      <tr>
                        <td>Sell Volume %</td>
                        <td className="text-end">
                          {formatPercentage(market_data.sell_volume_pct || 0.5)}
                        </td>
                      </tr>
                    </tbody>
                  </Table>
                </Col>
                <Col md={4}>
                  <h6>Support Levels</h6>
                  <ul className="list-unstyled">
                    {(market_data.support_levels || []).map((level, index) => (
                      <li key={`support-${index}`} className="mb-1">
                        <span className="text-primary">S{index + 1}:</span> {formatCurrency(level)}
                      </li>
                    ))}
                  </ul>
                </Col>
                <Col md={4}>
                  <h6>Resistance Levels</h6>
                  <ul className="list-unstyled">
                    {(market_data.resistance_levels || []).map((level, index) => (
                      <li key={`resistance-${index}`} className="mb-1">
                        <span className="text-danger">R{index + 1}:</span> {formatCurrency(level)}
                      </li>
                    ))}
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

export default CryptoDetails;
