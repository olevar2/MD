# Risk Management Service API Documentation

## Authentication

All API endpoints in the Risk Management Service are protected using API Key authentication. Clients must include a valid API key in the header of each request:

```
X-API-Key: <your-api-key>
```

If the API key is missing or invalid, the request will be rejected with an appropriate HTTP error code.

## API Endpoints

### Risk Limits Management

#### Create Risk Limit
- **URL**: `/api/v1/risk/limits`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Create a new risk limit for an account.

#### Get Risk Limit
- **URL**: `/api/v1/risk/limits/{limit_id}`
- **Method**: `GET`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Get details for a specific risk limit.

#### Update Risk Limit
- **URL**: `/api/v1/risk/limits/{limit_id}`
- **Method**: `PUT`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Update a risk limit.

#### Get Account Limits
- **URL**: `/api/v1/risk/accounts/{account_id}/limits`
- **Method**: `GET`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Get all risk limits for an account.

### Risk Assessment

#### Check Position Risk
- **URL**: `/api/v1/risk/check/position`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Check if a new position would violate risk limits.

#### Check Portfolio Risk
- **URL**: `/api/v1/risk/check/portfolio/{account_id}`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Check overall portfolio risk against limits.

### Risk Calculations

#### Calculate Position Size
- **URL**: `/api/v1/risk/calculate/position-size`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Calculate position size based on account risk percentage.

#### Calculate Value at Risk (VaR)
- **URL**: `/api/v1/risk/calculate/var`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Calculate Value at Risk (VaR) for a portfolio.

#### Calculate Drawdown Risk
- **URL**: `/api/v1/risk/calculate/drawdown`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Calculate drawdown risk metrics.

#### Calculate Correlation Risk
- **URL**: `/api/v1/risk/calculate/correlation`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Calculate correlation risk for a portfolio of positions.

#### Calculate Maximum Trades
- **URL**: `/api/v1/risk/calculate/max-trades`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Calculate maximum number of simultaneous trades based on risk limits.

### Risk Profiles

#### Create Risk Profile
- **URL**: `/api/v1/risk/profiles`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Create a risk profile with predefined limits.

#### Apply Risk Profile to Account
- **URL**: `/api/v1/risk/accounts/{account_id}/apply-profile/{profile_id}`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Apply a risk profile to an account.

### Dynamic Risk Adjustment

#### Analyze Strategy Weaknesses
- **URL**: `/api/risk/dynamic/strategy/weaknesses`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Analyze a trading strategy for potential weaknesses across market regimes.

#### ML Risk Metrics
- **URL**: `/api/risk/dynamic/ml/metrics`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Generate risk metrics in a format suitable for machine learning model integration.

#### Process ML Feedback
- **URL**: `/api/risk/dynamic/ml/feedback`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Process feedback from ML model predictions to improve risk assessments.

#### Monitor Risk Thresholds
- **URL**: `/api/risk/dynamic/monitor/thresholds`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Monitor risk metrics against thresholds and generate alerts.

#### Trigger Automated Control
- **URL**: `/api/risk/dynamic/control/automated`
- **Method**: `POST`
- **Auth Required**: Yes (`X-API-Key` header)
- **Description**: Trigger automated risk control actions based on alerts.

## Error Responses

The API returns the following error responses for authentication failures:

- **401 Unauthorized**: API key is missing
- **403 Forbidden**: API key is invalid

For other errors, appropriate HTTP status codes will be returned with a descriptive message.
