# Common JS Library

This library contains shared JavaScript utilities for the Forex Trading Platform.

## Features

- Security middleware for API authentication
- JWT token validation
- Common error handling

## Installation

```bash
npm install ../common-js-lib
```

## Usage

```javascript
const { validateApiKey } = require('common-js-lib/security');

// Use the utility functions
app.use(validateApiKey);
```

## Testing

```bash
npm test
```
