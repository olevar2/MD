const security = require('../security');

test('API key validation helper function exists', () => {
  expect(typeof security.validateApiKey).toBe('function');
});
