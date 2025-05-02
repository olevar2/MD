/**
 * @jest/globals is automatically configured by Jest and doesn't require explicit imports.
 * @see https://jestjs.io/docs/getting-started
 */

import axios from 'axios';
import { TradingService, TradingRequest, TradingResponse } from '../../src/services/tradingService';

jest.mock('axios');

describe('TradingService', () => {
  let tradingService: TradingService;
  let mockAxiosCreate: jest.SpyInstance;
  let mockAxiosInstance: any;

  beforeEach(() => {
    jest.clearAllMocks();
    mockAxiosInstance = {
      get: jest.fn(),
      post: jest.fn(),
      patch: jest.fn()
    };
    mockAxiosCreate = jest.spyOn(axios, 'create').mockReturnValue(mockAxiosInstance);
    tradingService = new TradingService('http://api.test.com');
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('should submit buy order successfully', async () => {
    const orderRequest: TradingRequest = {
      symbol: 'EURUSD',
      volume: 1.0,
      price: 1.2345,
      type: 'LIMIT'
    };

    const expectedResponse: TradingResponse = {
      orderId: '12345',
      status: 'PENDING',
      message: 'Order submitted successfully'
    };

    mockAxiosInstance.post.mockResolvedValueOnce({ data: expectedResponse });

    const result = await tradingService.buy(orderRequest);

    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/trading/orders', {
      ...orderRequest,
      side: 'BUY'
    });
    expect(result).toEqual(expectedResponse);
  });

  test('should submit sell order successfully', async () => {
    const orderRequest: TradingRequest = {
      symbol: 'GBPUSD',
      volume: 0.5,
      type: 'MARKET'
    };

    const expectedResponse: TradingResponse = {
      orderId: '12346',
      status: 'FILLED',
      filledPrice: 1.2500
    };

    mockAxiosInstance.post.mockResolvedValueOnce({ data: expectedResponse });

    const result = await tradingService.sell(orderRequest);

    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/trading/orders', {
      ...orderRequest,
      side: 'SELL'
    });
    expect(result).toEqual(expectedResponse);
  });

  test('should fetch order status successfully', async () => {
    const orderId = '12345';
    const expectedResponse: TradingResponse = {
      orderId,
      status: 'FILLED',
      filledPrice: 1.2345
    };

    mockAxiosInstance.get.mockResolvedValueOnce({ data: expectedResponse });

    const result = await tradingService.getOrderStatus(orderId);

    expect(mockAxiosInstance.get).toHaveBeenCalledWith(`/trading/orders/${orderId}`);
    expect(result).toEqual(expectedResponse);
  });

  test('should handle API error gracefully', async () => {
    const orderRequest: TradingRequest = {
      symbol: 'EURUSD',
      volume: 1.0
    };

    const apiError = new Error('Network Error');
    mockAxiosInstance.post.mockRejectedValueOnce(apiError);

    await expect(tradingService.buy(orderRequest))
      .rejects
      .toThrow('Failed to submit buy order: Network Error');

    expect(mockAxiosInstance.post).toHaveBeenCalledWith('/trading/orders', {
      ...orderRequest,
      side: 'BUY'
    });
  });
});
