import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import RegimeAwareDashboard from '../RegimeAwareDashboard';
import { useMarketData } from '../../../hooks/useMarketData';
import { useMarketRegime } from '../../../hooks/useMarketRegime';

// Mock the hooks
jest.mock('../../../hooks/useMarketData');
jest.mock('../../../hooks/useMarketRegime');

// Mock child components
jest.mock('../../visualization/SignalConfidenceChart', () => ({
  SignalConfidenceChart: () => <div data-testid="signal-confidence-chart">SignalConfidenceChart</div>
}));
jest.mock('../MarketRegimeIndicator', () => ({
  MarketRegimeIndicator: () => <div data-testid="market-regime-indicator">MarketRegimeIndicator</div>
}));
jest.mock('../PositionTable', () => ({
  PositionTable: () => <div data-testid="position-table">PositionTable</div>
}));
jest.mock('../OrderEntry', () => ({
  OrderEntry: () => <div data-testid="order-entry">OrderEntry</div>
}));

const mockTheme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={mockTheme}>
      {component}
    </ThemeProvider>
  );
};

describe('RegimeAwareDashboard', () => {
  const mockProps = {
    accountId: 'test-account',
    symbol: 'EURUSD',
    onRegimeChange: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows loading state when data is being fetched', () => {
    (useMarketData as jest.Mock).mockReturnValue({
      marketData: null,
      isLoading: true,
      error: null
    });
    (useMarketRegime as jest.Mock).mockReturnValue({
      regime: null,
      isLoading: true,
      error: null
    });

    renderWithTheme(<RegimeAwareDashboard {...mockProps} />);
    
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('shows error state when data fetch fails', () => {
    const errorMessage = 'Failed to fetch data';
    (useMarketData as jest.Mock).mockReturnValue({
      marketData: null,
      isLoading: false,
      error: new Error(errorMessage)
    });
    (useMarketRegime as jest.Mock).mockReturnValue({
      regime: null,
      isLoading: false,
      error: null
    });

    renderWithTheme(<RegimeAwareDashboard {...mockProps} />);
    
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('renders high volatility layout correctly', async () => {
    const mockRegime = {
      type: 'HighVolatility',
      signals: [],
      volatility: 0.2,
      trend: 'up'
    };

    (useMarketData as jest.Mock).mockReturnValue({
      marketData: {},
      isLoading: false,
      error: null
    });
    (useMarketRegime as jest.Mock).mockReturnValue({
      regime: mockRegime,
      isLoading: false,
      error: null
    });

    renderWithTheme(<RegimeAwareDashboard {...mockProps} />);
    
    await waitFor(() => {
      expect(screen.getByTestId('signal-confidence-chart')).toBeInTheDocument();
      expect(screen.getByTestId('market-regime-indicator')).toBeInTheDocument();
    });

    expect(mockProps.onRegimeChange).toHaveBeenCalledWith(mockRegime);
  });

  it('renders low volatility layout correctly', async () => {
    (useMarketRegime as jest.Mock).mockReturnValue({
      regime: {
        type: 'LowVolatility',
        signals: [],
        volatility: 0.05,
        trend: 'sideways'
      },
      isLoading: false,
      error: null
    });
    (useMarketData as jest.Mock).mockReturnValue({
      marketData: {},
      isLoading: false,
      error: null
    });

    renderWithTheme(<RegimeAwareDashboard {...mockProps} />);
    
    await waitFor(() => {
      expect(screen.getByTestId('signal-confidence-chart')).toBeInTheDocument();
      expect(screen.getByTestId('order-entry')).toBeInTheDocument();
    });
  });

  it('renders trending market layout correctly', async () => {
    (useMarketRegime as jest.Mock).mockReturnValue({
      regime: {
        type: 'Trending',
        signals: [],
        trend: 'up'
      },
      isLoading: false,
      error: null
    });
    (useMarketData as jest.Mock).mockReturnValue({
      marketData: {},
      isLoading: false,
      error: null
    });

    renderWithTheme(<RegimeAwareDashboard {...mockProps} />);
    
    await waitFor(() => {
      expect(screen.getByTestId('signal-confidence-chart')).toBeInTheDocument();
      expect(screen.getByTestId('position-table')).toBeInTheDocument();
    });
  });

  it('renders normal market layout correctly', async () => {
    (useMarketRegime as jest.Mock).mockReturnValue({
      regime: {
        type: 'Normal',
        signals: []
      },
      isLoading: false,
      error: null
    });
    (useMarketData as jest.Mock).mockReturnValue({
      marketData: {},
      isLoading: false,
      error: null
    });

    renderWithTheme(<RegimeAwareDashboard {...mockProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Normal Market Conditions')).toBeInTheDocument();
      expect(screen.getByTestId('signal-confidence-chart')).toBeInTheDocument();
    });
  });
});
