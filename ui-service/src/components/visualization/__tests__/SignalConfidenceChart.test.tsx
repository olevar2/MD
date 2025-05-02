import React from 'react';
import { render, screen } from '@testing-library/react';
import SignalConfidenceChart, { SignalDataPoint } from '../SignalConfidenceChart';

// Mock recharts components since we can't render them in tests
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => 
    <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => 
    <div data-testid="line-chart">{children}</div>,
  Line: ({ dataKey, name }: { dataKey: string, name: string }) => 
    <div data-testid={`line-${name.toLowerCase().replace(/\s+/g, '-')}`}>{dataKey}</div>,
  XAxis: ({ label }: { label: { value: string } }) => 
    <div data-testid="x-axis">{label.value}</div>,
  YAxis: ({ label }: { label: { value: string } }) => 
    <div data-testid="y-axis">{label.value}</div>,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />
}));

describe('SignalConfidenceChart', () => {
  const mockData: SignalDataPoint[] = [
    { timestamp: 1619827200000, signal: 'buy', confidence: 0.85, direction: 'buy' },
    { timestamp: 1619830800000, signal: 'sell', confidence: 0.75, direction: 'sell' },
    { timestamp: 1619834400000, signal: 'hold', confidence: 0.60, direction: 'neutral' }
  ];

  it('renders with required props', () => {
    render(<SignalConfidenceChart signalData={mockData} />);
    
    // Check if main container exists with correct attributes
    const chart = screen.getByRole('figure', { name: /signal confidence chart/i });
    expect(chart).toBeInTheDocument();

    // Check if main components are rendered
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    expect(screen.getByTestId('cartesian-grid')).toBeInTheDocument();
    expect(screen.getByTestId('x-axis')).toBeInTheDocument();
    expect(screen.getByTestId('y-axis')).toBeInTheDocument();
    expect(screen.getByTestId('tooltip')).toBeInTheDocument();
    expect(screen.getByTestId('legend')).toBeInTheDocument();
  });

  it('renders both threshold and confidence lines', () => {
    render(<SignalConfidenceChart signalData={mockData} />);
    
    expect(screen.getByTestId('line-threshold')).toBeInTheDocument();
    expect(screen.getByTestId('line-signal-confidence')).toBeInTheDocument();
  });

  it('has correct axis labels', () => {
    render(<SignalConfidenceChart signalData={mockData} />);
    
    expect(screen.getByText('Time')).toBeInTheDocument();
    expect(screen.getByText('Confidence')).toBeInTheDocument();
  });

  it('applies custom height', () => {
    const customHeight = 500;
    render(<SignalConfidenceChart signalData={mockData} height={customHeight} />);
    
    const container = screen.getByTestId('responsive-container');
    expect(container).toHaveStyle({ height: customHeight });
  });
});
