import React, { useState, useEffect } from 'react';
import { Grid, Paper, Box, useTheme } from '@mui/material';
import CandlestickChart, { Candle } from '../charts/CandlestickChart';
import OrderEntry, { OrderFormData } from './OrderEntry';
import OrdersList, { Order } from './OrdersList';
import PositionsMonitor, { Position } from './PositionsMonitor';
import Watchlist, { WatchlistItem } from './Watchlist';
import SignalVisualizer, { TradingSignal, NewsItem } from './SignalVisualizer';

interface TradingDashboardProps {
  accountId: string;
  availableSymbols: string[];
  onOrderSubmit: (order: OrderFormData) => Promise<void>;
  onOrderCancel: (orderId: string) => Promise<void>;
  onOrderModify: (orderId: string) => void;
  onPositionClose: (positionId: string) => Promise<void>;
  onPositionModify: (positionId: string) => void;
  onSymbolSelect: (symbol: string) => void;
  onWatchlistUpdate: (symbols: string[]) => void;
  onSignalSelect: (signal: TradingSignal) => void;
  onNewsSelect: (news: NewsItem) => void;
}

const TradingDashboard: React.FC<TradingDashboardProps> = ({
  accountId,
  availableSymbols,
  onOrderSubmit,
  onOrderCancel,
  onOrderModify,
  onPositionClose,
  onPositionModify,
  onSymbolSelect,
  onWatchlistUpdate,
  onSignalSelect,
  onNewsSelect
}) => {
  const theme = useTheme();
  const [selectedSymbol, setSelectedSymbol] = useState<string>(availableSymbols[0]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [marketData, setMarketData] = useState<{
    candles: Candle[];
    currentPrice: number;
  }>({ candles: [], currentPrice: 0 });
  const [orders, setOrders] = useState<Order[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [news, setNews] = useState<NewsItem[]>([]);

  // Simulate fetching initial data
  useEffect(() => {    fetchMarketData(selectedSymbol);
    fetchOrders(accountId);
    fetchPositions(accountId);
    fetchWatchlist(accountId);
    fetchSignals(selectedSymbol);
    fetchNews();
  }, [selectedSymbol, accountId]);

  const handleOrderSubmit = async (order: OrderFormData) => {
    setIsSubmitting(true);
    try {
      await onOrderSubmit(order);
      // Refresh orders after submission
      // await fetchOrders(accountId);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSymbolSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
    onSymbolSelect(symbol);
  };

  return (
    <Box sx={{ p: 2, height: '100vh', bgcolor: theme.palette.background.default }}>
      <Grid container spacing={2}>
        {/* Chart and Order Entry */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2, mb: 2 }}>
            <CandlestickChart
              data={marketData.candles}
              symbol={selectedSymbol}
              height={400}
            />
          </Paper>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <OrderEntry
                  availableSymbols={availableSymbols}
                  currentPrice={marketData.currentPrice}
                  onSubmit={handleOrderSubmit}
                  isSubmitting={isSubmitting}
                />
              </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <OrdersList
                  orders={orders}
                  onCancelOrder={onOrderCancel}
                  onModifyOrder={onOrderModify}
                />
              </Paper>
            </Grid>
          </Grid>
        </Grid>

        {/* Positions, Watchlist, and Signals */}
        <Grid item xs={12} lg={4}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <PositionsMonitor
                  positions={positions}
                  onClosePosition={onPositionClose}
                  onModifyPosition={onPositionModify}
                />
              </Paper>
            </Grid>
            
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Watchlist
                  items={watchlist}
                  onSymbolSelect={handleSymbolSelect}
                  onRemoveSymbol={(symbol) => {
                    const updatedWatchlist = watchlist.filter(item => item.symbol !== symbol);
                    setWatchlist(updatedWatchlist);
                    onWatchlistUpdate(updatedWatchlist.map(item => item.symbol));
                  }}
                  selectedSymbol={selectedSymbol}
                />
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <SignalVisualizer
                signals={signals}
                news={news}
                onSignalSelect={onSignalSelect}
                onNewsSelect={onNewsSelect}
              />
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TradingDashboard;
