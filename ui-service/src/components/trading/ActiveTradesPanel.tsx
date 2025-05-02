import React from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface Trade {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  openPrice: number;
  currentPrice: number;
  size: number;
  pnl: number;
  pnlPercentage: number;
}

const ActiveTradesPanel: React.FC = () => {
  const [trades, setTrades] = React.useState<Trade[]>([]);

  // In a real implementation, this would be connected to your WebSocket stream
  React.useEffect(() => {
    const mockTrades: Trade[] = [
      {
        id: '1',
        symbol: 'EUR/USD',
        type: 'BUY',
        openPrice: 1.0500,
        currentPrice: 1.0520,
        size: 100000,
        pnl: 200,
        pnlPercentage: 0.19,
      },
      // Add more mock trades as needed
    ];
    setTrades(mockTrades);
  }, []);

  const handleCloseTrade = (tradeId: string) => {
    // Implement trade closing logic here
    console.log('Closing trade:', tradeId);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Active Trades
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Size</TableCell>
              <TableCell>P&L</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {trades.map((trade) => (
              <TableRow key={trade.id}>
                <TableCell>
                  <Typography variant="body2">{trade.symbol}</Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    icon={trade.type === 'BUY' ? <TrendingUpIcon /> : <TrendingDownIcon />}
                    label={trade.type}
                    color={trade.type === 'BUY' ? 'success' : 'error'}
                    size="small"
                  />
                </TableCell>
                <TableCell>{trade.size.toLocaleString()}</TableCell>
                <TableCell>
                  <Typography
                    color={trade.pnl >= 0 ? 'success.main' : 'error.main'}
                    variant="body2"
                  >
                    ${trade.pnl.toFixed(2)}
                    <Typography
                      component="span"
                      variant="caption"
                      sx={{ ml: 0.5 }}
                    >
                      ({trade.pnlPercentage.toFixed(2)}%)
                    </Typography>
                  </Typography>
                </TableCell>
                <TableCell>
                  <IconButton
                    size="small"
                    onClick={() => handleCloseTrade(trade.id)}
                    color="error"
                  >
                    <CloseIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default ActiveTradesPanel;
