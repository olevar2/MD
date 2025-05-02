import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Grid,
  Typography,
  IconButton
} from '@mui/material';
import { ArrowDropUp, ArrowDropDown, Remove } from '@mui/icons-material';

export interface WatchlistItem {
  symbol: string;
  bid: number;
  ask: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
}

interface WatchlistProps {
  items: WatchlistItem[];
  onSymbolSelect: (symbol: string) => void;
  onRemoveSymbol: (symbol: string) => void;
  selectedSymbol?: string;
}

const Watchlist: React.FC<WatchlistProps> = ({
  items,
  onSymbolSelect,
  onRemoveSymbol,
  selectedSymbol
}) => {
  const getChangeColor = (change: number) => {
    if (change > 0) return 'success.main';
    if (change < 0) return 'error.main';
    return 'text.secondary';
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) return <ArrowDropUp color="success" />;
    if (change < 0) return <ArrowDropDown color="error" />;
    return <Remove color="action" />;
  };

  return (
    <TableContainer component={Paper}>
      <Table size="small" aria-label="watchlist table">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Bid/Ask</TableCell>
            <TableCell align="right">Change</TableCell>
            <TableCell align="right">Volume</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {items.map((item) => (
            <TableRow
              key={item.symbol}
              hover
              selected={item.symbol === selectedSymbol}
              onClick={() => onSymbolSelect(item.symbol)}
              sx={{
                cursor: 'pointer',
                '&:last-child td, &:last-child th': { border: 0 }
              }}
            >
              <TableCell component="th" scope="row">
                <Typography variant="subtitle2" fontWeight="bold">
                  {item.symbol}
                </Typography>
              </TableCell>
              <TableCell align="right">
                <Grid container direction="column">
                  <Typography variant="caption" color="success.main">
                    {item.ask.toFixed(5)}
                  </Typography>
                  <Typography variant="caption" color="error.main">
                    {item.bid.toFixed(5)}
                  </Typography>
                </Grid>
              </TableCell>
              <TableCell align="right">
                <Grid container alignItems="center" justifyContent="flex-end">
                  {getChangeIcon(item.change)}
                  <Typography
                    variant="body2"
                    sx={{ color: getChangeColor(item.change) }}
                  >
                    {item.changePercent.toFixed(2)}%
                  </Typography>
                </Grid>
              </TableCell>
              <TableCell align="right">
                <Typography variant="caption">
                  {item.volume.toLocaleString()}
                </Typography>
              </TableCell>
              <TableCell align="right">
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveSymbol(item.symbol);
                  }}
                  aria-label={`remove ${item.symbol} from watchlist`}
                >
                  <Remove fontSize="small" />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
          {items.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} align="center">
                <Typography variant="caption">
                  No symbols in watchlist
                </Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default Watchlist;
