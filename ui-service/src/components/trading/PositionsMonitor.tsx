import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Chip
} from '@mui/material';
import { Close } from '@mui/icons-material';

export interface Position {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  stopLoss?: number;
  takeProfit?: number;
  margin: number;
  openTime: number;
}

interface PositionsMonitorProps {
  positions: Position[];
  onClosePosition: (positionId: string) => void;
  onModifyPosition: (positionId: string) => void;
}

const PositionsMonitor: React.FC<PositionsMonitorProps> = ({
  positions,
  onClosePosition,
  onModifyPosition
}) => {
  return (
    <TableContainer component={Paper}>
      <Table size="small" aria-label="positions monitor table">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Direction</TableCell>
            <TableCell align="right">Size</TableCell>
            <TableCell align="right">Entry Price</TableCell>
            <TableCell align="right">Current Price</TableCell>
            <TableCell align="right">P&L</TableCell>
            <TableCell align="right">Margin</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((position) => (
            <TableRow
              key={position.id}
              sx={{
                '&:last-child td, &:last-child th': { border: 0 },
                backgroundColor: position.unrealizedPnl >= 0 ? 'success.light' : 'error.light'
              }}
              onClick={() => onModifyPosition(position.id)}
              style={{ cursor: 'pointer' }}
            >
              <TableCell component="th" scope="row">
                {position.symbol}
              </TableCell>
              <TableCell align="right">
                <Chip
                  label={position.direction}
                  color={position.direction === 'LONG' ? 'success' : 'error'}
                  size="small"
                />
              </TableCell>
              <TableCell align="right">{position.size}</TableCell>
              <TableCell align="right">{position.entryPrice.toFixed(5)}</TableCell>
              <TableCell align="right">{position.currentPrice.toFixed(5)}</TableCell>
              <TableCell 
                align="right"
                sx={{ 
                  color: position.unrealizedPnl >= 0 ? 'success.main' : 'error.main',
                  fontWeight: 'bold'
                }}
              >
                {position.unrealizedPnl.toFixed(2)}
              </TableCell>
              <TableCell align="right">{position.margin.toFixed(2)}</TableCell>
              <TableCell align="right">
                <Tooltip title="Close Position">
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      onClosePosition(position.id);
                    }}
                    aria-label="close position"
                    color="error"
                  >
                    <Close />
                  </IconButton>
                </Tooltip>
              </TableCell>
            </TableRow>
          ))}
          {positions.length === 0 && (
            <TableRow>
              <TableCell colSpan={8} align="center">
                No open positions
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default PositionsMonitor;
