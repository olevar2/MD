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
  Tooltip
} from '@mui/material';
import { Edit, Delete } from '@mui/icons-material';

export interface Order {
  id: string;
  symbol: string;
  type: 'MARKET' | 'LIMIT' | 'STOP';
  side: 'BUY' | 'SELL';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  timestamp: number;
}

interface OrdersListProps {
  orders: Order[];
  onCancelOrder: (orderId: string) => void;
  onModifyOrder: (orderId: string) => void;
}

const OrdersList: React.FC<OrdersListProps> = ({
  orders,
  onCancelOrder,
  onModifyOrder
}) => {
  return (
    <TableContainer component={Paper}>
      <Table size="small" aria-label="active orders table">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Type</TableCell>
            <TableCell align="right">Side</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Price</TableCell>
            <TableCell align="right">Status</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {orders.map((order) => (
            <TableRow
              key={order.id}
              sx={{
                '&:last-child td, &:last-child th': { border: 0 },
                backgroundColor: 
                  order.status === 'REJECTED' ? 'error.light' :
                  order.status === 'FILLED' ? 'success.light' : 'inherit'
              }}
            >
              <TableCell component="th" scope="row">
                {order.symbol}
              </TableCell>
              <TableCell align="right">{order.type}</TableCell>
              <TableCell 
                align="right"
                sx={{ 
                  color: order.side === 'BUY' ? 'success.main' : 'error.main' 
                }}
              >
                {order.side}
              </TableCell>
              <TableCell align="right">{order.quantity}</TableCell>
              <TableCell align="right">
                {order.price ? order.price.toFixed(5) : '-'}
                {order.stopPrice ? ` / ${order.stopPrice.toFixed(5)}` : ''}
              </TableCell>
              <TableCell align="right">{order.status}</TableCell>
              <TableCell align="right">
                {order.status === 'PENDING' && (
                  <>
                    <Tooltip title="Modify Order">
                      <IconButton
                        size="small"
                        onClick={() => onModifyOrder(order.id)}
                        aria-label="modify order"
                      >
                        <Edit />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Cancel Order">
                      <IconButton
                        size="small"
                        onClick={() => onCancelOrder(order.id)}
                        aria-label="cancel order"
                        color="error"
                      >
                        <Delete />
                      </IconButton>
                    </Tooltip>
                  </>
                )}
              </TableCell>
            </TableRow>
          ))}
          {orders.length === 0 && (
            <TableRow>
              <TableCell colSpan={7} align="center">
                No active orders
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default OrdersList;
