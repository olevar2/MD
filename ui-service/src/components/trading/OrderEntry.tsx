import React, { useState } from 'react';
import {
  Card,
  CardContent,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  SelectChangeEvent,
  Typography,
  Switch,
  FormControlLabel
} from '@mui/material';

export interface OrderFormData {
  symbol: string;
  type: 'MARKET' | 'LIMIT' | 'STOP';
  side: 'BUY' | 'SELL';
  quantity: number;
  price?: number;
  stopPrice?: number;
  takeProfit?: number;
  stopLoss?: number;
}

interface OrderEntryProps {
  availableSymbols: string[];
  currentPrice?: number;
  onSubmit: (order: OrderFormData) => void;
  isSubmitting?: boolean;
}

const OrderEntry: React.FC<OrderEntryProps> = ({
  availableSymbols,
  currentPrice,
  onSubmit,
  isSubmitting = false
}) => {
  const [formData, setFormData] = useState<OrderFormData>({
    symbol: availableSymbols[0] || '',
    type: 'MARKET',
    side: 'BUY',
    quantity: 0
  });

  const [advancedMode, setAdvancedMode] = useState(false);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.includes('quantity') || name.includes('price') ? 
        parseFloat(value) || 0 : value
    }));
  };

  const handleSelectChange = (event: SelectChangeEvent) => {
    const { name, value } = event.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit(formData);
  };

  return (
    <Card>
      <CardContent>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={advancedMode}
                    onChange={(e) => setAdvancedMode(e.target.checked)}
                  />
                }
                label="Advanced Mode"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Symbol</InputLabel>
                <Select
                  name="symbol"
                  value={formData.symbol}
                  onChange={handleSelectChange}
                  required
                >
                  {availableSymbols.map(symbol => (
                    <MenuItem key={symbol} value={symbol}>
                      {symbol}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Type</InputLabel>
                <Select
                  name="type"
                  value={formData.type}
                  onChange={handleSelectChange}
                >
                  <MenuItem value="MARKET">Market</MenuItem>
                  <MenuItem value="LIMIT">Limit</MenuItem>
                  <MenuItem value="STOP">Stop</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Side</InputLabel>
                <Select
                  name="side"
                  value={formData.side}
                  onChange={handleSelectChange}
                >
                  <MenuItem value="BUY">Buy</MenuItem>
                  <MenuItem value="SELL">Sell</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Quantity"
                name="quantity"
                type="number"
                value={formData.quantity}
                onChange={handleInputChange}
                required
              />
            </Grid>

            {formData.type !== 'MARKET' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label={formData.type === 'LIMIT' ? 'Limit Price' : 'Stop Price'}
                  name={formData.type === 'LIMIT' ? 'price' : 'stopPrice'}
                  type="number"
                  value={formData.price || formData.stopPrice || ''}
                  onChange={handleInputChange}
                  required
                />
              </Grid>
            )}

            {advancedMode && (
              <>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Take Profit"
                    name="takeProfit"
                    type="number"
                    value={formData.takeProfit || ''}
                    onChange={handleInputChange}
                  />
                </Grid>

                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Stop Loss"
                    name="stopLoss"
                    type="number"
                    value={formData.stopLoss || ''}
                    onChange={handleInputChange}
                  />
                </Grid>
              </>
            )}

            {currentPrice && (
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="textSecondary">
                  Current Price: {currentPrice.toFixed(5)}
                </Typography>
              </Grid>
            )}

            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                fullWidth
                color={formData.side === 'BUY' ? 'success' : 'error'}
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Submitting...' : `${formData.side} ${formData.symbol}`}
              </Button>
            </Grid>
          </Grid>
        </form>
      </CardContent>
    </Card>
  );
};

export default OrderEntry;
