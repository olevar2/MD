import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Box
} from '@mui/material';

export interface ApiKeyData {
  brokerId: string;
  apiKey: string;
  apiSecret?: string;
  label?: string;
}

interface ApiKeyDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: ApiKeyData) => Promise<void>;
  availableBrokers: { id: string; name: string; }[];
  isSubmitting?: boolean;
}

const ApiKeyDialog: React.FC<ApiKeyDialogProps> = ({
  open,
  onClose,
  onSubmit,
  availableBrokers,
  isSubmitting = false
}) => {
  const [formData, setFormData] = useState<ApiKeyData>({
    brokerId: '',
    apiKey: '',
    apiSecret: '',
    label: ''
  });
  const [error, setError] = useState<string>('');

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError('');
  };

  const handleBrokerChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setFormData(prev => ({
      ...prev,
      brokerId: event.target.value as string
    }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (!formData.brokerId) {
      setError('Please select a broker');
      return;
    }
    
    if (!formData.apiKey) {
      setError('API Key is required');
      return;
    }

    try {
      await onSubmit(formData);
      onClose();
      setFormData({
        brokerId: '',
        apiKey: '',
        apiSecret: '',
        label: ''
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save API key');
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Add New API Key</DialogTitle>
        <DialogContent>
          {error && (
            <Box sx={{ mb: 2 }}>
              <Alert severity="error">{error}</Alert>
            </Box>
          )}

          <FormControl fullWidth sx={{ mb: 2, mt: 1 }}>
            <InputLabel id="broker-select-label">Broker</InputLabel>
            <Select
              labelId="broker-select-label"
              value={formData.brokerId}
              label="Broker"
              onChange={handleBrokerChange}
              required
            >
              {availableBrokers.map(broker => (
                <MenuItem key={broker.id} value={broker.id}>
                  {broker.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            fullWidth
            label="API Key"
            name="apiKey"
            value={formData.apiKey}
            onChange={handleInputChange}
            required
            sx={{ mb: 2 }}
            type="password"
          />

          {formData.brokerId && availableBrokers.find(b => b.id === formData.brokerId)?.name === 'OANDA' && (
            <TextField
              fullWidth
              label="API Secret"
              name="apiSecret"
              value={formData.apiSecret}
              onChange={handleInputChange}
              required
              sx={{ mb: 2 }}
              type="password"
            />
          )}

          <TextField
            fullWidth
            label="Label (Optional)"
            name="label"
            value={formData.label}
            onChange={handleInputChange}
            placeholder="e.g., Demo Account"
            sx={{ mb: 2 }}
          />
        </DialogContent>
        
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button 
            type="submit" 
            variant="contained" 
            color="primary"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Adding...' : 'Add API Key'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default ApiKeyDialog;
