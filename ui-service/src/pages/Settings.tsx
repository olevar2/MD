import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Box,
  Divider,
  IconButton,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  FormGroup
} from '@mui/material';
import { Delete as DeleteIcon } from '@mui/icons-material';
import IndicatorSelector from '../components/settings/IndicatorSelector';
import ApiKeyDialog from '../components/settings/ApiKeyDialog';
import Setup2FADialog from '../components/settings/Setup2FADialog';
import { useTheme } from '../hooks/useTheme';
import { useSettings } from '../hooks/useSettings';

const availableBrokers = [
  { id: 'OANDA', name: 'OANDA' },
  { id: 'FXCM', name: 'FXCM' },
  { id: 'IG', name: 'IG Markets' }
];

const Settings: React.FC = () => {
  const { currentTheme, toggleTheme } = useTheme();
  const {
    settings,
    isLoading,
    error,
    saveStatus,
    updateSettings,
    saveSettings
  } = useSettings();

  const [showApiKeyDialog, setShowApiKeyDialog] = useState(false);
  const [show2FADialog, setShow2FADialog] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleApiKeySubmit = async (data: { brokerId: string; apiKey: string; }) => {
    setIsSubmitting(true);
    try {
      await updateSettings({
        apiKeys: [
          ...(settings?.apiKeys || []),
          {
            brokerId: data.brokerId,
            apiKeyMasked: `********${data.apiKey.slice(-3)}`,
            addedDate: new Date().toISOString()
          }
        ]
      });
      setShowApiKeyDialog(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleApiKeyRevoke = async (keyToRemove: string) => {
    if (!settings) return;
    
    await updateSettings({
      apiKeys: settings.apiKeys.filter(key => key.apiKeyMasked !== keyToRemove)
    });
  };

  const handle2FAComplete = async (backupCodes: string[]) => {
    if (!settings) return;
    
    await updateSettings({
      twoFactorEnabled: true,
      backupCodes // In reality, these would be stored securely server-side
    });
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (!settings) {
    return (
      <Alert severity="error">
        Could not load settings. Please try again later.
      </Alert>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>

        <Grid container spacing={3}>
          {/* Appearance */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Appearance
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={currentTheme === 'dark'}
                    onChange={toggleTheme}
                  />
                }
                label="Dark Mode"
              />
            </Paper>
          </Grid>

          {/* Notifications */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Notifications
              </Typography>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.notifications.email}
                      onChange={(e) => updateSettings({
                        notifications: {
                          ...settings.notifications,
                          email: e.target.checked
                        }
                      })}
                    />
                  }
                  label="Email Notifications"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.notifications.push}
                      onChange={(e) => updateSettings({
                        notifications: {
                          ...settings.notifications,
                          push: e.target.checked
                        }
                      })}
                    />
                  }
                  label="Push Notifications"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.notifications.inApp}
                      onChange={(e) => updateSettings({
                        notifications: {
                          ...settings.notifications,
                          inApp: e.target.checked
                        }
                      })}
                    />
                  }
                  label="In-App Notifications"
                />
              </FormGroup>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Notification Types
                </Typography>
                <FormGroup>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.notifications.tradeExecution}
                        onChange={(e) => updateSettings({
                          notifications: {
                            ...settings.notifications,
                            tradeExecution: e.target.checked
                          }
                        })}
                      />
                    }
                    label="Trade Executions"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.notifications.priceAlerts}
                        onChange={(e) => updateSettings({
                          notifications: {
                            ...settings.notifications,
                            priceAlerts: e.target.checked
                          }
                        })}
                      />
                    }
                    label="Price Alerts"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.notifications.systemAlerts}
                        onChange={(e) => updateSettings({
                          notifications: {
                            ...settings.notifications,
                            systemAlerts: e.target.checked
                          }
                        })}
                      />
                    }
                    label="System Alerts"
                  />
                </FormGroup>
              </Box>
            </Paper>
          </Grid>

          {/* Chart Defaults */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Chart Defaults
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Default Timeframe"
                    value={settings.defaultChartTimeframe}
                    onChange={(e) => updateSettings({
                      defaultChartTimeframe: e.target.value
                    })}
                    helperText="e.g., H1, D1"
                  />
                </Grid>
                <Grid item xs={12}>
                  <IndicatorSelector
                    value={settings.defaultIndicators}
                    onChange={(newIndicators) => updateSettings({
                      defaultIndicators: newIndicators
                    })}
                  />
                </Grid>
              </Grid>
            </Paper>
          </Grid>

          {/* Risk Management */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Risk Management
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Default Stop Loss (%)"
                    value={settings.riskDefaults.stopLossPercent || ''}
                    onChange={(e) => updateSettings({
                      riskDefaults: {
                        ...settings.riskDefaults,
                        stopLossPercent: parseFloat(e.target.value)
                      }
                    })}
                    inputProps={{ step: 0.1 }}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Default Take Profit (%)"
                    value={settings.riskDefaults.takeProfitPercent || ''}
                    onChange={(e) => updateSettings({
                      riskDefaults: {
                        ...settings.riskDefaults,
                        takeProfitPercent: parseFloat(e.target.value)
                      }
                    })}
                    inputProps={{ step: 0.1 }}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Max Position Size (%)"
                    value={settings.riskDefaults.maxPositionSizePercent || ''}
                    onChange={(e) => updateSettings({
                      riskDefaults: {
                        ...settings.riskDefaults,
                        maxPositionSizePercent: parseFloat(e.target.value)
                      }
                    })}
                    inputProps={{ step: 0.5 }}
                  />
                </Grid>
              </Grid>
            </Paper>
          </Grid>

          {/* API Keys */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                API Keys
              </Typography>
              {settings.apiKeys.map((key, index) => (
                <Card key={index} sx={{ mb: 2 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <Typography variant="subtitle1">
                          {key.brokerId}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {key.apiKeyMasked} • Added {new Date(key.addedDate).toLocaleDateString()}
                        </Typography>
                      </div>
                      <IconButton
                        onClick={() => handleApiKeyRevoke(key.apiKeyMasked)}
                        color="error"
                        aria-label="revoke api key"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              ))}
              <Button
                variant="outlined"
                onClick={() => setShowApiKeyDialog(true)}
              >
                Add New API Key
              </Button>
            </Paper>
          </Grid>

          {/* Security */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Security
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.twoFactorEnabled}
                    onChange={(e) => {
                      if (!settings.twoFactorEnabled && e.target.checked) {
                        setShow2FADialog(true);
                      }
                    }}
                  />
                }
                label="Two-Factor Authentication (2FA)"
              />
              {settings.twoFactorEnabled && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  2FA is enabled for your account
                </Typography>
              )}
            </Paper>
          </Grid>
        </Grid>

        {/* Save Button */}
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
          {error && (
            <Alert severity="error" sx={{ flexGrow: 1 }}>
              {error}
            </Alert>
          )}
          <Button
            variant="contained"
            onClick={saveSettings}
            disabled={saveStatus === 'saving' || saveStatus === 'success'}
          >
            {saveStatus === 'saving' ? 'Saving...' : 
             saveStatus === 'success' ? 'Saved!' : 
             'Save Changes'}
          </Button>
        </Box>
      </Box>

      {/* Dialogs */}
      <ApiKeyDialog
        open={showApiKeyDialog}
        onClose={() => setShowApiKeyDialog(false)}
        onSubmit={handleApiKeySubmit}
        availableBrokers={availableBrokers}
        isSubmitting={isSubmitting}
      />

      <Setup2FADialog
        open={show2FADialog}
        onClose={() => setShow2FADialog(false)}
        onComplete={handle2FAComplete}
        isSubmitting={isSubmitting}
      />
    </Container>
  );
};

export default Settings;
