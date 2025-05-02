import { useState, useEffect, useCallback } from 'react';
import { UserSettings } from '../types/settings';

interface UseSettingsResult {
  settings: UserSettings | null;
  isLoading: boolean;
  error: string | null;
  saveStatus: 'idle' | 'saving' | 'success' | 'error';
  updateSettings: (updates: Partial<UserSettings>) => Promise<void>;
  saveSettings: () => Promise<void>;
}

const mockUserSettings: UserSettings = {
  userId: 'user123',
  theme: 'dark',
  notifications: {
    email: true,
    push: false,
    inApp: true,
    tradeExecution: true,
    priceAlerts: false,
    systemAlerts: true,
  },
  defaultChartTimeframe: 'H1',
  defaultIndicators: ['EMA-50', 'MACD'],
  riskDefaults: {
    stopLossPercent: 1.5,
    maxPositionSizePercent: 2,
  },
  apiKeys: [
    {
      brokerId: 'OANDA',
      apiKeyMasked: '********abc',
      addedDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
    },
  ],
  twoFactorEnabled: false,
};

export const useSettings = (): UseSettingsResult => {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        // In a real implementation, this would be an API call
        // const response = await api.get('/user/settings');
        // const data = await response.json();
        await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API call
        setSettings(mockUserSettings);
        setError(null);
      } catch (err) {
        setError('Failed to load settings');
        console.error('Settings fetch error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSettings();
  }, []);

  const updateSettings = useCallback(async (updates: Partial<UserSettings>) => {
    if (!settings) return;

    try {
      setSettings(prevSettings => ({
        ...prevSettings!,
        ...updates
      }));
      setSaveStatus('idle');
    } catch (err) {
      setError('Failed to update settings');
      console.error('Settings update error:', err);
    }
  }, [settings]);

  const saveSettings = useCallback(async () => {
    if (!settings) return;

    setSaveStatus('saving');
    try {
      // In a real implementation, this would be an API call
      // await api.put('/user/settings', settings);
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 3000);
      setError(null);
    } catch (err) {
      setSaveStatus('error');
      setError('Failed to save settings');
      console.error('Settings save error:', err);
    }
  }, [settings]);

  return {
    settings,
    isLoading,
    error,
    saveStatus,
    updateSettings,
    saveSettings
  };
};
