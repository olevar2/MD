export interface UserSettings {
  userId: string;
  theme: 'light' | 'dark';
  notifications: {
    email: boolean;
    push: boolean;
    inApp: boolean;
    tradeExecution: boolean;
    priceAlerts: boolean;
    systemAlerts: boolean;
  };
  defaultChartTimeframe: string;
  defaultIndicators: string[];
  riskDefaults: {
    stopLossPercent?: number;
    takeProfitPercent?: number;
    maxPositionSizePercent?: number;
  };
  apiKeys: {
    brokerId: string;
    apiKeyMasked: string;
    addedDate: string;
  }[];
  twoFactorEnabled: boolean;
  backupCodes?: string[];
}
