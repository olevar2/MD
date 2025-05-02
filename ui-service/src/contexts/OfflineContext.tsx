import React, { createContext, useContext, useState, useEffect } from 'react';

interface OfflineContextType {
  isOnline: boolean;
  isPendingSync: boolean;
  pendingActions: any[];
  addPendingAction: (action: any) => void;
  syncPendingActions: () => Promise<void>;
}

const OfflineContext = createContext<OfflineContextType | undefined>(undefined);

export const OfflineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isOnline, setIsOnline] = useState(true);
  const [isPendingSync, setIsPendingSync] = useState(false);
  const [pendingActions, setPendingActions] = useState<any[]>([]);

  useEffect(() => {
    // Initialize online status
    setIsOnline(navigator.onLine);

    // Add event listeners for online/offline status
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Try to sync any pending actions when coming back online
    const handleSync = async () => {
      if (navigator.onLine && pendingActions.length > 0) {
        await syncPendingActions();
      }
    };

    window.addEventListener('online', handleSync);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      window.removeEventListener('online', handleSync);
    };
  }, [pendingActions]);

  const addPendingAction = (action: any) => {
    setPendingActions((prev) => [...prev, action]);
  };

  const syncPendingActions = async () => {
    if (!navigator.onLine || pendingActions.length === 0) return;

    setIsPendingSync(true);
    try {
      // Process each pending action
      for (const action of pendingActions) {
        try {
          // Implement the sync logic based on action type
          switch (action.type) {
            case 'PLACE_ORDER':
              // Sync order with backend
              break;
            case 'UPDATE_STRATEGY':
              // Sync strategy updates
              break;
            case 'UPDATE_SETTINGS':
              // Sync user settings
              break;
          }
        } catch (error) {
          console.error('Failed to sync action:', action, error);
          // Implement retry logic or user notification
        }
      }
      // Clear successfully synced actions
      setPendingActions([]);
    } finally {
      setIsPendingSync(false);
    }
  };

  return (
    <OfflineContext.Provider
      value={{
        isOnline,
        isPendingSync,
        pendingActions,
        addPendingAction,
        syncPendingActions,
      }}
    >
      {children}
    </OfflineContext.Provider>
  );
};

export const useOffline = () => {
  const context = useContext(OfflineContext);
  if (context === undefined) {
    throw new Error('useOffline must be used within an OfflineProvider');
  }
  return context;
};
