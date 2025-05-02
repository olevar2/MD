import { openDB, IDBPDatabase } from 'idb';

interface SyncableData {
  id: string;
  type: string;
  data: any;
  timestamp: number;
  synced: boolean;
}

class DataSyncService {
  private dbName = 'forex-trading-platform-db';
  private dbVersion = 1;
  private db: IDBPDatabase | null = null;

  async initDatabase() {
    this.db = await openDB(this.dbName, this.dbVersion, {
      upgrade(db) {
        // Create stores for different data types
        db.createObjectStore('market-data', { keyPath: 'id' });
        db.createObjectStore('trading-signals', { keyPath: 'id' });
        db.createObjectStore('portfolio-updates', { keyPath: 'id' });
        db.createObjectStore('pending-actions', { keyPath: 'id' });
      },
    });
  }

  async storeData(storeName: string, data: SyncableData) {
    if (!this.db) await this.initDatabase();
    await this.db!.put(storeName, {
      ...data,
      timestamp: Date.now(),
      synced: navigator.onLine,
    });
  }

  async getData(storeName: string, id: string) {
    if (!this.db) await this.initDatabase();
    return await this.db!.get(storeName, id);
  }

  async getAllUnsyncedData(storeName: string) {
    if (!this.db) await this.initDatabase();
    const tx = this.db!.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const items = await store.getAll();
    return items.filter(item => !item.synced);
  }

  async markAsSynced(storeName: string, id: string) {
    if (!this.db) await this.initDatabase();
    const item = await this.getData(storeName, id);
    if (item) {
      await this.storeData(storeName, { ...item, synced: true });
    }
  }

  // Real-time data handling
  async handleRealtimeUpdate(update: any) {
    const { type, data } = update;
    
    // Store the update locally
    await this.storeData('market-data', {
      id: `${type}-${Date.now()}`,
      type,
      data,
      timestamp: Date.now(),
      synced: true,
    });

    // Broadcast the update to relevant components
    this.broadcastUpdate(type, data);
  }

  private broadcastUpdate(type: string, data: any) {
    const event = new CustomEvent('forex-data-update', {
      detail: { type, data },
    });
    window.dispatchEvent(event);
  }

  // Background sync registration
  async registerBackgroundSync() {
    if ('serviceWorker' in navigator && 'sync' in navigator.serviceWorker) {
      try {
        const registration = await navigator.serviceWorker.ready;
        await registration.sync.register('forex-data-sync');
      } catch (err) {
        console.error('Background sync registration failed:', err);
      }
    }
  }

  // Push notification subscription
  async subscribeToPushNotifications() {
    if ('Notification' in window && 'serviceWorker' in navigator) {
      try {
        const permission = await Notification.requestPermission();
        if (permission === 'granted') {
          const registration = await navigator.serviceWorker.ready;
          const subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY,
          });
          
          // Send subscription to server
          await fetch('/api/push-subscription', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(subscription),
          });
        }
      } catch (err) {
        console.error('Push notification subscription failed:', err);
      }
    }
  }
}

export const dataSyncService = new DataSyncService();
