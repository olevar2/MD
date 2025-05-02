import { useState, useEffect, useCallback } from 'react';

interface WebSocketHookResult {
  data: string | null;
  isConnected: boolean;
  error: Error | null;
  connect: () => void;
  disconnect: () => void;
  send: (data: string) => void;
}

const useWebSocket = (url: string): WebSocketHookResult => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [data, setData] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      ws.onerror = (event) => {
        setError(new Error('WebSocket error occurred'));
        console.error('WebSocket error:', event);
      };

      ws.onmessage = (event) => {
        setData(event.data);
      };

      setSocket(ws);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to create WebSocket'));
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (socket) {
      socket.close();
      setSocket(null);
      setIsConnected(false);
    }
  }, [socket]);

  const send = useCallback((data: string) => {
    if (socket && isConnected) {
      socket.send(data);
    } else {
      setError(new Error('WebSocket is not connected'));
    }
  }, [socket, isConnected]);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [url]);

  return {
    data,
    isConnected,
    error,
    connect,
    disconnect,
    send
  };
};

export default useWebSocket;
