import React from 'react';
import { 
  ThemeProvider as MuiThemeProvider, 
  createTheme, 
  Theme, 
  CssBaseline,
  PaletteMode
} from '@mui/material';

// Create type-safe theme extension
declare module '@mui/material/styles' {
  interface Palette {
    tradingProfit: Palette['primary'];
    tradingLoss: Palette['error'];
    marketBuy: Palette['success'];
    marketSell: Palette['error'];
  }
  
  interface PaletteOptions {
    tradingProfit?: PaletteOptions['primary'];
    tradingLoss?: PaletteOptions['error'];
    marketBuy?: PaletteOptions['success'];
    marketSell?: PaletteOptions['error'];
  }
}

interface ThemeProviderProps {
  children: React.ReactNode;
  mode?: PaletteMode;
}

const lightTheme = {
  palette: {
    mode: 'light' as PaletteMode,
    primary: {
      main: '#2c6ecb',
      light: '#5a95f5',
      dark: '#1a4b8f',
    },
    secondary: {
      main: '#6c49b8',
      light: '#9772e3',
      dark: '#4c3381',
    },
    tradingProfit: {
      main: '#16a34a',
      light: '#4ade80',
      dark: '#15803d',
    },
    tradingLoss: {
      main: '#dc2626',
      light: '#ef4444',
      dark: '#b91c1c',
    },
    marketBuy: {
      main: '#16a34a',
      light: '#4ade80',
      dark: '#15803d',
    },
    marketSell: {
      main: '#dc2626',
      light: '#ef4444',
      dark: '#b91c1c',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        },
      },
    },
  },
};

const darkTheme = {
  ...lightTheme,
  palette: {
    ...lightTheme.palette,
    mode: 'dark' as PaletteMode,
    primary: {
      main: '#3b82f6',
      light: '#60a5fa',
      dark: '#2563eb',
    },
    background: {
      default: '#111827',
      paper: '#1f2937',
    }
  },
};

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children, mode = 'light' }) => {
  const theme = React.useMemo(() => {
    return createTheme(mode === 'light' ? lightTheme : darkTheme);
  }, [mode]);

  return (
    <MuiThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </MuiThemeProvider>
  );
};

export default ThemeProvider;
