import React from 'react';
import { Box, AppBar, Toolbar, Drawer, List, Typography, IconButton } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChartIcon from '@mui/icons-material/ShowChart';
import PortfolioIcon from '@mui/icons-material/AccountBalance';
import MonitorIcon from '@mui/icons-material/Monitor';
import StrategyIcon from '@mui/icons-material/Psychology';
import { styled } from '@mui/material/styles';
import { useRouter } from 'next/router';

const DRAWER_WIDTH = 240;

const MainContent = styled('main', {
  shouldForwardProp: (prop) => prop !== 'open',
})<{ open?: boolean }>(({ theme, open }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  marginLeft: `-${DRAWER_WIDTH}px`,
  ...(open && {
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: 0,
  }),
}));

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [open, setOpen] = React.useState(true);
  const router = useRouter();

  const menuItems = [
    { text: 'Trading Dashboard', icon: <ChartIcon />, path: '/dashboard' },
    { text: 'Portfolio Management', icon: <PortfolioIcon />, path: '/portfolio' },
    { text: 'System Monitor', icon: <MonitorIcon />, path: '/monitor' },
    { text: 'Strategy Builder', icon: <StrategyIcon />, path: '/strategy' },
  ];

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="toggle drawer"
            onClick={() => setOpen(!open)}
            edge="start"
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            Forex Trading Platform
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="persistent"
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
          },
        }}
        anchor="left"
        open={open}
      >
        <Toolbar />
        <List>
          {menuItems.map((item) => (
            <IconButton
              key={item.text}
              onClick={() => router.push(item.path)}
              sx={{
                width: '100%',
                justifyContent: 'flex-start',
                padding: '12px',
                color: router.pathname === item.path ? 'primary.main' : 'inherit',
              }}
            >
              {item.icon}
              <Typography sx={{ ml: 2 }}>{item.text}</Typography>
            </IconButton>
          ))}
        </List>
      </Drawer>
      <MainContent open={open}>
        <Toolbar />
        {children}
      </MainContent>
    </Box>
  );
};

export default DashboardLayout;
