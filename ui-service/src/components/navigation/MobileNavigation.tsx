import React from 'react';
import {
  BottomNavigation,
  BottomNavigationAction,
  Paper,
  SwipeableDrawer,
  useMediaQuery,
  Theme,
} from '@mui/material';
import { useRouter } from 'next/router';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import MonitorIcon from '@mui/icons-material/Monitor';
import PsychologyIcon from '@mui/icons-material/Psychology';

interface MobileNavigationProps {
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
}

const MobileNavigation: React.FC<MobileNavigationProps> = ({ open, onOpen, onClose }) => {
  const router = useRouter();
  const isMobile = useMediaQuery((theme: Theme) => theme.breakpoints.down('sm'));

  const navigationItems = [
    { label: 'Trading', icon: <ShowChartIcon />, path: '/dashboard' },
    { label: 'Portfolio', icon: <AccountBalanceIcon />, path: '/portfolio' },
    { label: 'Monitor', icon: <MonitorIcon />, path: '/monitor' },
    { label: 'Strategy', icon: <PsychologyIcon />, path: '/strategy' },
  ];

  const handleNavigation = (path: string) => {
    router.push(path);
  };

  if (!isMobile) return null;

  return (
    <>
      <SwipeableDrawer
        anchor="bottom"
        open={open}
        onOpen={onOpen}
        onClose={onClose}
        disableDiscovery
      >
        {/* Add mobile-specific menu items here */}
      </SwipeableDrawer>

      <Paper
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
        }}
        elevation={3}
      >
        <BottomNavigation
          value={router.pathname}
          onChange={(_, newValue) => handleNavigation(newValue)}
          showLabels
        >
          {navigationItems.map((item) => (
            <BottomNavigationAction
              key={item.path}
              label={item.label}
              icon={item.icon}
              value={item.path}
            />
          ))}
        </BottomNavigation>
      </Paper>
    </>
  );
};

export default MobileNavigation;
