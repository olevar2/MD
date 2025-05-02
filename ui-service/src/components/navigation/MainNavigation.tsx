import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { 
  Box, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Divider, 
  Typography, 
  Avatar
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  Timeline as ChartIcon,
  Assessment as AssessmentIcon,
  MonitorHeart as MonitorIcon,
  Paid as TradingIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon
} from '@mui/icons-material';

const drawerWidth = 260;

const LogoContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(2),
  paddingLeft: theme.spacing(3),
}));

const NavLinkStyled = styled(NavLink)(({ theme }) => ({
  textDecoration: 'none',
  color: theme.palette.text.primary,
  '&.active .MuiListItem-root': {
    backgroundColor: theme.palette.action.selected,
    borderRight: `3px solid ${theme.palette.primary.main}`,
  },
  '&:hover .MuiListItem-root': {
    backgroundColor: theme.palette.action.hover,
  }
}));

interface MainNavigationProps {
  open: boolean;
  onClose: () => void;
}

export const MainNavigation: React.FC<MainNavigationProps> = ({ open, onClose }) => {
  const location = useLocation();

  const navigationItems = [
    { path: '/dashboard', text: 'Trading Dashboard', icon: <DashboardIcon /> },
    { path: '/analysis', text: 'Analysis Tools', icon: <AnalyticsIcon /> },
    { path: '/performance', text: 'Performance', icon: <AssessmentIcon /> },
    { path: '/charts', text: 'Charts', icon: <ChartIcon /> },
    { path: '/monitoring', text: 'Monitoring', icon: <MonitorIcon /> },
    { path: '/trading', text: 'Trading', icon: <TradingIcon /> },
    { path: '/settings', text: 'Settings', icon: <SettingsIcon /> },
  ];

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      onClose={onClose}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
    >
      <LogoContainer>
        <Avatar
          sx={{ width: 40, height: 40, marginRight: 2, backgroundColor: 'primary.main' }}
        >
          FX
        </Avatar>
        <Typography variant="h6" component="div">
          ForexTrader Pro
        </Typography>
      </LogoContainer>
      
      <Divider />
      
      <Box sx={{ overflow: 'auto', flexGrow: 1 }}>
        <List>
          {navigationItems.map((item) => (
            <NavLinkStyled key={item.path} to={item.path} className={location.pathname.startsWith(item.path) ? 'active' : ''}>
              <ListItem button sx={{ py: 1.5, px: 3 }}>
                <ListItemIcon>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            </NavLinkStyled>
          ))}
        </List>
      </Box>
      
      <Divider />
      
      <List>
        <ListItem button sx={{ py: 1.5, px: 3 }}>
          <ListItemIcon>
            <LogoutIcon />
          </ListItemIcon>
          <ListItemText primary="Logout" />
        </ListItem>
      </List>
    </Drawer>
  );
};

export default MainNavigation;
