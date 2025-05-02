import React, { useState } from 'react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import { Box, Paper, IconButton, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';
import SettingsIcon from '@mui/icons-material/Settings';
import CloseIcon from '@mui/icons-material/Close';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';

const ResponsiveGridLayout = WidthProvider(Responsive);

const WidgetContainer = styled(Paper)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
}));

const WidgetHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: theme.spacing(0.5, 1),
  borderBottom: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.paper,
  cursor: 'move',
}));

const WidgetContent = styled(Box)({
  flexGrow: 1,
  overflow: 'auto',
  position: 'relative',
});

interface CustomizableLayoutProps {
  initialLayout: Array<{
    i: string;
    x: number;
    y: number;
    w: number;
    h: number;
    minW?: number;
    minH?: number;
    maxW?: number;
    maxH?: number;
  }>;
  components: {
    [key: string]: React.ReactNode;
  };
  onLayoutChange?: (layout: any) => void;
}

export const CustomizableLayout: React.FC<CustomizableLayoutProps> = ({
  initialLayout,
  components,
  onLayoutChange,
}) => {
  const [layout, setLayout] = useState(initialLayout);
  const [expandedWidget, setExpandedWidget] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [visibleWidgets, setVisibleWidgets] = useState<string[]>(
    initialLayout.map((item) => item.i)
  );

  const handleLayoutChange = (newLayout: any) => {
    setLayout(newLayout);
    if (onLayoutChange) {
      onLayoutChange(newLayout);
    }
  };

  const toggleWidget = (widgetId: string) => {
    if (visibleWidgets.includes(widgetId)) {
      setVisibleWidgets(visibleWidgets.filter((id) => id !== widgetId));
    } else {
      setVisibleWidgets([...visibleWidgets, widgetId]);
    }
  };

  const toggleExpand = (widgetId: string) => {
    setExpandedWidget(expandedWidget === widgetId ? null : widgetId);
  };

  const getWidgetTitle = (widgetId: string): string => {
    switch (widgetId) {
      case 'chart':
        return 'Trading Chart';
      case 'overview':
        return 'Market Overview';
      case 'order':
        return 'Order Panel';
      case 'positions':
        return 'Positions';
      case 'alerts':
        return 'Alerts';
      case 'history':
        return 'Trade History';
      case 'performance':
        return 'Performance Metrics';
      default:
        return widgetId.charAt(0).toUpperCase() + widgetId.slice(1);
    }
  };

  if (expandedWidget) {
    const widgetId = expandedWidget;
    return (
      <Box sx={{ width: '100%', height: 'calc(100vh - 150px)' }}>
        <WidgetContainer>
          <WidgetHeader>
            <Typography variant="subtitle1">{getWidgetTitle(widgetId)}</Typography>
            <IconButton size="small" onClick={() => toggleExpand(widgetId)}>
              <FullscreenExitIcon fontSize="small" />
            </IconButton>
          </WidgetHeader>
          <WidgetContent>{components[widgetId]}</WidgetContent>
        </WidgetContainer>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end' }}>
        <IconButton 
          onClick={() => setEditMode(!editMode)}
          color={editMode ? 'primary' : 'default'}
        >
          <SettingsIcon />
        </IconButton>
      </Box>

      <ResponsiveGridLayout
        className="layout"
        layouts={{ lg: layout }}
        breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
        cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
        rowHeight={100}
        isDraggable={editMode}
        isResizable={editMode}
        onLayoutChange={(layout) => handleLayoutChange(layout)}
        margin={[10, 10]}
      >
        {layout
          .filter((item) => visibleWidgets.includes(item.i))
          .map((item) => (
            <div key={item.i}>
              <WidgetContainer>
                <WidgetHeader className="dragHandle">
                  <Typography variant="subtitle1">{getWidgetTitle(item.i)}</Typography>
                  <Box>
                    {editMode && (
                      <IconButton size="small" onClick={() => toggleWidget(item.i)}>
                        <CloseIcon fontSize="small" />
                      </IconButton>
                    )}
                    <IconButton size="small" onClick={() => toggleExpand(item.i)}>
                      <FullscreenIcon fontSize="small" />
                    </IconButton>
                  </Box>
                </WidgetHeader>
                <WidgetContent>{components[item.i]}</WidgetContent>
              </WidgetContainer>
            </div>
          ))}
      </ResponsiveGridLayout>

      {editMode && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Hidden Widgets
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {layout
              .filter((item) => !visibleWidgets.includes(item.i))
              .map((item) => (
                <Paper
                  key={item.i}
                  sx={{ p: 1, cursor: 'pointer' }}
                  onClick={() => toggleWidget(item.i)}
                >
                  {getWidgetTitle(item.i)}
                </Paper>
              ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default CustomizableLayout;
