import React from 'react';
import { Box, IconButton, SpeedDial, SpeedDialAction } from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import TouchAppIcon from '@mui/icons-material/TouchApp';
import TimelineIcon from '@mui/icons-material/Timeline';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import CandlestickChartIcon from '@mui/icons-material/CandlestickChart';

interface TouchControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onChartTypeChange: (type: 'candlestick' | 'line') => void;
  onResetZoom: () => void;
}

const TouchControls: React.FC<TouchControlsProps> = ({
  onZoomIn,
  onZoomOut,
  onChartTypeChange,
  onResetZoom,
}) => {
  const [isOpen, setIsOpen] = React.useState(false);

  const actions = [
    { icon: <ZoomInIcon />, name: 'Zoom In', onClick: onZoomIn },
    { icon: <ZoomOutIcon />, name: 'Zoom Out', onClick: onZoomOut },
    { icon: <CandlestickChartIcon />, name: 'Candlestick', onClick: () => onChartTypeChange('candlestick') },
    { icon: <TimelineIcon />, name: 'Line', onClick: () => onChartTypeChange('line') },
    { icon: <ShowChartIcon />, name: 'Reset', onClick: onResetZoom },
  ];

  return (
    <Box sx={{ position: 'absolute', bottom: 16, right: 16, zIndex: 1000 }}>
      <SpeedDial
        ariaLabel="Chart Controls"
        icon={<TouchAppIcon />}
        onClose={() => setIsOpen(false)}
        onOpen={() => setIsOpen(true)}
        open={isOpen}
        direction="up"
        sx={{
          '& .MuiFab-primary': {
            width: 48,
            height: 48,
            backgroundColor: 'primary.main',
          },
        }}
      >
        {actions.map((action) => (
          <SpeedDialAction
            key={action.name}
            icon={action.icon}
            tooltipTitle={action.name}
            onClick={() => {
              action.onClick();
              setIsOpen(false);
            }}
          />
        ))}
      </SpeedDial>
    </Box>
  );
};

const useTouchGestures = (containerRef: React.RefObject<HTMLDivElement>) => {
  const [gesture, setGesture] = React.useState({
    isDragging: false,
    startX: 0,
    startY: 0,
    lastDistance: 0,
  });

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        // Pinch gesture
        const distance = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        setGesture(prev => ({ ...prev, lastDistance: distance }));
      } else {
        // Single touch drag
        setGesture(prev => ({
          ...prev,
          isDragging: true,
          startX: e.touches[0].clientX,
          startY: e.touches[0].clientY,
        }));
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        // Handle pinch zoom
        const distance = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        const delta = distance - gesture.lastDistance;
        // Emit zoom event based on delta
        const zoomEvent = new CustomEvent('chart-zoom', { detail: { delta } });
        container.dispatchEvent(zoomEvent);
        setGesture(prev => ({ ...prev, lastDistance: distance }));
      } else if (gesture.isDragging) {
        // Handle drag
        const deltaX = e.touches[0].clientX - gesture.startX;
        const deltaY = e.touches[0].clientY - gesture.startY;
        // Emit pan event
        const panEvent = new CustomEvent('chart-pan', { detail: { deltaX, deltaY } });
        container.dispatchEvent(panEvent);
      }
    };

    const handleTouchEnd = () => {
      setGesture(prev => ({ ...prev, isDragging: false, lastDistance: 0 }));
    };

    container.addEventListener('touchstart', handleTouchStart);
    container.addEventListener('touchmove', handleTouchMove);
    container.addEventListener('touchend', handleTouchEnd);

    return () => {
      container.removeEventListener('touchstart', handleTouchStart);
      container.removeEventListener('touchmove', handleTouchMove);
      container.removeEventListener('touchend', handleTouchEnd);
    };
  }, [containerRef, gesture]);

  return gesture;
};

export { TouchControls, useTouchGestures };
