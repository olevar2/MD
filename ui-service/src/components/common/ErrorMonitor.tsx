import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  List, 
  ListItem, 
  ListItemText, 
  Chip, 
  IconButton, 
  Collapse, 
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import { 
  ExpandMore as ExpandMoreIcon, 
  ExpandLess as ExpandLessIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { getStoredErrors, clearStoredErrors, ErrorSeverity } from '../../utils/errorHandler';

/**
 * Error Monitor component for displaying and managing errors
 * This is primarily for development and debugging purposes
 */
const ErrorMonitor: React.FC = () => {
  const [errors, setErrors] = useState<any[]>([]);
  const [expandedError, setExpandedError] = useState<number | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState<boolean>(false);
  const [selectedError, setSelectedError] = useState<any | null>(null);
  
  // Load errors from session storage
  useEffect(() => {
    setErrors(getStoredErrors());
    
    // Refresh errors every 5 seconds
    const interval = setInterval(() => {
      setErrors(getStoredErrors());
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Handle error expansion
  const handleToggleExpand = (index: number) => {
    setExpandedError(expandedError === index ? null : index);
  };
  
  // Handle error details dialog
  const handleOpenDetails = (error: any) => {
    setSelectedError(error);
    setIsDialogOpen(true);
  };
  
  // Handle clear all errors
  const handleClearErrors = () => {
    clearStoredErrors();
    setErrors([]);
  };
  
  // Get severity color
  const getSeverityColor = (severity: ErrorSeverity) => {
    switch (severity) {
      case ErrorSeverity.FATAL:
      case ErrorSeverity.ERROR:
        return 'error';
      case ErrorSeverity.WARNING:
        return 'warning';
      case ErrorSeverity.INFO:
        return 'info';
      case ErrorSeverity.DEBUG:
        return 'default';
      default:
        return 'default';
    }
  };
  
  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    } catch (e) {
      return timestamp;
    }
  };
  
  if (errors.length === 0) {
    return (
      <Paper sx={{ p: 2, m: 2 }}>
        <Typography variant="body1">No errors recorded</Typography>
      </Paper>
    );
  }
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Error Monitor ({errors.length})</Typography>
        <Box>
          <Button 
            startIcon={<RefreshIcon />}
            onClick={() => setErrors(getStoredErrors())}
            size="small"
            sx={{ mr: 1 }}
          >
            Refresh
          </Button>
          <Button 
            startIcon={<DeleteIcon />}
            onClick={handleClearErrors}
            color="error"
            size="small"
          >
            Clear All
          </Button>
        </Box>
      </Box>
      
      <Paper elevation={2}>
        <List sx={{ width: '100%' }}>
          {errors.map((error, index) => (
            <React.Fragment key={index}>
              <ListItem
                secondaryAction={
                  <IconButton edge="end" onClick={() => handleToggleExpand(index)}>
                    {expandedError === index ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </IconButton>
                }
                sx={{ 
                  borderBottom: '1px solid',
                  borderColor: 'divider',
                  bgcolor: expandedError === index ? 'action.hover' : 'inherit'
                }}
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip 
                        label={error.errorType} 
                        size="small" 
                        color={getSeverityColor(error.severity) as any}
                      />
                      <Typography variant="body2" component="span">
                        {error.message}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Typography variant="caption" color="text.secondary">
                      {formatTimestamp(error.timestamp)}
                    </Typography>
                  }
                />
              </ListItem>
              
              <Collapse in={expandedError === index} timeout="auto" unmountOnExit>
                <Box sx={{ p: 2, bgcolor: 'action.hover' }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Error Type: {error.errorType}
                  </Typography>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Severity: {error.severity}
                  </Typography>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Timestamp: {error.timestamp}
                  </Typography>
                  
                  {error.stack && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Stack Trace:
                      </Typography>
                      <Box 
                        component="pre" 
                        sx={{ 
                          p: 1, 
                          bgcolor: 'background.paper', 
                          borderRadius: 1, 
                          fontSize: '0.75rem',
                          maxHeight: '200px',
                          overflow: 'auto'
                        }}
                      >
                        {error.stack}
                      </Box>
                    </Box>
                  )}
                  
                  <Button 
                    variant="outlined" 
                    size="small" 
                    sx={{ mt: 2 }}
                    onClick={() => handleOpenDetails(error)}
                  >
                    View Full Details
                  </Button>
                </Box>
              </Collapse>
            </React.Fragment>
          ))}
        </List>
      </Paper>
      
      {/* Error Details Dialog */}
      <Dialog
        open={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Error Details
        </DialogTitle>
        
        <DialogContent dividers>
          {selectedError && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Message: {selectedError.message}
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>
                Type: {selectedError.errorType}
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>
                Severity: {selectedError.severity}
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>
                Timestamp: {selectedError.timestamp}
              </Typography>
              
              {selectedError.stack && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Stack Trace:
                  </Typography>
                  <Box 
                    component="pre" 
                    sx={{ 
                      p: 1, 
                      bgcolor: 'background.paper', 
                      borderRadius: 1, 
                      fontSize: '0.75rem',
                      maxHeight: '200px',
                      overflow: 'auto'
                    }}
                  >
                    {selectedError.stack}
                  </Box>
                </Box>
              )}
              
              {selectedError.context && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Context:
                  </Typography>
                  <Box 
                    component="pre" 
                    sx={{ 
                      p: 1, 
                      bgcolor: 'background.paper', 
                      borderRadius: 1, 
                      fontSize: '0.75rem',
                      maxHeight: '200px',
                      overflow: 'auto'
                    }}
                  >
                    {JSON.stringify(selectedError.context, null, 2)}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setIsDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ErrorMonitor;
