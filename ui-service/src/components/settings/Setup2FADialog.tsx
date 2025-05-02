import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  TextField,
  Paper
} from '@mui/material';
import QRCode from 'qrcode.react';

interface Setup2FADialogProps {
  open: boolean;
  onClose: () => void;
  onComplete: (backupCodes: string[]) => void;
  isSubmitting?: boolean;
}

const Setup2FADialog: React.FC<Setup2FADialogProps> = ({
  open,
  onClose,
  onComplete,
  isSubmitting = false
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [qrCode, setQrCode] = useState<string>('');
  const [verificationCode, setVerificationCode] = useState<string>('');
  const [backupCodes, setBackupCodes] = useState<string[]>([]);
  const [error, setError] = useState<string>('');

  const handleNext = async () => {
    try {
      switch (activeStep) {
        case 0:
          // Generate QR code
          // In real implementation, this would be an API call
          const mockQrCode = 'otpauth://totp/ForexTradingPlatform:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=ForexTradingPlatform';
          setQrCode(mockQrCode);
          setActiveStep(1);
          break;

        case 1:
          // Verify the code
          if (verificationCode.length !== 6) {
            setError('Please enter a valid 6-digit code');
            return;
          }
          // In real implementation, this would verify with the server
          await new Promise(resolve => setTimeout(resolve, 1000));
          // Generate backup codes
          setBackupCodes([
            'ABCD-EFGH-IJKL',
            'MNOP-QRST-UVWX',
            'YZAB-CDEF-GHIJ',
            'KLMN-OPQR-STUV',
            'WXYZ-1234-5678'
          ]);
          setActiveStep(2);
          break;

        case 2:
          onComplete(backupCodes);
          onClose();
          break;
      }
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
    setError('');
  };

  const steps = ['Generate Key', 'Verify Code', 'Save Backup Codes'];

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>
        Setup Two-Factor Authentication (2FA)
      </DialogTitle>

      <DialogContent>
        {error && (
          <Box sx={{ mb: 2 }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {activeStep === 0 && (
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body1" paragraph>
              To get started with two-factor authentication, we'll generate a QR code
              that you can scan with your authenticator app.
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              We recommend using Google Authenticator or Authy.
            </Typography>
          </Box>
        )}

        {activeStep === 1 && (
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body1" paragraph>
              Scan this QR code with your authenticator app:
            </Typography>
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'center' }}>
              <QRCode value={qrCode} size={200} />
            </Box>
            <TextField
              label="Enter 6-digit verification code"
              value={verificationCode}
              onChange={(e) => setVerificationCode(e.target.value)}
              fullWidth
              type="number"
              inputProps={{ maxLength: 6 }}
              sx={{ mt: 2 }}
            />
          </Box>
        )}

        {activeStep === 2 && (
          <Box>
            <Typography variant="body1" paragraph>
              Please save these backup codes in a secure location. You'll need them if you
              lose access to your authenticator app.
            </Typography>
            <Paper 
              variant="outlined" 
              sx={{ 
                p: 2, 
                mb: 2,
                fontFamily: 'monospace',
                backgroundColor: (theme) => theme.palette.grey[100]
              }}
            >
              {backupCodes.map((code, index) => (
                <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                  {code}
                </Typography>
              ))}
            </Paper>
            <Alert severity="warning">
              These codes will only be shown once. Make sure to save them now!
            </Alert>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        {activeStep > 0 && (
          <Button onClick={handleBack} disabled={isSubmitting}>
            Back
          </Button>
        )}
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleNext}
          variant="contained"
          disabled={isSubmitting || (activeStep === 1 && verificationCode.length !== 6)}
        >
          {isSubmitting ? (
            <CircularProgress size={24} />
          ) : activeStep === steps.length - 1 ? (
            'Finish'
          ) : (
            'Next'
          )}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default Setup2FADialog;
