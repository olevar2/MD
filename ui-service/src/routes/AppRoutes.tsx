import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Layouts
import AppLayout from '../components/layout/AppLayout';

// Components
import AdvancedTradingDashboard from '../components/dashboard/AdvancedTradingDashboard';
import InteractiveAnalysisTools from '../components/analysis/InteractiveAnalysisTools';
import RealTimeMonitoringDisplays from '../components/monitoring/RealTimeMonitoringDisplays';
import PerformanceVisualizationTools from '../components/performance/PerformanceVisualizationTools';

// Auth and other components
import Login from '../components/auth/Login';
import NotFound from '../components/common/NotFound';

const AppRoutes = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        
        {/* Protected routes */}
        <Route path="/" element={<AppLayout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<AdvancedTradingDashboard />} />
          <Route path="analysis" element={<InteractiveAnalysisTools />} />
          <Route path="monitoring" element={<RealTimeMonitoringDisplays />} />
          <Route path="performance" element={<PerformanceVisualizationTools />} />
          
          {/* Add more routes as needed */}
          <Route path="charts" element={<div>Charts coming soon</div>} />
          <Route path="trading" element={<div>Trading interface coming soon</div>} />
          <Route path="settings" element={<div>Settings coming soon</div>} />
        </Route>
        
        {/* Not found route */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
};

export default AppRoutes;
