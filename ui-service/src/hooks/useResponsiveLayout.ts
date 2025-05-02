import { useState, useEffect } from 'react';

interface ResponsiveLayoutData {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isLargeDesktop: boolean;
  screenWidth: number;
  screenHeight: number;
  isLandscape: boolean;
  isPortrait: boolean;
  deviceType: 'mobile' | 'tablet' | 'desktop';
}

/**
 * Hook for responsive layouts that provides screen size information
 * and device type classification
 * 
 * @returns Object containing responsive layout information
 */
const useResponsiveLayout = (): ResponsiveLayoutData => {
  // Default values for SSR
  const defaultValues: ResponsiveLayoutData = {
    isMobile: false,
    isTablet: false,
    isDesktop: true,
    isLargeDesktop: false,
    screenWidth: 1200,
    screenHeight: 800,
    isLandscape: true,
    isPortrait: false,
    deviceType: 'desktop',
  };
  
  // State to hold the responsive data
  const [responsiveData, setResponsiveData] = useState<ResponsiveLayoutData>(defaultValues);

  // Breakpoints (can be customized)
  const BREAKPOINTS = {
    mobile: 576,  // Max width for mobile
    tablet: 992,  // Max width for tablet
    desktop: 1400, // Max width for desktop (above is large desktop)
  };

  // Effect to compute and update responsive data
  useEffect(() => {
    // Skip if not in browser environment
    if (typeof window === 'undefined') return;
    
    // Function to compute responsive data
    const computeResponsiveData = (): ResponsiveLayoutData => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      const isMobile = width < BREAKPOINTS.mobile;
      const isTablet = width >= BREAKPOINTS.mobile && width < BREAKPOINTS.tablet;
      const isDesktop = width >= BREAKPOINTS.tablet && width < BREAKPOINTS.desktop;
      const isLargeDesktop = width >= BREAKPOINTS.desktop;
      
      const isLandscape = width > height;
      const isPortrait = !isLandscape;
      
      let deviceType: 'mobile' | 'tablet' | 'desktop';
      if (isMobile) deviceType = 'mobile';
      else if (isTablet) deviceType = 'tablet';
      else deviceType = 'desktop';
      
      return {
        isMobile,
        isTablet,
        isDesktop,
        isLargeDesktop,
        screenWidth: width,
        screenHeight: height,
        isLandscape,
        isPortrait,
        deviceType,
      };
    };
    
    // Initial computation
    setResponsiveData(computeResponsiveData());
    
    // Add resize listener
    const handleResize = () => {
      setResponsiveData(computeResponsiveData());
    };
    
    window.addEventListener('resize', handleResize);
    
    // Clean up
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);
  
  return responsiveData;
};

export default useResponsiveLayout;
