import { useState, useEffect } from 'react';

type Theme = 'light' | 'dark';

interface UseThemeResult {
  currentTheme: Theme;
  toggleTheme: () => void;
}

export const useTheme = (): UseThemeResult => {
  const [currentTheme, setCurrentTheme] = useState<Theme>(() => {
    const savedTheme = localStorage.getItem('theme');
    return (savedTheme === 'light' || savedTheme === 'dark') ? savedTheme : 'light';
  });

  useEffect(() => {
    localStorage.setItem('theme', currentTheme);
    document.documentElement.setAttribute('data-theme', currentTheme);
  }, [currentTheme]);

  const toggleTheme = () => {
    setCurrentTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return {
    currentTheme,
    toggleTheme
  };
};
