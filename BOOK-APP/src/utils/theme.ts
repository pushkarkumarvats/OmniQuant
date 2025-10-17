/**
 * Theme Utility
 * Handles dark mode, theme colors, and style consistency
 * @module utils/theme
 */

import { useColorScheme } from 'react-native';

// ==================== COLOR DEFINITIONS ====================

export const Colors = {
  // Primary colors
  primary: {
    main: '#6366F1',        // Indigo-500
    light: '#818CF8',       // Indigo-400
    dark: '#4F46E5',        // Indigo-600
    contrastText: '#FFFFFF',
  },

  // Secondary colors
  secondary: {
    main: '#EC4899',        // Pink-500
    light: '#F472B6',       // Pink-400
    dark: '#DB2777',        // Pink-600
    contrastText: '#FFFFFF',
  },

  // Success colors
  success: {
    main: '#10B981',        // Green-500
    light: '#34D399',       // Green-400
    dark: '#059669',        // Green-600
    contrastText: '#FFFFFF',
  },

  // Warning colors
  warning: {
    main: '#F59E0B',        // Amber-500
    light: '#FBD38',        // Amber-400
    dark: '#D97706',        // Amber-600
    contrastText: '#FFFFFF',
  },

  // Error colors
  error: {
    main: '#EF4444',        // Red-500
    light: '#F87171',       // Red-400
    dark: '#DC2626',        // Red-600
    contrastText: '#FFFFFF',
  },

  // Info colors
  info: {
    main: '#3B82F6',        // Blue-500
    light: '#60A5FA',       // Blue-400
    dark: '#2563EB',        // Blue-600
    contrastText: '#FFFFFF',
  },

  // Light theme colors
  light: {
    background: '#FFFFFF',
    surface: '#F9FAFB',      // Gray-50
    card: '#FFFFFF',
    text: {
      primary: '#1F2937',    // Gray-800
      secondary: '#6B7280',  // Gray-500
      disabled: '#9CA3AF',   // Gray-400
    },
    divider: '#E5E7EB',      // Gray-200
    border: '#D1D5DB',       // Gray-300
    ripple: 'rgba(0, 0, 0, 0.12)',
  },

  // Dark theme colors
  dark: {
    background: '#111827',   // Gray-900
    surface: '#1F2937',      // Gray-800
    card: '#374151',         // Gray-700
    text: {
      primary: '#F9FAFB',    // Gray-50
      secondary: '#D1D5DB',  // Gray-300
      disabled: '#6B7280',   // Gray-500
    },
    divider: '#374151',      // Gray-700
    border: '#4B5563',       // Gray-600
    ripple: 'rgba(255, 255, 255, 0.12)',
  },

  // Annotation colors
  annotation: {
    yellow: '#FEF08A',       // Yellow highlight
    green: '#86EFAC',        // Green highlight
    blue: '#93C5FD',         // Blue highlight
    pink: '#F9A8D4',         // Pink highlight
    purple: '#C4B5FD',       // Purple highlight
    orange: '#FDBA74',       // Orange highlight
  },
} as const;

// ==================== THEME OBJECT ====================

export interface Theme {
  dark: boolean;
  colors: {
    primary: string;
    background: string;
    card: string;
    text: string;
    textSecondary: string;
    border: string;
    notification: string;
    surface: string;
    error: string;
    success: string;
    warning: string;
  };
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
    xxl: number;
  };
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
    full: number;
  };
  typography: {
    h1: { fontSize: number; fontWeight: string; lineHeight: number };
    h2: { fontSize: number; fontWeight: string; lineHeight: number };
    h3: { fontSize: number; fontWeight: string; lineHeight: number };
    h4: { fontSize: number; fontWeight: string; lineHeight: number };
    body: { fontSize: number; lineHeight: number };
    bodySmall: { fontSize: number; lineHeight: number };
    caption: { fontSize: number; lineHeight: number };
  };
  shadows: {
    sm: object;
    md: object;
    lg: object;
  };
}

const baseSpacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

const baseBorderRadius = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  full: 9999,
};

const baseTypography = {
  h1: { fontSize: 32, fontWeight: '700' as const, lineHeight: 40 },
  h2: { fontSize: 24, fontWeight: '600' as const, lineHeight: 32 },
  h3: { fontSize: 20, fontWeight: '600' as const, lineHeight: 28 },
  h4: { fontSize: 18, fontWeight: '600' as const, lineHeight: 24 },
  body: { fontSize: 16, lineHeight: 24 },
  bodySmall: { fontSize: 14, lineHeight: 20 },
  caption: { fontSize: 12, lineHeight: 16 },
};

const baseShadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 8,
  },
};

export const LightTheme: Theme = {
  dark: false,
  colors: {
    primary: Colors.primary.main,
    background: Colors.light.background,
    card: Colors.light.card,
    text: Colors.light.text.primary,
    textSecondary: Colors.light.text.secondary,
    border: Colors.light.border,
    notification: Colors.error.main,
    surface: Colors.light.surface,
    error: Colors.error.main,
    success: Colors.success.main,
    warning: Colors.warning.main,
  },
  spacing: baseSpacing,
  borderRadius: baseBorderRadius,
  typography: baseTypography,
  shadows: baseShadows,
};

export const DarkTheme: Theme = {
  dark: true,
  colors: {
    primary: Colors.primary.light,
    background: Colors.dark.background,
    card: Colors.dark.card,
    text: Colors.dark.text.primary,
    textSecondary: Colors.dark.text.secondary,
    border: Colors.dark.border,
    notification: Colors.error.light,
    surface: Colors.dark.surface,
    error: Colors.error.light,
    success: Colors.success.light,
    warning: Colors.warning.light,
  },
  spacing: baseSpacing,
  borderRadius: baseBorderRadius,
  typography: baseTypography,
  shadows: baseShadows,
};

// ==================== HOOKS ====================

/**
 * Hook to get the current theme based on system color scheme
 * @returns Current theme object
 */
export function useTheme(): Theme {
  const colorScheme = useColorScheme();
  return colorScheme === 'dark' ? DarkTheme : LightTheme;
}

/**
 * Hook to check if dark mode is active
 * @returns true if dark mode is enabled
 */
export function useIsDarkMode(): boolean {
  const colorScheme = useColorScheme();
  return colorScheme === 'dark';
}

// ==================== UTILITY FUNCTIONS ====================

/**
 * Gets color value for current theme
 * @param lightColor - Color for light theme
 * @param darkColor - Color for dark theme
 * @param isDark - Whether dark mode is active
 * @returns Appropriate color for current theme
 */
export function getThemedColor(
  lightColor: string,
  darkColor: string,
  isDark: boolean
): string {
  return isDark ? darkColor : lightColor;
}

/**
 * Creates a style object that responds to theme
 * @param lightStyle - Style for light theme
 * @param darkStyle - Style for dark theme
 * @param isDark - Whether dark mode is active
 * @returns Appropriate style for current theme
 */
export function getThemedStyle<T>(
  lightStyle: T,
  darkStyle: T,
  isDark: boolean
): T {
  return isDark ? darkStyle : lightStyle;
}

/**
 * Converts hex color to RGBA
 * @param hex - Hex color code
 * @param alpha - Alpha value (0-1)
 * @returns RGBA color string
 */
export function hexToRGBA(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Gets contrast color (black or white) for a given background
 * @param backgroundColor - Background color hex code
 * @returns '#000000' or '#FFFFFF' for optimal contrast
 */
export function getContrastColor(backgroundColor: string): string {
  // Remove # if present
  const hex = backgroundColor.replace('#', '');
  
  // Convert to RGB
  const r = parseInt(hex.substr(0, 2), 16);
  const g = parseInt(hex.substr(2, 2), 16);
  const b = parseInt(hex.substr(4, 2), 16);
  
  // Calculate luminance
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  
  return luminance > 0.5 ? '#000000' : '#FFFFFF';
}

// ==================== EXPORT ====================

export default {
  Colors,
  LightTheme,
  DarkTheme,
  useTheme,
  useIsDarkMode,
  getThemedColor,
  getThemedStyle,
  hexToRGBA,
  getContrastColor,
};
