/**
 * Loading Spinner Component
 * Reusable loading indicator with different sizes and styles
 * @module components/LoadingSpinner
 */

import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { ActivityIndicator, Text } from 'react-native-paper';

interface LoadingSpinnerProps {
  /** Size of the spinner */
  size?: 'small' | 'large' | number;
  /** Color of the spinner */
  color?: string;
  /** Loading message to display */
  message?: string;
  /** Whether to show full screen overlay */
  fullScreen?: boolean;
  /** Custom style */
  style?: ViewStyle;
}

/**
 * Loading Spinner Component
 * Shows an activity indicator with optional message
 * 
 * @example
 * <LoadingSpinner message="Loading books..." />
 * <LoadingSpinner size="large" fullScreen />
 */
export function LoadingSpinner({
  size = 'large',
  color = '#6366F1',
  message,
  fullScreen = false,
  style,
}: LoadingSpinnerProps) {
  const content = (
    <View style={[styles.container, fullScreen && styles.fullScreen, style]}>
      <ActivityIndicator size={size} color={color} />
      {message && <Text style={styles.message}>{message}</Text>}
    </View>
  );

  if (fullScreen) {
    return (
      <View style={styles.overlay}>
        {content}
      </View>
    );
  }

  return content;
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  fullScreen: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 32,
    minWidth: 200,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  message: {
    marginTop: 16,
    fontSize: 16,
    color: '#6B7280',
    textAlign: 'center',
  },
});

export default LoadingSpinner;
