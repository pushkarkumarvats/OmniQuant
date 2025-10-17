/**
 * Empty State Component
 * Displays when there's no data to show
 * @module components/EmptyState
 */

import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { Button } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

interface EmptyStateProps {
  /** Icon name from MaterialCommunityIcons */
  icon?: keyof typeof MaterialCommunityIcons.glyphMap;
  /** Title text */
  title: string;
  /** Description text */
  description?: string;
  /** Action button text */
  actionText?: string;
  /** Action button handler */
  onAction?: () => void;
  /** Custom style */
  style?: ViewStyle;
}

/**
 * Empty State Component
 * Shows a friendly message when there's no content
 * 
 * @example
 * <EmptyState
 *   icon="book-open-variant"
 *   title="No books yet"
 *   description="Upload your first PDF to get started"
 *   actionText="Upload Book"
 *   onAction={handleUpload}
 * />
 */
export function EmptyState({
  icon = 'file-document-outline',
  title,
  description,
  actionText,
  onAction,
  style,
}: EmptyStateProps) {
  return (
    <View style={[styles.container, style]}>
      <MaterialCommunityIcons name={icon} size={80} color="#D1D5DB" />
      <Text style={styles.title}>{title}</Text>
      {description && <Text style={styles.description}>{description}</Text>}
      {actionText && onAction && (
        <Button mode="contained" onPress={onAction} style={styles.button}>
          {actionText}
        </Button>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1F2937',
    marginTop: 24,
    marginBottom: 8,
    textAlign: 'center',
  },
  description: {
    fontSize: 16,
    color: '#6B7280',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 24,
  },
  button: {
    marginTop: 8,
    paddingHorizontal: 24,
  },
});

export default EmptyState;
