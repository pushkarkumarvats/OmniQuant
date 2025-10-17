import { View, StyleSheet, FlatList } from 'react-native';
import { Text, IconButton, Chip, Card, ActivityIndicator } from 'react-native-paper';
import { useLocalSearchParams, router } from 'expo-router';
import { useAnnotations } from '@/hooks/useAnnotations';
import { useState } from 'react';
import type { Annotation } from '@/types/app';

export default function AnnotationsScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { annotations, isLoading, deleteAnnotation } = useAnnotations(id);
  const [filter, setFilter] = useState<'all' | 'highlight' | 'note' | 'drawing'>('all');

  const filteredAnnotations = annotations.filter(
    a => filter === 'all' || a.type === filter
  );

  const groupedByPage = filteredAnnotations.reduce((acc, annotation) => {
    const page = annotation.page;
    if (!acc[page]) {
      acc[page] = [];
    }
    acc[page].push(annotation);
    return acc;
  }, {} as Record<number, Annotation[]>);

  const handleAnnotationPress = (annotation: Annotation) => {
    router.push(`/reader/${id}?page=${annotation.page}`);
  };

  const handleDelete = (annotationId: string) => {
    deleteAnnotation(annotationId);
  };

  const getAnnotationIcon = (type: string) => {
    switch (type) {
      case 'highlight':
        return '✏️';
      case 'note':
        return '📝';
      case 'drawing':
        return '✍️';
      default:
        return '📌';
    }
  };

  const getAnnotationColor = (color: string) => {
    return color || '#FFEB3B';
  };

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <IconButton icon="arrow-left" size={24} onPress={() => router.back()} />
        <Text variant="titleLarge" style={styles.title}>
          Annotations
        </Text>
        <View style={{ width: 48 }} />
      </View>

      {/* Filter Chips */}
      <View style={styles.filterContainer}>
        <Chip
          selected={filter === 'all'}
          onPress={() => setFilter('all')}
          style={styles.chip}
        >
          All ({annotations.length})
        </Chip>
        <Chip
          selected={filter === 'highlight'}
          onPress={() => setFilter('highlight')}
          style={styles.chip}
        >
          Highlights ({annotations.filter(a => a.type === 'highlight').length})
        </Chip>
        <Chip
          selected={filter === 'note'}
          onPress={() => setFilter('note')}
          style={styles.chip}
        >
          Notes ({annotations.filter(a => a.type === 'note').length})
        </Chip>
      </View>

      {/* Annotations List */}
      {filteredAnnotations.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>📝</Text>
          <Text variant="headlineSmall" style={styles.emptyTitle}>
            No annotations yet
          </Text>
          <Text variant="bodyLarge" style={styles.emptyText}>
            Start highlighting and adding notes while reading
          </Text>
        </View>
      ) : (
        <FlatList
          data={Object.entries(groupedByPage)}
          keyExtractor={([page]) => page}
          renderItem={({ item: [page, pageAnnotations] }) => (
            <View style={styles.pageGroup}>
              <Text variant="titleMedium" style={styles.pageTitle}>
                Page {page}
              </Text>
              {pageAnnotations.map(annotation => (
                <Card
                  key={annotation.id}
                  style={styles.annotationCard}
                  onPress={() => handleAnnotationPress(annotation)}
                >
                  <Card.Content>
                    <View style={styles.annotationHeader}>
                      <View style={styles.annotationMeta}>
                        <Text style={styles.annotationIcon}>
                          {getAnnotationIcon(annotation.type)}
                        </Text>
                        {annotation.type === 'highlight' && (
                          <View
                            style={[
                              styles.colorIndicator,
                              { backgroundColor: getAnnotationColor(annotation.color) },
                            ]}
                          />
                        )}
                      </View>
                      <IconButton
                        icon="close"
                        size={20}
                        onPress={() => handleDelete(annotation.id)}
                      />
                    </View>

                    {annotation.selected_text && (
                      <Text
                        variant="bodyMedium"
                        style={[
                          styles.selectedText,
                          annotation.type === 'highlight' && {
                            backgroundColor: `${getAnnotationColor(annotation.color)}40`,
                          },
                        ]}
                      >
                        "{annotation.selected_text}"
                      </Text>
                    )}

                    {annotation.content && (
                      <Text variant="bodyMedium" style={styles.annotationContent}>
                        {annotation.content}
                      </Text>
                    )}

                    <Text variant="bodySmall" style={styles.timestamp}>
                      {new Date(annotation.created_at).toLocaleDateString()}
                    </Text>
                  </Card.Content>
                </Card>
              ))}
            </View>
          )}
          contentContainerStyle={styles.list}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingTop: 40,
    paddingBottom: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    flex: 1,
    textAlign: 'center',
    fontWeight: '600',
  },
  filterContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#fff',
  },
  chip: {
    marginRight: 8,
  },
  list: {
    padding: 16,
  },
  pageGroup: {
    marginBottom: 24,
  },
  pageTitle: {
    marginBottom: 12,
    fontWeight: '600',
    color: '#1976D2',
  },
  annotationCard: {
    marginBottom: 12,
  },
  annotationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  annotationMeta: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  annotationIcon: {
    fontSize: 20,
    marginRight: 8,
  },
  colorIndicator: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  selectedText: {
    fontStyle: 'italic',
    marginBottom: 8,
    padding: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#1976D2',
  },
  annotationContent: {
    marginBottom: 8,
    color: '#424242',
  },
  timestamp: {
    color: '#9E9E9E',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyTitle: {
    marginBottom: 8,
    textAlign: 'center',
  },
  emptyText: {
    color: '#757575',
    textAlign: 'center',
  },
});
