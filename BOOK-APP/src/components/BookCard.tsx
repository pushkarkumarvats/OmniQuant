import { View, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { Text, IconButton, ProgressBar } from 'react-native-paper';
import { router } from 'expo-router';
import type { Book } from '@/types/app';

interface BookCardProps {
  book: Book;
  onDelete?: () => void;
}

export function BookCard({ book, onDelete }: BookCardProps) {
  const handlePress = () => {
    router.push(`/reader/${book.id}`);
  };

  // Calculate reading progress (mock for now)
  const progress = 0;

  return (
    <TouchableOpacity style={styles.container} onPress={handlePress} activeOpacity={0.7}>
      <View style={styles.thumbnail}>
        {book.thumbnail_url ? (
          <Image source={{ uri: book.thumbnail_url }} style={styles.image} />
        ) : (
          <View style={[styles.placeholderThumbnail, { backgroundColor: book.cover_color }]}>
            <Text style={styles.placeholderIcon}>📄</Text>
          </View>
        )}
      </View>

      <View style={styles.content}>
        <Text variant="titleMedium" numberOfLines={2} style={styles.title}>
          {book.title}
        </Text>

        {book.authors && book.authors.length > 0 && (
          <Text variant="bodySmall" numberOfLines={1} style={styles.author}>
            {book.authors.join(', ')}
          </Text>
        )}

        {book.page_count && (
          <Text variant="bodySmall" style={styles.pages}>
            {book.page_count} pages
          </Text>
        )}

        {progress > 0 && (
          <View style={styles.progressContainer}>
            <ProgressBar progress={progress / 100} color="#1976D2" style={styles.progressBar} />
            <Text variant="bodySmall" style={styles.progressText}>
              {Math.round(progress)}%
            </Text>
          </View>
        )}
      </View>

      {onDelete && (
        <IconButton
          icon="close"
          size={20}
          onPress={e => {
            e.stopPropagation();
            onDelete();
          }}
          style={styles.deleteButton}
        />
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 8,
    marginBottom: 16,
    padding: 12,
    flexDirection: 'row',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  thumbnail: {
    width: 60,
    height: 80,
    borderRadius: 4,
    overflow: 'hidden',
    marginRight: 12,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  placeholderThumbnail: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderIcon: {
    fontSize: 32,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
  },
  title: {
    marginBottom: 4,
    fontWeight: '600',
  },
  author: {
    color: '#757575',
    marginBottom: 2,
  },
  pages: {
    color: '#9E9E9E',
    fontSize: 12,
  },
  progressContainer: {
    marginTop: 8,
    flexDirection: 'row',
    alignItems: 'center',
  },
  progressBar: {
    flex: 1,
    height: 4,
    borderRadius: 2,
  },
  progressText: {
    marginLeft: 8,
    color: '#757575',
    fontSize: 12,
  },
  deleteButton: {
    position: 'absolute',
    top: 4,
    right: 4,
  },
});
