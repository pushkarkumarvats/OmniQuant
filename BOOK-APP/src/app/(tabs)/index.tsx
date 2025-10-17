import { useState } from 'react';
import { View, StyleSheet, FlatList, RefreshControl } from 'react-native';
import { Text, Searchbar, FAB, ActivityIndicator, Button } from 'react-native-paper';
import { useBooks } from '@/hooks/useBooks';
import { BookCard } from '@/components/BookCard';
import { router } from 'expo-router';

export default function LibraryScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const { books, isLoading, refetch, deleteBook, isDeleting } = useBooks();

  const filteredBooks = books.filter(
    book =>
      book.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      book.authors.some(author => author.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const handleDelete = (bookId: string) => {
    if (confirm('Are you sure you want to delete this book?')) {
      deleteBook(bookId);
    }
  };

  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text variant="headlineMedium" style={styles.title}>
          My Library
        </Text>
        <Searchbar
          placeholder="Search books..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          style={styles.searchbar}
        />
      </View>

      {filteredBooks.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>📚</Text>
          <Text variant="headlineSmall" style={styles.emptyTitle}>
            {searchQuery ? 'No books found' : 'No books yet'}
          </Text>
          <Text variant="bodyLarge" style={styles.emptyText}>
            {searchQuery
              ? 'Try a different search term'
              : 'Start building your library by uploading your first book'}
          </Text>
          {!searchQuery && (
            <Button
              mode="contained"
              onPress={() => router.push('/(tabs)/upload')}
              style={styles.uploadButton}
            >
              Upload Your First Book
            </Button>
          )}
        </View>
      ) : (
        <FlatList
          data={filteredBooks}
          keyExtractor={item => item.id}
          renderItem={({ item }) => (
            <BookCard book={item} onDelete={() => handleDelete(item.id)} />
          )}
          contentContainerStyle={styles.list}
          refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
        />
      )}

      <FAB
        icon="plus"
        style={styles.fab}
        onPress={() => router.push('/(tabs)/upload')}
        label="Upload"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    backgroundColor: '#fff',
    padding: 16,
    paddingTop: 48,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    marginBottom: 16,
    fontWeight: '600',
  },
  searchbar: {
    elevation: 0,
    backgroundColor: '#F5F5F5',
  },
  list: {
    padding: 16,
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
    marginBottom: 24,
  },
  uploadButton: {
    paddingHorizontal: 16,
  },
  fab: {
    position: 'absolute',
    right: 16,
    bottom: 16,
    backgroundColor: '#1976D2',
  },
});
