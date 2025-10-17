import { useState } from 'react';
import { View, StyleSheet, FlatList } from 'react-native';
import { Text, Searchbar, ActivityIndicator, Chip } from 'react-native-paper';
import { useAuthStore } from '@/store/authStore';
import { booksService } from '@/services/books';
import { BookCard } from '@/components/BookCard';

export default function SearchScreen() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  const user = useAuthStore(state => state.user);

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim() || !user) return;

    setIsSearching(true);
    try {
      const searchResults = await booksService.searchBooks(user.id, searchQuery);
      setResults(searchResults);

      // Add to search history
      if (!searchHistory.includes(searchQuery)) {
        setSearchHistory([searchQuery, ...searchHistory].slice(0, 5));
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleQueryChange = (text: string) => {
    setQuery(text);
    if (text.trim() === '') {
      setResults([]);
    }
  };

  const handleHistoryPress = (historyQuery: string) => {
    setQuery(historyQuery);
    handleSearch(historyQuery);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text variant="headlineMedium" style={styles.title}>
          Search
        </Text>
        <Searchbar
          placeholder="Search your library..."
          onChangeText={handleQueryChange}
          value={query}
          onSubmitEditing={() => handleSearch(query)}
          style={styles.searchbar}
        />
      </View>

      {query === '' && searchHistory.length > 0 && (
        <View style={styles.historyContainer}>
          <Text variant="titleMedium" style={styles.historyTitle}>
            Recent Searches
          </Text>
          <View style={styles.chipContainer}>
            {searchHistory.map((item, index) => (
              <Chip
                key={index}
                onPress={() => handleHistoryPress(item)}
                style={styles.chip}
                icon="history"
              >
                {item}
              </Chip>
            ))}
          </View>
        </View>
      )}

      {isSearching ? (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" />
        </View>
      ) : results.length > 0 ? (
        <View style={styles.resultsContainer}>
          <Text variant="titleMedium" style={styles.resultsTitle}>
            Found {results.length} result{results.length !== 1 ? 's' : ''}
          </Text>
          <FlatList
            data={results}
            keyExtractor={item => item.book_id}
            renderItem={({ item }) => (
              <View style={styles.resultItem}>
                <Text variant="titleMedium" numberOfLines={2}>
                  {item.title}
                </Text>
                <Text variant="bodySmall" numberOfLines={2} style={styles.snippet}>
                  ...{item.match_snippet}...
                </Text>
                {item.authors && item.authors.length > 0 && (
                  <Text variant="bodySmall" style={styles.author}>
                    {item.authors.join(', ')}
                  </Text>
                )}
              </View>
            )}
            contentContainerStyle={styles.list}
          />
        </View>
      ) : query !== '' ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>🔍</Text>
          <Text variant="headlineSmall" style={styles.emptyTitle}>
            No results found
          </Text>
          <Text variant="bodyLarge" style={styles.emptyText}>
            Try different keywords or check your spelling
          </Text>
        </View>
      ) : (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>🔍</Text>
          <Text variant="headlineSmall" style={styles.emptyTitle}>
            Search your library
          </Text>
          <Text variant="bodyLarge" style={styles.emptyText}>
            Find books by title, author, or content
          </Text>
        </View>
      )}
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
  historyContainer: {
    padding: 16,
    backgroundColor: '#fff',
    marginTop: 8,
  },
  historyTitle: {
    marginBottom: 12,
    fontWeight: '600',
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  chip: {
    marginRight: 8,
    marginBottom: 8,
  },
  resultsContainer: {
    flex: 1,
  },
  resultsTitle: {
    padding: 16,
    fontWeight: '600',
  },
  list: {
    paddingHorizontal: 16,
  },
  resultItem: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  snippet: {
    color: '#757575',
    marginTop: 4,
    fontStyle: 'italic',
  },
  author: {
    color: '#9E9E9E',
    marginTop: 4,
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
