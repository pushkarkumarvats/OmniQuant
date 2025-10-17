import { useState, useEffect } from 'react';
import { View, StyleSheet, Platform, Dimensions } from 'react-native';
import { Text, IconButton, ActivityIndicator, FAB, Portal } from 'react-native-paper';
import { useLocalSearchParams, router } from 'expo-router';
import { useBook } from '@/hooks/useBooks';
import { storageService } from '@/services/storage';
import Pdf from 'react-native-pdf';

const { width, height } = Dimensions.get('window');

export default function ReaderScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { data: book, isLoading } = useBook(id);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [annotationMode, setAnnotationMode] = useState<'none' | 'highlight' | 'note'>('none');

  useEffect(() => {
    if (book?.storage_path) {
      loadPdfUrl();
    }
  }, [book]);

  const loadPdfUrl = async () => {
    try {
      if (book?.storage_path) {
        const url = await storageService.getSignedUrl(book.storage_path);
        setPdfUrl(url);
      }
    } catch (error) {
      console.error('Error loading PDF:', error);
    }
  };

  const handlePageChanged = (page: number, numberOfPages: number) => {
    setCurrentPage(page);
    setTotalPages(numberOfPages);
  };

  const handleGoBack = () => {
    router.back();
  };

  const toggleAnnotationMode = (mode: 'highlight' | 'note') => {
    setAnnotationMode(annotationMode === mode ? 'none' : mode);
  };

  if (isLoading || !pdfUrl) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" />
        <Text style={styles.loadingText}>Loading book...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Top Bar */}
      <View style={styles.topBar}>
        <IconButton icon="arrow-left" size={24} onPress={handleGoBack} />
        <Text variant="titleMedium" numberOfLines={1} style={styles.title}>
          {book?.title}
        </Text>
        <IconButton icon="bookmark-outline" size={24} onPress={() => {}} />
      </View>

      {/* PDF Viewer */}
      {Platform.OS !== 'web' ? (
        <Pdf
          trustAllCerts={false}
          source={{ uri: pdfUrl }}
          onLoadComplete={(numberOfPages, filePath) => {
            setTotalPages(numberOfPages);
          }}
          onPageChanged={handlePageChanged}
          onError={error => {
            console.error('PDF Error:', error);
          }}
          style={styles.pdf}
          horizontal={false}
          enablePaging={true}
        />
      ) : (
        <View style={styles.webPdfContainer}>
          <Text style={styles.webMessage}>
            PDF viewer for web will use pdf.js. For now, displaying placeholder.
          </Text>
          <Text style={styles.webInfo}>Book: {book?.title}</Text>
          <Text style={styles.webInfo}>Pages: {book?.page_count}</Text>
        </View>
      )}

      {/* Page Indicator */}
      <View style={styles.pageIndicator}>
        <Text variant="bodyMedium" style={styles.pageText}>
          Page {currentPage} of {totalPages || book?.page_count || 0}
        </Text>
      </View>

      {/* Annotation Toolbar */}
      <Portal>
        <View style={styles.annotationToolbar}>
          <FAB
            icon="marker"
            small
            color={annotationMode === 'highlight' ? '#fff' : '#1976D2'}
            style={[
              styles.toolButton,
              annotationMode === 'highlight' && styles.toolButtonActive,
            ]}
            onPress={() => toggleAnnotationMode('highlight')}
          />
          <FAB
            icon="note-text"
            small
            color={annotationMode === 'note' ? '#fff' : '#1976D2'}
            style={[styles.toolButton, annotationMode === 'note' && styles.toolButtonActive]}
            onPress={() => toggleAnnotationMode('note')}
          />
          <FAB
            icon="draw"
            small
            style={styles.toolButton}
            onPress={() => {}}
          />
          <FAB
            icon="format-list-bulleted"
            small
            style={styles.toolButton}
            onPress={() => router.push(`/annotations/${id}`)}
          />
        </View>
      </Portal>

      {annotationMode !== 'none' && (
        <View style={styles.modeIndicator}>
          <Text style={styles.modeText}>
            {annotationMode === 'highlight' ? '✏️ Highlight Mode' : '📝 Note Mode'}
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
  },
  loadingText: {
    marginTop: 16,
    color: '#757575',
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingTop: 40,
    paddingBottom: 8,
    paddingHorizontal: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    flex: 1,
    marginHorizontal: 8,
  },
  pdf: {
    flex: 1,
    width,
    backgroundColor: '#fff',
  },
  webPdfContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  webMessage: {
    fontSize: 16,
    marginBottom: 16,
    textAlign: 'center',
  },
  webInfo: {
    fontSize: 14,
    color: '#757575',
    marginTop: 8,
  },
  pageIndicator: {
    position: 'absolute',
    bottom: 80,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  pageText: {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    color: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  annotationToolbar: {
    position: 'absolute',
    right: 16,
    top: height / 2 - 100,
    flexDirection: 'column',
    alignItems: 'center',
  },
  toolButton: {
    marginBottom: 8,
    backgroundColor: '#fff',
  },
  toolButtonActive: {
    backgroundColor: '#1976D2',
  },
  modeIndicator: {
    position: 'absolute',
    top: 100,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  modeText: {
    backgroundColor: '#FFC107',
    color: '#000',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    fontWeight: '600',
  },
});
