import { useState } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { Text, Button, ProgressBar, Card } from 'react-native-paper';
import * as DocumentPicker from 'expo-document-picker';
import { useAuthStore } from '@/store/authStore';
import { storageService } from '@/services/storage';
import { booksService } from '@/services/books';
import { useBooks } from '@/hooks/useBooks';
import { router } from 'expo-router';

export default function UploadScreen() {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<DocumentPicker.DocumentPickerAsset | null>(
    null
  );

  const user = useAuthStore(state => state.user);
  const { refetch } = useBooks();

  const pickDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'application/pdf',
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedFile(result.assets[0]);
      }
    } catch (error) {
      console.error('Error picking document:', error);
      Alert.alert('Error', 'Failed to pick document');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !user) return;

    try {
      setUploading(true);
      setProgress(0.1);

      // Upload file to storage
      const storagePath = await storageService.uploadFile(
        user.id,
        selectedFile.uri,
        selectedFile.name,
        progress => setProgress(progress / 100)
      );

      setProgress(0.7);

      // Get file size
      const fileInfo = await storageService.getFileInfo(selectedFile.uri);
      const fileSize = fileInfo.exists && 'size' in fileInfo ? fileInfo.size : 0;

      // Extract title from filename
      const title = selectedFile.name.replace(/\.pdf$/i, '');

      // Create book entry
      await booksService.createBook({
        owner_id: user.id,
        title,
        file_type: 'pdf',
        storage_path: storagePath,
        file_size_bytes: fileSize,
        authors: [],
      });

      setProgress(1);

      Alert.alert('Success', 'Book uploaded successfully!', [
        {
          text: 'OK',
          onPress: () => {
            setSelectedFile(null);
            setProgress(0);
            refetch();
            router.push('/(tabs)');
          },
        },
      ]);
    } catch (error) {
      console.error('Upload error:', error);
      Alert.alert('Error', 'Failed to upload book. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text variant="headlineMedium" style={styles.title}>
          Upload Book
        </Text>
        <Text variant="bodyLarge" style={styles.subtitle}>
          Add PDFs to your library
        </Text>
      </View>

      <Card style={styles.uploadCard}>
        <Card.Content>
          <View style={styles.uploadArea}>
            <Text style={styles.uploadIcon}>📄</Text>
            <Text variant="titleMedium" style={styles.uploadTitle}>
              Select a PDF file
            </Text>
            <Text variant="bodyMedium" style={styles.uploadText}>
              Tap the button below to choose a PDF from your device
            </Text>

            <Button
              mode="outlined"
              onPress={pickDocument}
              disabled={uploading}
              style={styles.selectButton}
              icon="file-pdf-box"
            >
              Choose PDF
            </Button>
          </View>
        </Card.Content>
      </Card>

      {selectedFile && (
        <Card style={styles.fileCard}>
          <Card.Content>
            <Text variant="titleMedium" style={styles.fileName}>
              {selectedFile.name}
            </Text>
            <Text variant="bodySmall" style={styles.fileSize}>
              {(selectedFile.size! / 1024 / 1024).toFixed(2)} MB
            </Text>

            {uploading ? (
              <View style={styles.progressContainer}>
                <Text variant="bodyMedium" style={styles.progressText}>
                  Uploading... {Math.round(progress * 100)}%
                </Text>
                <ProgressBar progress={progress} color="#1976D2" style={styles.progressBar} />
              </View>
            ) : (
              <View style={styles.buttonContainer}>
                <Button
                  mode="contained"
                  onPress={handleUpload}
                  disabled={uploading}
                  style={styles.uploadButton}
                  icon="upload"
                >
                  Upload
                </Button>
                <Button
                  mode="text"
                  onPress={() => setSelectedFile(null)}
                  disabled={uploading}
                  style={styles.cancelButton}
                >
                  Cancel
                </Button>
              </View>
            )}
          </Card.Content>
        </Card>
      )}

      <View style={styles.infoSection}>
        <Text variant="titleMedium" style={styles.infoTitle}>
          Supported Formats
        </Text>
        <Text variant="bodyMedium" style={styles.infoText}>
          • PDF files (up to 200MB)
        </Text>

        <Text variant="titleMedium" style={styles.infoTitle}>
          Features
        </Text>
        <Text variant="bodyMedium" style={styles.infoText}>
          • Full-text search across all books
        </Text>
        <Text variant="bodyMedium" style={styles.infoText}>
          • Highlight and annotate
        </Text>
        <Text variant="bodyMedium" style={styles.infoText}>
          • Sync across all your devices
        </Text>
        <Text variant="bodyMedium" style={styles.infoText}>
          • Offline reading support
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  content: {
    padding: 16,
  },
  header: {
    paddingTop: 32,
    paddingBottom: 16,
  },
  title: {
    marginBottom: 4,
    fontWeight: '600',
  },
  subtitle: {
    color: '#757575',
  },
  uploadCard: {
    marginBottom: 16,
  },
  uploadArea: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  uploadIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  uploadTitle: {
    marginBottom: 8,
  },
  uploadText: {
    color: '#757575',
    textAlign: 'center',
    marginBottom: 24,
  },
  selectButton: {
    paddingHorizontal: 24,
  },
  fileCard: {
    marginBottom: 16,
  },
  fileName: {
    marginBottom: 4,
  },
  fileSize: {
    color: '#757575',
    marginBottom: 16,
  },
  progressContainer: {
    marginTop: 8,
  },
  progressText: {
    marginBottom: 8,
    textAlign: 'center',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  uploadButton: {
    flex: 1,
    marginRight: 8,
  },
  cancelButton: {
    flex: 1,
    marginLeft: 8,
  },
  infoSection: {
    marginTop: 16,
  },
  infoTitle: {
    marginTop: 16,
    marginBottom: 8,
    fontWeight: '600',
  },
  infoText: {
    color: '#757575',
    marginBottom: 4,
  },
});
