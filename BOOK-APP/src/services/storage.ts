import { supabase } from './supabase';
import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';

export const storageService = {
  async uploadFile(
    userId: string,
    fileUri: string,
    fileName: string,
    onProgress?: (progress: number) => void
  ): Promise<string> {
    try {
      const fileExtension = fileName.split('.').pop();
      const uniqueFileName = `${userId}/${Date.now()}.${fileExtension}`;

      // Read file as base64
      const base64 = await FileSystem.readAsStringAsync(fileUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Convert base64 to blob for upload
      const blob = await fetch(`data:application/pdf;base64,${base64}`).then(r => r.blob());

      const { data, error } = await supabase.storage
        .from('books')
        .upload(uniqueFileName, blob, {
          contentType: 'application/pdf',
          upsert: false,
        });

      if (error) throw error;

      return data.path;
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  },

  async getSignedUrl(path: string): Promise<string> {
    const { data, error } = await supabase.storage
      .from('books')
      .createSignedUrl(path, 3600); // 1 hour expiry

    if (error) throw error;
    return data.signedUrl;
  },

  async deleteFile(path: string): Promise<void> {
    const { error } = await supabase.storage.from('books').remove([path]);

    if (error) throw error;
  },

  async getFileInfo(uri: string) {
    const info = await FileSystem.getInfoAsync(uri);
    return info;
  },
};
