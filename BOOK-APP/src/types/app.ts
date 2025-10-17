import type { Database } from './database';

// Extract table types
export type User = Database['public']['Tables']['users']['Row'];
export type Book = Database['public']['Tables']['books']['Row'];
export type Annotation = Database['public']['Tables']['annotations']['Row'];
export type Bookmark = Database['public']['Tables']['bookmarks']['Row'];
export type Collection = Database['public']['Tables']['collections']['Row'];

// Insert types
export type UserInsert = Database['public']['Tables']['users']['Insert'];
export type BookInsert = Database['public']['Tables']['books']['Insert'];
export type AnnotationInsert = Database['public']['Tables']['annotations']['Insert'];
export type BookmarkInsert = Database['public']['Tables']['bookmarks']['Insert'];
export type CollectionInsert = Database['public']['Tables']['collections']['Insert'];

// Update types
export type UserUpdate = Database['public']['Tables']['users']['Update'];
export type BookUpdate = Database['public']['Tables']['books']['Update'];
export type AnnotationUpdate = Database['public']['Tables']['annotations']['Update'];
export type BookmarkUpdate = Database['public']['Tables']['bookmarks']['Update'];
export type CollectionUpdate = Database['public']['Tables']['collections']['Update'];

// Annotation types
export type AnnotationType =
  | 'highlight'
  | 'underline'
  | 'strikeout'
  | 'note'
  | 'drawing'
  | 'text'
  | 'comment';

// Annotation coordinates
export interface AnnotationCoords {
  x: number; // 0-1 normalized
  y: number; // 0-1 normalized
  width: number; // 0-1 normalized
  height: number; // 0-1 normalized
}

// Reader view mode
export type ReaderViewMode = 'single' | 'double' | 'scroll';

// Theme
export type Theme = 'light' | 'dark' | 'sepia';

// User preferences
export interface UserPreferences {
  theme: Theme;
  defaultView: 'grid' | 'list';
  fontSize: number;
  autoSync: boolean;
  offlineMode: boolean;
  readerViewMode?: ReaderViewMode;
  defaultHighlightColor?: string;
}

// Book with reading progress
export interface BookWithProgress extends Book {
  progress?: number; // 0-100
  currentPage?: number;
  lastReadAt?: string;
}

// Sync status
export type SyncStatus = 'synced' | 'pending' | 'error';

// Sync operation
export interface SyncOperation {
  id: string;
  type: 'create' | 'update' | 'delete';
  table: 'annotations' | 'bookmarks' | 'books';
  payload: any;
  timestamp: number;
  retries: number;
  status: SyncStatus;
  error?: string;
}

// File upload status
export type UploadStatus = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

// File upload progress
export interface UploadProgress {
  bookId?: string;
  fileName: string;
  fileSize: number;
  uploadedBytes: number;
  status: UploadStatus;
  error?: string;
}

// External book search result
export interface ExternalBook {
  id: string;
  title: string;
  authors: string[];
  publisher?: string;
  publishedDate?: string;
  isbn?: string;
  description?: string;
  coverUrl?: string;
  pdfUrl?: string | null;
  previewLink?: string;
}

// Search result
export interface SearchResult {
  bookId: string;
  title: string;
  authors: string[];
  thumbnailUrl?: string;
  matchSnippet: string;
  page?: number;
  rank: number;
}

// Device info
export interface DeviceInfo {
  os: string;
  osVersion: string;
  appVersion: string;
  model?: string;
}

// Reading session
export interface ReadingSession {
  bookId: string;
  startTime: number;
  endTime?: number;
  pagesRead: number;
  annotationsCreated: number;
}

// Error with context
export interface AppError {
  code: string;
  message: string;
  details?: any;
  timestamp: number;
}

// API Response wrapper
export interface ApiResponse<T> {
  data?: T;
  error?: AppError;
}
