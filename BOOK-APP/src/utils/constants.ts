export const APP_NAME = 'BookFlow';
export const APP_VERSION = '1.0.0';

// Storage limits
export const FREE_STORAGE_LIMIT = 104857600; // 100MB in bytes
export const PRO_STORAGE_LIMIT = 10737418240; // 10GB in bytes

// File limits
export const MAX_FILE_SIZE = 209715200; // 200MB in bytes
export const ALLOWED_FILE_TYPES = ['application/pdf'];

// Pagination
export const BOOKS_PER_PAGE = 20;
export const SEARCH_RESULTS_LIMIT = 50;

// Sync
export const SYNC_INTERVAL = 5000; // 5 seconds
export const MAX_RETRY_ATTEMPTS = 3;
export const RETRY_DELAY = 1000; // 1 second

// UI
export const ANIMATION_DURATION = 300;
export const DEBOUNCE_DELAY = 500;

// Colors
export const COLORS = {
  primary: '#1976D2',
  secondary: '#FF6F00',
  success: '#4CAF50',
  warning: '#FFC107',
  error: '#F44336',
  background: '#F5F5F5',
  surface: '#FFFFFF',
  text: '#212121',
  textSecondary: '#757575',
  border: '#E0E0E0',
};

// Annotation colors
export const ANNOTATION_COLORS = {
  yellow: '#FFEB3B',
  green: '#4CAF50',
  blue: '#2196F3',
  pink: '#E91E63',
  orange: '#FF9800',
  purple: '#9C27B0',
};

// Supported languages
export const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
];

// API Endpoints (if using external services)
export const GOOGLE_BOOKS_API = 'https://www.googleapis.com/books/v1/volumes';
export const OPEN_LIBRARY_API = 'https://openlibrary.org/api';
