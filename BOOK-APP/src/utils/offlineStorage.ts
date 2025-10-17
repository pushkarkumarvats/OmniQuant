/**
 * Offline Storage Utility
 * Handles offline data persistence and sync queue
 * @module utils/offlineStorage
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { logger } from './logger';

const STORAGE_KEYS = {
  SYNC_QUEUE: '@bookflow/sync_queue',
  OFFLINE_DATA: '@bookflow/offline_data',
  CACHED_BOOKS: '@bookflow/cached_books',
  USER_PREFERENCES: '@bookflow/user_preferences',
} as const;

// ==================== SYNC QUEUE ====================

export interface SyncQueueItem {
  id: string;
  type: 'create' | 'update' | 'delete';
  entity: 'book' | 'annotation' | 'bookmark' | 'collection';
  data: any;
  timestamp: number;
  retries: number;
}

/**
 * Adds an item to the sync queue for later processing
 * @param item - Item to queue
 */
export async function addToSyncQueue(
  item: Omit<SyncQueueItem, 'id' | 'timestamp' | 'retries'>
): Promise<void> {
  try {
    const queue = await getSyncQueue();
    const newItem: SyncQueueItem = {
      ...item,
      id: `${Date.now()}_${Math.random()}`,
      timestamp: Date.now(),
      retries: 0,
    };

    queue.push(newItem);
    await AsyncStorage.setItem(
      STORAGE_KEYS.SYNC_QUEUE,
      JSON.stringify(queue)
    );

    logger.info('Added to sync queue', { item: newItem });
  } catch (error) {
    logger.error('Failed to add to sync queue', error);
  }
}

/**
 * Gets all items in the sync queue
 * @returns Array of queued items
 */
export async function getSyncQueue(): Promise<SyncQueueItem[]> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEYS.SYNC_QUEUE);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    logger.error('Failed to get sync queue', error);
    return [];
  }
}

/**
 * Removes an item from the sync queue
 * @param itemId - ID of item to remove
 */
export async function removeFromSyncQueue(itemId: string): Promise<void> {
  try {
    const queue = await getSyncQueue();
    const filtered = queue.filter((item) => item.id !== itemId);
    await AsyncStorage.setItem(
      STORAGE_KEYS.SYNC_QUEUE,
      JSON.stringify(filtered)
    );

    logger.info('Removed from sync queue', { itemId });
  } catch (error) {
    logger.error('Failed to remove from sync queue', error);
  }
}

/**
 * Clears the entire sync queue
 */
export async function clearSyncQueue(): Promise<void> {
  try {
    await AsyncStorage.removeItem(STORAGE_KEYS.SYNC_QUEUE);
    logger.info('Sync queue cleared');
  } catch (error) {
    logger.error('Failed to clear sync queue', error);
  }
}

/**
 * Increments retry count for a sync item
 * @param itemId - ID of item to update
 */
export async function incrementSyncRetry(itemId: string): Promise<void> {
  try {
    const queue = await getSyncQueue();
    const updated = queue.map((item) =>
      item.id === itemId ? { ...item, retries: item.retries + 1 } : item
    );
    await AsyncStorage.setItem(
      STORAGE_KEYS.SYNC_QUEUE,
      JSON.stringify(updated)
    );
  } catch (error) {
    logger.error('Failed to increment sync retry', error);
  }
}

// ==================== OFFLINE DATA ====================

/**
 * Stores data for offline access
 * @param key - Data key
 * @param data - Data to store
 */
export async function setOfflineData(key: string, data: any): Promise<void> {
  try {
    const storage = await getOfflineData();
    storage[key] = {
      data,
      timestamp: Date.now(),
    };
    await AsyncStorage.setItem(
      STORAGE_KEYS.OFFLINE_DATA,
      JSON.stringify(storage)
    );
  } catch (error) {
    logger.error('Failed to set offline data', error);
  }
}

/**
 * Gets offline data by key
 * @param key - Data key
 * @param maxAge - Maximum age in milliseconds
 * @returns Stored data or null
 */
export async function getOfflineDataByKey(
  key: string,
  maxAge?: number
): Promise<any | null> {
  try {
    const storage = await getOfflineData();
    const item = storage[key];

    if (!item) return null;

    // Check if data is too old
    if (maxAge && Date.now() - item.timestamp > maxAge) {
      return null;
    }

    return item.data;
  } catch (error) {
    logger.error('Failed to get offline data', error);
    return null;
  }
}

/**
 * Gets all offline data
 * @returns Offline data storage object
 */
export async function getOfflineData(): Promise<Record<string, any>> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEYS.OFFLINE_DATA);
    return data ? JSON.parse(data) : {};
  } catch (error) {
    logger.error('Failed to get offline data', error);
    return {};
  }
}

/**
 * Clears offline data
 */
export async function clearOfflineData(): Promise<void> {
  try {
    await AsyncStorage.removeItem(STORAGE_KEYS.OFFLINE_DATA);
    logger.info('Offline data cleared');
  } catch (error) {
    logger.error('Failed to clear offline data', error);
  }
}

// ==================== CACHED BOOKS ====================

/**
 * Caches book data for offline reading
 * @param bookId - Book ID
 * @param bookData - Book data including file info
 */
export async function cacheBook(bookId: string, bookData: any): Promise<void> {
  try {
    const cached = await getCachedBooks();
    cached[bookId] = {
      ...bookData,
      cachedAt: Date.now(),
    };
    await AsyncStorage.setItem(
      STORAGE_KEYS.CACHED_BOOKS,
      JSON.stringify(cached)
    );

    logger.info('Book cached', { bookId });
  } catch (error) {
    logger.error('Failed to cache book', error);
  }
}

/**
 * Gets a cached book
 * @param bookId - Book ID
 * @returns Cached book data or null
 */
export async function getCachedBook(bookId: string): Promise<any | null> {
  try {
    const cached = await getCachedBooks();
    return cached[bookId] || null;
  } catch (error) {
    logger.error('Failed to get cached book', error);
    return null;
  }
}

/**
 * Gets all cached books
 * @returns Object of cached books
 */
export async function getCachedBooks(): Promise<Record<string, any>> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEYS.CACHED_BOOKS);
    return data ? JSON.parse(data) : {};
  } catch (error) {
    logger.error('Failed to get cached books', error);
    return {};
  }
}

/**
 * Removes a book from cache
 * @param bookId - Book ID
 */
export async function removeCachedBook(bookId: string): Promise<void> {
  try {
    const cached = await getCachedBooks();
    delete cached[bookId];
    await AsyncStorage.setItem(
      STORAGE_KEYS.CACHED_BOOKS,
      JSON.stringify(cached)
    );

    logger.info('Book removed from cache', { bookId });
  } catch (error) {
    logger.error('Failed to remove cached book', error);
  }
}

/**
 * Clears all cached books
 */
export async function clearCachedBooks(): Promise<void> {
  try {
    await AsyncStorage.removeItem(STORAGE_KEYS.CACHED_BOOKS);
    logger.info('All cached books cleared');
  } catch (error) {
    logger.error('Failed to clear cached books', error);
  }
}

/**
 * Gets size of cached data
 * @returns Approximate size in bytes
 */
export async function getCacheSize(): Promise<number> {
  try {
    const keys = [
      STORAGE_KEYS.SYNC_QUEUE,
      STORAGE_KEYS.OFFLINE_DATA,
      STORAGE_KEYS.CACHED_BOOKS,
    ];

    let totalSize = 0;

    for (const key of keys) {
      const data = await AsyncStorage.getItem(key);
      if (data) {
        totalSize += new Blob([data]).size;
      }
    }

    return totalSize;
  } catch (error) {
    logger.error('Failed to get cache size', error);
    return 0;
  }
}

// ==================== USER PREFERENCES ====================

/**
 * Saves user preferences
 * @param preferences - Preferences object
 */
export async function saveUserPreferences(
  preferences: Record<string, any>
): Promise<void> {
  try {
    await AsyncStorage.setItem(
      STORAGE_KEYS.USER_PREFERENCES,
      JSON.stringify(preferences)
    );
    logger.info('User preferences saved');
  } catch (error) {
    logger.error('Failed to save user preferences', error);
  }
}

/**
 * Gets user preferences
 * @returns Preferences object
 */
export async function getUserPreferences(): Promise<Record<string, any>> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEYS.USER_PREFERENCES);
    return data ? JSON.parse(data) : {};
  } catch (error) {
    logger.error('Failed to get user preferences', error);
    return {};
  }
}

/**
 * Clears all storage (use with caution!)
 */
export async function clearAllStorage(): Promise<void> {
  try {
    await AsyncStorage.clear();
    logger.warn('All storage cleared');
  } catch (error) {
    logger.error('Failed to clear all storage', error);
  }
}

// ==================== EXPORT ====================

export { STORAGE_KEYS };
