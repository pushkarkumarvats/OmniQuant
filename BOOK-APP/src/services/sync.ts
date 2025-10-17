/**
 * Sync Service
 * Handles real-time synchronization across devices using Supabase Realtime and CRDT
 * @module services/sync
 */

import { supabase } from './supabase';
import { logger } from '@/utils/logger';
import { RealtimeChannel } from '@supabase/supabase-js';

type SyncEntity = 'book' | 'annotation' | 'bookmark' | 'collection';

interface SyncEvent {
  type: 'INSERT' | 'UPDATE' | 'DELETE';
  table: string;
  record: any;
  old_record?: any;
}

type SyncCallback = (event: SyncEvent) => void;

/**
 * Sync Manager for real-time data synchronization
 */
class SyncManager {
  private channels: Map<string, RealtimeChannel> = new Map();
  private callbacks: Map<string, SyncCallback[]> = new Map();
  private userId?: string;
  private isOnline: boolean = true;
  private pendingChanges: SyncEvent[] = [];

  /**
   * Initializes sync for a user
   * @param userId - User ID to sync for
   */
  async initialize(userId: string): Promise<void> {
    this.userId = userId;
    logger.info('Sync initialized', { userId });

    // Subscribe to all user entities
    await this.subscribeToBooks();
    await this.subscribeToAnnotations();
    await this.subscribeToBookmarks();
    await this.subscribeToCollections();

    // Setup online/offline detection
    this.setupNetworkDetection();
  }

  /**
   * Subscribes to book changes
   */
  private async subscribeToBooks(): Promise<void> {
    if (!this.userId) return;

    const channel = supabase
      .channel(`books:${this.userId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'books',
          filter: `user_id=eq.${this.userId}`,
        },
        (payload) => {
          this.handleChange('book', payload);
        }
      )
      .subscribe((status) => {
        if (status === 'SUBSCRIBED') {
          logger.info('Subscribed to books channel');
        }
      });

    this.channels.set('books', channel);
  }

  /**
   * Subscribes to annotation changes
   */
  private async subscribeToAnnotations(): Promise<void> {
    if (!this.userId) return;

    const channel = supabase
      .channel(`annotations:${this.userId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'annotations',
          filter: `user_id=eq.${this.userId}`,
        },
        (payload) => {
          this.handleChange('annotation', payload);
        }
      )
      .subscribe((status) => {
        if (status === 'SUBSCRIBED') {
          logger.info('Subscribed to annotations channel');
        }
      });

    this.channels.set('annotations', channel);
  }

  /**
   * Subscribes to bookmark changes
   */
  private async subscribeToBookmarks(): Promise<void> {
    if (!this.userId) return;

    const channel = supabase
      .channel(`bookmarks:${this.userId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'bookmarks',
          filter: `user_id=eq.${this.userId}`,
        },
        (payload) => {
          this.handleChange('bookmark', payload);
        }
      )
      .subscribe((status) => {
        if (status === 'SUBSCRIBED') {
          logger.info('Subscribed to bookmarks channel');
        }
      });

    this.channels.set('bookmarks', channel);
  }

  /**
   * Subscribes to collection changes
   */
  private async subscribeToCollections(): Promise<void> {
    if (!this.userId) return;

    const channel = supabase
      .channel(`collections:${this.userId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'collections',
          filter: `user_id=eq.${this.userId}`,
        },
        (payload) => {
          this.handleChange('collection', payload);
        }
      )
      .subscribe((status) => {
        if (status === 'SUBSCRIBED') {
          logger.info('Subscribed to collections channel');
        }
      });

    this.channels.set('collections', channel);
  }

  /**
   * Handles a change event from Realtime
   */
  private handleChange(entity: SyncEntity, payload: any): void {
    const event: SyncEvent = {
      type: payload.eventType,
      table: payload.table,
      record: payload.new,
      old_record: payload.old,
    };

    logger.debug(`Sync event: ${entity}`, event);

    // Call registered callbacks
    const callbacks = this.callbacks.get(entity) || [];
    callbacks.forEach((callback) => {
      try {
        callback(event);
      } catch (error) {
        logger.error(`Error in sync callback for ${entity}`, error);
      }
    });

    // If offline, queue the change
    if (!this.isOnline) {
      this.pendingChanges.push(event);
    }
  }

  /**
   * Registers a callback for entity changes
   * @param entity - Entity type to listen to
   * @param callback - Callback function
   */
  on(entity: SyncEntity, callback: SyncCallback): () => void {
    const callbacks = this.callbacks.get(entity) || [];
    callbacks.push(callback);
    this.callbacks.set(entity, callbacks);

    // Return unsubscribe function
    return () => {
      const cbs = this.callbacks.get(entity) || [];
      const index = cbs.indexOf(callback);
      if (index > -1) {
        cbs.splice(index, 1);
        this.callbacks.set(entity, cbs);
      }
    };
  }

  /**
   * Unsubscribes from all channels
   */
  async unsubscribeAll(): Promise<void> {
    for (const [name, channel] of this.channels) {
      await supabase.removeChannel(channel);
      logger.info(`Unsubscribed from ${name} channel`);
    }

    this.channels.clear();
    this.callbacks.clear();
  }

  /**
   * Sets up network detection
   */
  private setupNetworkDetection(): void {
    // This would use NetInfo or similar in React Native
    // For now, we'll assume online
    this.isOnline = true;
  }

  /**
   * Processes pending changes when coming back online
   */
  async processPendingChanges(): Promise<void> {
    if (this.pendingChanges.length === 0) return;

    logger.info('Processing pending changes', {
      count: this.pendingChanges.length,
    });

    // Process changes in order
    for (const change of this.pendingChanges) {
      // Re-apply changes or resolve conflicts
      // This is a simplified version - production would need CRDT
      logger.debug('Processing pending change', change);
    }

    this.pendingChanges = [];
  }

  /**
   * Forces a full sync
   */
  async forceSync(): Promise<void> {
    logger.info('Force sync requested');

    // This would trigger a full data fetch and reconciliation
    // Implementation depends on your sync strategy
  }

  /**
   * Gets sync status
   */
  getStatus() {
    return {
      isOnline: this.isOnline,
      activeChannels: this.channels.size,
      pendingChanges: this.pendingChanges.length,
      userId: this.userId,
    };
  }

  /**
   * Manually triggers sync for a specific entity
   * @param entity - Entity type to sync
   */
  async syncEntity(entity: SyncEntity): Promise<void> {
    logger.info(`Manual sync triggered for ${entity}`);
    // Implementation would fetch latest data and update local state
  }
}

// Export singleton instance
export const syncManager = new SyncManager();

/**
 * Hook for using sync in React components
 * @param entity - Entity type to sync
 * @param callback - Callback for changes
 */
export function useSync(entity: SyncEntity, callback: SyncCallback) {
  // This would be a React hook in actual implementation
  // For now, just export the manager
  return {
    subscribe: () => syncManager.on(entity, callback),
    status: syncManager.getStatus(),
  };
}

export default syncManager;
