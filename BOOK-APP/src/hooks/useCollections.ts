/**
 * Collections Hook
 * React Query hooks for managing collections
 * @module hooks/useCollections
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getCollections,
  getCollectionWithBooks,
  createCollection,
  updateCollection,
  deleteCollection,
  addBookToCollection,
  removeBookFromCollection,
  getBookCollections,
  createDefaultCollections,
} from '@/services/collections';
import { logger } from '@/utils/logger';
import { analytics } from '@/utils/analytics';

// ==================== QUERY KEYS ====================

export const collectionKeys = {
  all: ['collections'] as const,
  lists: () => [...collectionKeys.all, 'list'] as const,
  list: () => [...collectionKeys.lists()] as const,
  details: () => [...collectionKeys.all, 'detail'] as const,
  detail: (id: string) => [...collectionKeys.details(), id] as const,
  bookCollections: (bookId: string) =>
    [...collectionKeys.all, 'book', bookId] as const,
};

// ==================== QUERIES ====================

/**
 * Hook to fetch all collections
 */
export function useCollections() {
  return useQuery({
    queryKey: collectionKeys.list(),
    queryFn: getCollections,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to fetch a single collection with its books
 * @param collectionId - Collection ID
 */
export function useCollection(collectionId: string) {
  return useQuery({
    queryKey: collectionKeys.detail(collectionId),
    queryFn: () => getCollectionWithBooks(collectionId),
    enabled: !!collectionId,
  });
}

/**
 * Hook to fetch collections for a specific book
 * @param bookId - Book ID
 */
export function useBookCollections(bookId: string) {
  return useQuery({
    queryKey: collectionKeys.bookCollections(bookId),
    queryFn: () => getBookCollections(bookId),
    enabled: !!bookId,
  });
}

// ==================== MUTATIONS ====================

/**
 * Hook to create a new collection
 */
export function useCreateCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      name,
      description,
      color,
    }: {
      name: string;
      description?: string;
      color?: string;
    }) => createCollection(name, description, color),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      logger.info('Collection created', { collectionId: data.id });
      analytics.trackEvent('collection_created', {
        collection_id: data.id,
        collection_name: data.name,
      });
    },
    onError: (error) => {
      logger.error('Failed to create collection', error);
    },
  });
}

/**
 * Hook to update a collection
 */
export function useUpdateCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      collectionId,
      updates,
    }: {
      collectionId: string;
      updates: {
        name?: string;
        description?: string;
        color?: string;
      };
    }) => updateCollection(collectionId, updates),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      queryClient.invalidateQueries({
        queryKey: collectionKeys.detail(variables.collectionId),
      });
      logger.info('Collection updated', { collectionId: variables.collectionId });
    },
    onError: (error) => {
      logger.error('Failed to update collection', error);
    },
  });
}

/**
 * Hook to delete a collection
 */
export function useDeleteCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (collectionId: string) => deleteCollection(collectionId),
    onSuccess: (_, collectionId) => {
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      queryClient.removeQueries({ queryKey: collectionKeys.detail(collectionId) });
      logger.info('Collection deleted', { collectionId });
      analytics.trackEvent('collection_deleted', {
        collection_id: collectionId,
      });
    },
    onError: (error) => {
      logger.error('Failed to delete collection', error);
    },
  });
}

/**
 * Hook to add a book to a collection
 */
export function useAddBookToCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      bookId,
      collectionId,
    }: {
      bookId: string;
      collectionId: string;
    }) => addBookToCollection(bookId, collectionId),
    onSuccess: (_, { bookId, collectionId }) => {
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      queryClient.invalidateQueries({
        queryKey: collectionKeys.detail(collectionId),
      });
      queryClient.invalidateQueries({
        queryKey: collectionKeys.bookCollections(bookId),
      });
      logger.info('Book added to collection', { bookId, collectionId });
      analytics.trackEvent('book_added_to_collection', {
        book_id: bookId,
        collection_id: collectionId,
      });
    },
    onError: (error) => {
      logger.error('Failed to add book to collection', error);
    },
  });
}

/**
 * Hook to remove a book from a collection
 */
export function useRemoveBookFromCollection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      bookId,
      collectionId,
    }: {
      bookId: string;
      collectionId: string;
    }) => removeBookFromCollection(bookId, collectionId),
    onSuccess: (_, { bookId, collectionId }) => {
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      queryClient.invalidateQueries({
        queryKey: collectionKeys.detail(collectionId),
      });
      queryClient.invalidateQueries({
        queryKey: collectionKeys.bookCollections(bookId),
      });
      logger.info('Book removed from collection', { bookId, collectionId });
    },
    onError: (error) => {
      logger.error('Failed to remove book from collection', error);
    },
  });
}

/**
 * Hook to create default collections for new users
 */
export function useCreateDefaultCollections() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: createDefaultCollections,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
      logger.info('Default collections created');
    },
    onError: (error) => {
      logger.error('Failed to create default collections', error);
    },
  });
}
