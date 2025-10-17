/**
 * Bookmarks Hook
 * React Query hooks for managing bookmarks and reading progress
 * @module hooks/useBookmarks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getCurrentBookmark,
  getBookBookmarks,
  updateReadingProgress,
  deleteBookmark,
  getReadingStats,
  getRecentlyReadBooks,
} from '@/services/bookmarks';
import { logger } from '@/utils/logger';
import { analytics } from '@/utils/analytics';

// ==================== QUERY KEYS ====================

export const bookmarkKeys = {
  all: ['bookmarks'] as const,
  lists: () => [...bookmarkKeys.all, 'list'] as const,
  list: (bookId: string) => [...bookmarkKeys.lists(), bookId] as const,
  current: (bookId: string) => [...bookmarkKeys.all, 'current', bookId] as const,
  stats: (bookId: string) => [...bookmarkKeys.all, 'stats', bookId] as const,
  recent: () => [...bookmarkKeys.all, 'recent'] as const,
};

// ==================== QUERIES ====================

/**
 * Hook to fetch current bookmark for a book
 * @param bookId - Book ID
 */
export function useCurrentBookmark(bookId: string) {
  return useQuery({
    queryKey: bookmarkKeys.current(bookId),
    queryFn: () => getCurrentBookmark(bookId),
    enabled: !!bookId,
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook to fetch all bookmarks for a book
 * @param bookId - Book ID
 */
export function useBookmarks(bookId: string) {
  return useQuery({
    queryKey: bookmarkKeys.list(bookId),
    queryFn: () => getBookBookmarks(bookId),
    enabled: !!bookId,
  });
}

/**
 * Hook to fetch reading statistics for a book
 * @param bookId - Book ID
 */
export function useReadingStats(bookId: string) {
  return useQuery({
    queryKey: bookmarkKeys.stats(bookId),
    queryFn: () => getReadingStats(bookId),
    enabled: !!bookId,
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook to fetch recently read books
 * @param limit - Maximum number of books
 */
export function useRecentlyReadBooks(limit: number = 10) {
  return useQuery({
    queryKey: bookmarkKeys.recent(),
    queryFn: () => getRecentlyReadBooks(limit),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
}

// ==================== MUTATIONS ====================

/**
 * Hook to update reading progress
 */
export function useUpdateReadingProgress() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      bookId,
      pageNumber,
      scrollPercent,
    }: {
      bookId: string;
      pageNumber: number;
      scrollPercent?: number;
    }) => updateReadingProgress(bookId, pageNumber, scrollPercent),
    onMutate: async ({ bookId, pageNumber }) => {
      // Optimistically update current bookmark
      const queryKey = bookmarkKeys.current(bookId);
      await queryClient.cancelQueries({ queryKey });

      const previousBookmark = queryClient.getQueryData(queryKey);

      queryClient.setQueryData(queryKey, (old: any) => ({
        ...old,
        page_number: pageNumber,
        updated_at: new Date().toISOString(),
      }));

      return { previousBookmark };
    },
    onSuccess: (_, { bookId, pageNumber }) => {
      queryClient.invalidateQueries({ queryKey: bookmarkKeys.current(bookId) });
      queryClient.invalidateQueries({ queryKey: bookmarkKeys.list(bookId) });
      queryClient.invalidateQueries({ queryKey: bookmarkKeys.stats(bookId) });
      queryClient.invalidateQueries({ queryKey: bookmarkKeys.recent() });

      logger.debug('Reading progress updated', { bookId, pageNumber });
    },
    onError: (error, { bookId }, context) => {
      // Rollback optimistic update
      if (context?.previousBookmark) {
        queryClient.setQueryData(
          bookmarkKeys.current(bookId),
          context.previousBookmark
        );
      }
      logger.error('Failed to update reading progress', error);
    },
  });
}

/**
 * Hook to delete a bookmark
 */
export function useDeleteBookmark() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      bookmarkId,
      bookId,
    }: {
      bookmarkId: string;
      bookId: string;
    }) => deleteBookmark(bookmarkId),
    onSuccess: (_, { bookId }) => {
      queryClient.invalidateQueries({ queryKey: bookmarkKeys.list(bookId) });
      logger.info('Bookmark deleted');
      analytics.trackEvent('bookmark_deleted', { book_id: bookId });
    },
    onError: (error) => {
      logger.error('Failed to delete bookmark', error);
    },
  });
}

/**
 * Hook for auto-saving reading progress
 * Call this periodically while reading
 * @param bookId - Book ID
 * @param enabled - Whether auto-save is enabled
 */
export function useAutoSaveProgress(bookId: string, enabled: boolean = true) {
  const { mutate: updateProgress } = useUpdateReadingProgress();
  const queryClient = useQueryClient();

  const saveProgress = (pageNumber: number, scrollPercent?: number) => {
    if (!enabled) return;

    // Debounce is handled by the caller
    updateProgress({ bookId, pageNumber, scrollPercent });
  };

  return { saveProgress };
}
