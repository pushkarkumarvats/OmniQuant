/**
 * Bookmarks Service
 * Handles reading progress, bookmarks, and reading sessions
 * @module services/bookmarks
 */

import { supabase } from './supabase';
import type { Database } from '@/types/database';

type Bookmark = Database['public']['Tables']['bookmarks']['Row'];
type BookmarkInsert = Database['public']['Tables']['bookmarks']['Insert'];
type BookmarkUpdate = Database['public']['Tables']['bookmarks']['Update'];

// ==================== BOOKMARKS CRUD ====================

/**
 * Gets the current bookmark (reading position) for a book
 * @param bookId - Book ID
 * @returns Current bookmark or null
 */
export async function getCurrentBookmark(
  bookId: string
): Promise<Bookmark | null> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data, error } = await supabase
    .from('bookmarks')
    .select('*')
    .eq('book_id', bookId)
    .eq('user_id', user.id)
    .eq('is_current', true)
    .maybeSingle();

  if (error) throw error;
  return data;
}

/**
 * Gets all bookmarks for a book
 * @param bookId - Book ID
 * @returns Array of bookmarks
 */
export async function getBookBookmarks(
  bookId: string
): Promise<Bookmark[]> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data, error } = await supabase
    .from('bookmarks')
    .select('*')
    .eq('book_id', bookId)
    .eq('user_id', user.id)
    .order('page_number', { ascending: true });

  if (error) throw error;
  return data;
}

/**
 * Creates or updates a bookmark
 * @param bookId - Book ID
 * @param pageNumber - Page number
 * @param isCurrent - Whether this is the current reading position
 * @param note - Optional note
 * @returns Created/updated bookmark
 */
export async function upsertBookmark(
  bookId: string,
  pageNumber: number,
  isCurrent: boolean = true,
  note?: string
): Promise<Bookmark> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  // If this is the current bookmark, unset all other current bookmarks
  if (isCurrent) {
    await supabase
      .from('bookmarks')
      .update({ is_current: false })
      .eq('book_id', bookId)
      .eq('user_id', user.id);
  }

  const bookmarkData: BookmarkInsert = {
    book_id: bookId,
    user_id: user.id,
    page_number: pageNumber,
    is_current: isCurrent,
    note,
  };

  const { data, error } = await supabase
    .from('bookmarks')
    .upsert(bookmarkData, {
      onConflict: 'book_id,user_id,page_number',
    })
    .select()
    .single();

  if (error) throw error;
  return data;
}

/**
 * Updates reading progress for a book
 * @param bookId - Book ID
 * @param pageNumber - Current page number
 * @param scrollPercent - Optional scroll percentage within the page
 * @returns Updated bookmark
 */
export async function updateReadingProgress(
  bookId: string,
  pageNumber: number,
  scrollPercent?: number
): Promise<Bookmark> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  // Unset all current bookmarks for this book
  await supabase
    .from('bookmarks')
    .update({ is_current: false })
    .eq('book_id', bookId)
    .eq('user_id', user.id);

  const bookmarkData: BookmarkInsert = {
    book_id: bookId,
    user_id: user.id,
    page_number: pageNumber,
    scroll_percent: scrollPercent || 0,
    is_current: true,
  };

  const { data, error } = await supabase
    .from('bookmarks')
    .upsert(bookmarkData, {
      onConflict: 'book_id,user_id,page_number',
    })
    .select()
    .single();

  if (error) throw error;

  // Also update the book's last_opened_at
  await supabase
    .from('books')
    .update({ last_opened_at: new Date().toISOString() })
    .eq('id', bookId);

  return data;
}

/**
 * Deletes a bookmark
 * @param bookmarkId - Bookmark ID
 */
export async function deleteBookmark(bookmarkId: string): Promise<void> {
  const { error } = await supabase
    .from('bookmarks')
    .delete()
    .eq('id', bookmarkId);

  if (error) throw error;
}

/**
 * Deletes all bookmarks for a book
 * @param bookId - Book ID
 */
export async function deleteAllBookmarks(bookId: string): Promise<void> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { error } = await supabase
    .from('bookmarks')
    .delete()
    .eq('book_id', bookId)
    .eq('user_id', user.id);

  if (error) throw error;
}

// ==================== READING PROGRESS ====================

/**
 * Gets reading progress percentage for a book
 * @param bookId - Book ID
 * @param totalPages - Total pages in the book
 * @returns Progress percentage (0-100)
 */
export async function getReadingProgress(
  bookId: string,
  totalPages: number
): Promise<number> {
  const bookmark = await getCurrentBookmark(bookId);

  if (!bookmark) return 0;

  return Math.round((bookmark.page_number / totalPages) * 100);
}

/**
 * Gets reading statistics for a book
 * @param bookId - Book ID
 * @returns Reading statistics
 */
export async function getReadingStats(bookId: string) {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  // Get book details
  const { data: book } = await supabase
    .from('books')
    .select('total_pages, created_at, last_opened_at')
    .eq('id', bookId)
    .single();

  // Get current bookmark
  const bookmark = await getCurrentBookmark(bookId);

  // Get all bookmarks count
  const { count: bookmarkCount } = await supabase
    .from('bookmarks')
    .select('*', { count: 'exact', head: true })
    .eq('book_id', bookId)
    .eq('user_id', user.id);

  // Get annotations count
  const { count: annotationCount } = await supabase
    .from('annotations')
    .select('*', { count: 'exact', head: true })
    .eq('book_id', bookId)
    .eq('user_id', user.id);

  const progress = book?.total_pages
    ? ((bookmark?.page_number || 0) / book.total_pages) * 100
    : 0;

  return {
    currentPage: bookmark?.page_number || 0,
    totalPages: book?.total_pages || 0,
    progress: Math.round(progress),
    bookmarksCount: bookmarkCount || 0,
    annotationsCount: annotationCount || 0,
    lastOpened: book?.last_opened_at,
    createdAt: book?.created_at,
  };
}

// ==================== RECENTLY READ ====================

/**
 * Gets recently read books for the current user
 * @param limit - Maximum number of books to return
 * @returns Array of books with reading progress
 */
export async function getRecentlyReadBooks(limit: number = 10) {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data, error } = await supabase
    .from('books')
    .select(
      `
      *,
      bookmarks!inner (
        page_number,
        is_current,
        updated_at
      )
    `
    )
    .eq('user_id', user.id)
    .eq('bookmarks.is_current', true)
    .order('last_opened_at', { ascending: false })
    .limit(limit);

  if (error) throw error;

  return (data as any[]).map((book) => ({
    ...book,
    currentPage: book.bookmarks?.[0]?.page_number || 0,
    progress: book.total_pages
      ? Math.round(
          ((book.bookmarks?.[0]?.page_number || 0) / book.total_pages) *
            100
        )
      : 0,
  }));
}

// ==================== READING SESSIONS ====================

/**
 * Tracks a reading session
 * @param bookId - Book ID
 * @param startPage - Starting page
 * @param endPage - Ending page
 * @param durationSeconds - Session duration in seconds
 */
export async function trackReadingSession(
  bookId: string,
  startPage: number,
  endPage: number,
  durationSeconds: number
): Promise<void> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  // This would insert into a reading_sessions table if we add one
  // For now, we'll just update the bookmark
  await updateReadingProgress(bookId, endPage);
}

// ==================== EXPORT ====================

export type { Bookmark, BookmarkInsert, BookmarkUpdate };
