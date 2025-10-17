/**
 * Collections Service
 * Handles book collections, folders, and organization features
 * @module services/collections
 */

import { supabase } from './supabase';
import type { Database } from '@/types/database';

type Collection = Database['public']['Tables']['collections']['Row'];
type CollectionInsert = Database['public']['Tables']['collections']['Insert'];
type CollectionUpdate = Database['public']['Tables']['collections']['Update'];

type BookCollection =
  Database['public']['Tables']['book_collections']['Row'];

// ==================== COLLECTIONS CRUD ====================

/**
 * Fetches all collections for the current user
 * @returns Array of collections with book counts
 */
export async function getCollections(): Promise<
  (Collection & { book_count: number })[]
> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const { data, error } = await supabase
    .from('collections')
    .select(
      `
      *,
      book_collections (count)
    `
    )
    .eq('user_id', user.id)
    .order('name', { ascending: true });

  if (error) throw error;

  return (data as any[]).map((collection) => ({
    ...collection,
    book_count: collection.book_collections?.[0]?.count || 0,
  }));
}

/**
 * Fetches a single collection by ID with all its books
 * @param collectionId - Collection ID
 * @returns Collection with books
 */
export async function getCollectionWithBooks(collectionId: string) {
  const { data, error } = await supabase
    .from('collections')
    .select(
      `
      *,
      book_collections (
        book_id,
        books (
          *
        )
      )
    `
    )
    .eq('id', collectionId)
    .single();

  if (error) throw error;

  return {
    ...data,
    books: (data as any).book_collections?.map(
      (bc: any) => bc.books
    ) || [],
  };
}

/**
 * Creates a new collection
 * @param name - Collection name
 * @param description - Optional description
 * @param color - Optional color hex code
 * @returns Created collection
 */
export async function createCollection(
  name: string,
  description?: string,
  color?: string
): Promise<Collection> {
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) throw new Error('Not authenticated');

  const collectionData: CollectionInsert = {
    user_id: user.id,
    name,
    description,
    color: color || '#6366F1', // Default indigo color
  };

  const { data, error } = await supabase
    .from('collections')
    .insert(collectionData)
    .select()
    .single();

  if (error) throw error;
  return data;
}

/**
 * Updates an existing collection
 * @param collectionId - Collection ID
 * @param updates - Fields to update
 * @returns Updated collection
 */
export async function updateCollection(
  collectionId: string,
  updates: CollectionUpdate
): Promise<Collection> {
  const { data, error } = await supabase
    .from('collections')
    .update(updates)
    .eq('id', collectionId)
    .select()
    .single();

  if (error) throw error;
  return data;
}

/**
 * Deletes a collection (books remain, just the collection is removed)
 * @param collectionId - Collection ID
 */
export async function deleteCollection(
  collectionId: string
): Promise<void> {
  const { error } = await supabase
    .from('collections')
    .delete()
    .eq('id', collectionId);

  if (error) throw error;
}

// ==================== BOOK-COLLECTION RELATIONSHIPS ====================

/**
 * Adds a book to a collection
 * @param bookId - Book ID
 * @param collectionId - Collection ID
 * @returns Created relationship
 */
export async function addBookToCollection(
  bookId: string,
  collectionId: string
): Promise<BookCollection> {
  const { data, error } = await supabase
    .from('book_collections')
    .insert({
      book_id: bookId,
      collection_id: collectionId,
    })
    .select()
    .single();

  if (error) {
    // Handle duplicate entry
    if (error.code === '23505') {
      throw new Error('Book is already in this collection');
    }
    throw error;
  }

  return data;
}

/**
 * Removes a book from a collection
 * @param bookId - Book ID
 * @param collectionId - Collection ID
 */
export async function removeBookFromCollection(
  bookId: string,
  collectionId: string
): Promise<void> {
  const { error } = await supabase
    .from('book_collections')
    .delete()
    .eq('book_id', bookId)
    .eq('collection_id', collectionId);

  if (error) throw error;
}

/**
 * Gets all collections for a specific book
 * @param bookId - Book ID
 * @returns Array of collections containing this book
 */
export async function getBookCollections(
  bookId: string
): Promise<Collection[]> {
  const { data, error } = await supabase
    .from('book_collections')
    .select(
      `
      collections (*)
    `
    )
    .eq('book_id', bookId);

  if (error) throw error;

  return (data as any[]).map((item) => item.collections);
}

/**
 * Adds multiple books to a collection at once
 * @param bookIds - Array of book IDs
 * @param collectionId - Collection ID
 * @returns Array of created relationships
 */
export async function addBooksToCollection(
  bookIds: string[],
  collectionId: string
): Promise<BookCollection[]> {
  const inserts = bookIds.map((bookId) => ({
    book_id: bookId,
    collection_id: collectionId,
  }));

  const { data, error } = await supabase
    .from('book_collections')
    .insert(inserts)
    .select();

  if (error) throw error;
  return data;
}

/**
 * Moves a book from one collection to another
 * @param bookId - Book ID
 * @param fromCollectionId - Source collection ID
 * @param toCollectionId - Target collection ID
 */
export async function moveBookBetweenCollections(
  bookId: string,
  fromCollectionId: string,
  toCollectionId: string
): Promise<void> {
  // Remove from old collection
  await removeBookFromCollection(bookId, fromCollectionId);

  // Add to new collection
  await addBookToCollection(bookId, toCollectionId);
}

// ==================== SMART COLLECTIONS ====================

/**
 * Creates default collections for a new user
 * @returns Array of created collections
 */
export async function createDefaultCollections(): Promise<Collection[]> {
  const defaultCollections = [
    {
      name: 'Reading',
      description: 'Books I'm currently reading',
      color: '#10B981', // Green
    },
    {
      name: 'To Read',
      description: 'Books on my reading list',
      color: '#F59E0B', // Amber
    },
    {
      name: 'Favorites',
      description: 'My favorite books',
      color: '#EF4444', // Red
    },
    {
      name: 'Reference',
      description: 'Reference materials and textbooks',
      color: '#6366F1', // Indigo
    },
  ];

  const promises = defaultCollections.map((col) =>
    createCollection(col.name, col.description, col.color)
  );

  return Promise.all(promises);
}

/**
 * Gets collection statistics
 * @param collectionId - Collection ID
 * @returns Statistics object
 */
export async function getCollectionStats(collectionId: string) {
  const { data, error } = await supabase.rpc(
    'get_collection_stats',
    { collection_id: collectionId }
  );

  if (error) throw error;

  return data;
}

// ==================== COLLECTION SORTING ====================

/**
 * Updates the sort order of collections
 * @param orderedIds - Array of collection IDs in desired order
 */
export async function updateCollectionOrder(
  orderedIds: string[]
): Promise<void> {
  const updates = orderedIds.map((id, index) => ({
    id,
    sort_order: index,
  }));

  const { error } = await supabase
    .from('collections')
    .upsert(updates);

  if (error) throw error;
}

// ==================== EXPORT ====================

export type { Collection, CollectionInsert, CollectionUpdate, BookCollection };
