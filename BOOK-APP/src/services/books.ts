import { supabase } from './supabase';
import type { Book, BookInsert, BookUpdate } from '@/types/app';

export const booksService = {
  async getBooks(userId: string) {
    const { data, error } = await supabase
      .from('books')
      .select('*')
      .eq('owner_id', userId)
      .order('last_opened_at', { ascending: false, nullsFirst: false });

    if (error) throw error;
    return data as Book[];
  },

  async getBook(id: string) {
    const { data, error } = await supabase
      .from('books')
      .select('*')
      .eq('id', id)
      .single();

    if (error) throw error;
    return data as Book;
  },

  async createBook(book: BookInsert) {
    const { data, error } = await supabase
      .from('books')
      .insert(book)
      .select()
      .single();

    if (error) throw error;
    return data as Book;
  },

  async updateBook(id: string, updates: BookUpdate) {
    const { data, error } = await supabase
      .from('books')
      .update(updates)
      .eq('id', id)
      .select()
      .single();

    if (error) throw error;
    return data as Book;
  },

  async deleteBook(id: string) {
    const { error } = await supabase.from('books').delete().eq('id', id);

    if (error) throw error;
  },

  async updateLastOpened(id: string) {
    const { error } = await supabase
      .from('books')
      .update({ last_opened_at: new Date().toISOString() })
      .eq('id', id);

    if (error) throw error;
  },

  async searchBooks(userId: string, query: string) {
    const { data, error } = await supabase.rpc('search_books', {
      search_query: query,
      user_id_filter: userId,
      result_limit: 20,
    });

    if (error) throw error;
    return data;
  },
};
