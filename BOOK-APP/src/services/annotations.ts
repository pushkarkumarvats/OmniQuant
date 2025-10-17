import { supabase } from './supabase';
import type { Annotation, AnnotationInsert, AnnotationUpdate } from '@/types/app';

export const annotationsService = {
  async getAnnotations(bookId: string) {
    const { data, error } = await supabase
      .from('annotations')
      .select('*')
      .eq('book_id', bookId)
      .order('page', { ascending: true })
      .order('created_at', { ascending: true });

    if (error) throw error;
    return data as Annotation[];
  },

  async getAnnotationsByPage(bookId: string, page: number) {
    const { data, error } = await supabase
      .from('annotations')
      .select('*')
      .eq('book_id', bookId)
      .eq('page', page)
      .order('created_at', { ascending: true });

    if (error) throw error;
    return data as Annotation[];
  },

  async createAnnotation(annotation: AnnotationInsert) {
    const { data, error } = await supabase
      .from('annotations')
      .insert(annotation)
      .select()
      .single();

    if (error) throw error;
    return data as Annotation;
  },

  async updateAnnotation(id: string, updates: AnnotationUpdate) {
    const { data, error } = await supabase
      .from('annotations')
      .update(updates)
      .eq('id', id)
      .select()
      .single();

    if (error) throw error;
    return data as Annotation;
  },

  async deleteAnnotation(id: string) {
    const { error } = await supabase.from('annotations').delete().eq('id', id);

    if (error) throw error;
  },

  async subscribeToAnnotations(bookId: string, callback: (annotation: Annotation) => void) {
    const channel = supabase
      .channel(`book:${bookId}:annotations`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'annotations',
          filter: `book_id=eq.${bookId}`,
        },
        payload => {
          if (payload.new) {
            callback(payload.new as Annotation);
          }
        }
      )
      .subscribe();

    return channel;
  },
};
