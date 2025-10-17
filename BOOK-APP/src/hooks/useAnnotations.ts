import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { annotationsService } from '@/services/annotations';
import { useAuthStore } from '@/store/authStore';
import type { AnnotationInsert, AnnotationUpdate } from '@/types/app';
import { useEffect } from 'react';

export function useAnnotations(bookId: string) {
  const user = useAuthStore(state => state.user);
  const queryClient = useQueryClient();

  const annotationsQuery = useQuery({
    queryKey: ['annotations', bookId],
    queryFn: () => annotationsService.getAnnotations(bookId),
    enabled: !!bookId && !!user,
  });

  // Real-time subscription
  useEffect(() => {
    if (!bookId) return;

    const channel = annotationsService.subscribeToAnnotations(bookId, annotation => {
      queryClient.invalidateQueries({ queryKey: ['annotations', bookId] });
    });

    return () => {
      channel.unsubscribe();
    };
  }, [bookId, queryClient]);

  const createAnnotationMutation = useMutation({
    mutationFn: (annotation: AnnotationInsert) =>
      annotationsService.createAnnotation(annotation),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', bookId] });
    },
  });

  const updateAnnotationMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: AnnotationUpdate }) =>
      annotationsService.updateAnnotation(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', bookId] });
    },
  });

  const deleteAnnotationMutation = useMutation({
    mutationFn: (id: string) => annotationsService.deleteAnnotation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', bookId] });
    },
  });

  return {
    annotations: annotationsQuery.data ?? [],
    isLoading: annotationsQuery.isLoading,
    error: annotationsQuery.error,
    refetch: annotationsQuery.refetch,
    createAnnotation: createAnnotationMutation.mutate,
    updateAnnotation: updateAnnotationMutation.mutate,
    deleteAnnotation: deleteAnnotationMutation.mutate,
    isCreating: createAnnotationMutation.isPending,
    isUpdating: updateAnnotationMutation.isPending,
    isDeleting: deleteAnnotationMutation.isPending,
  };
}
