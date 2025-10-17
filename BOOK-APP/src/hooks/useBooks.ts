import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { booksService } from '@/services/books';
import { useAuthStore } from '@/store/authStore';
import type { BookInsert, BookUpdate } from '@/types/app';

export function useBooks() {
  const user = useAuthStore(state => state.user);
  const queryClient = useQueryClient();

  const booksQuery = useQuery({
    queryKey: ['books', user?.id],
    queryFn: () => booksService.getBooks(user!.id),
    enabled: !!user,
  });

  const createBookMutation = useMutation({
    mutationFn: (book: BookInsert) => booksService.createBook(book),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['books'] });
    },
  });

  const updateBookMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: BookUpdate }) =>
      booksService.updateBook(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['books'] });
    },
  });

  const deleteBookMutation = useMutation({
    mutationFn: (id: string) => booksService.deleteBook(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['books'] });
    },
  });

  return {
    books: booksQuery.data ?? [],
    isLoading: booksQuery.isLoading,
    error: booksQuery.error,
    refetch: booksQuery.refetch,
    createBook: createBookMutation.mutate,
    updateBook: updateBookMutation.mutate,
    deleteBook: deleteBookMutation.mutate,
    isCreating: createBookMutation.isPending,
    isUpdating: updateBookMutation.isPending,
    isDeleting: deleteBookMutation.isPending,
  };
}

export function useBook(id: string) {
  return useQuery({
    queryKey: ['book', id],
    queryFn: () => booksService.getBook(id),
    enabled: !!id,
  });
}
