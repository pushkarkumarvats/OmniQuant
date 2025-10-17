# 📘 BookFlow Developer Guide

## Welcome, Developer! 👋

This guide will help you understand the codebase, contribute effectively, and build new features. The codebase is designed to be **scalable**, **maintainable**, and **easy to understand**.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Architecture Overview](#architecture-overview)
4. [Code Patterns](#code-patterns)
5. [Adding New Features](#adding-new-features)
6. [Testing](#testing)
7. [Performance](#performance)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
npm install

# Start Supabase local development
supabase start

# Copy environment variables
cp .env.example .env
# Edit .env with your Supabase keys from `supabase status`

# Generate TypeScript types from database
npm run supabase:gen-types
```

### Run the App
```bash
# Start development server
npm start

# Run on specific platform
npm run ios      # iOS simulator
npm run android  # Android emulator
npm run web      # Web browser
```

---

## 📁 Project Structure

```
src/
├── app/                    # Expo Router screens (file-based routing)
│   ├── (auth)/            # Authentication screens
│   ├── (tabs)/            # Main tab navigation
│   └── reader/            # PDF reader screen
│
├── components/            # Reusable UI components
│   ├── BookCard.tsx       # Book display card
│   ├── ErrorBoundary.tsx  # Error handling
│   ├── LoadingSpinner.tsx # Loading states
│   └── EmptyState.tsx     # Empty state UI
│
├── services/              # Backend integration layer
│   ├── supabase.ts        # Supabase client setup
│   ├── books.ts           # Book CRUD operations
│   ├── annotations.ts     # Annotation management
│   ├── bookmarks.ts       # Reading progress
│   ├── collections.ts     # Collection management
│   ├── storage.ts         # File upload/download
│   └── sync.ts            # Real-time synchronization
│
├── hooks/                 # Custom React hooks
│   ├── useBooks.ts        # Book data queries
│   ├── useAnnotations.ts  # Annotation queries
│   ├── useBookmarks.ts    # Bookmark queries
│   └── useCollections.ts  # Collection queries
│
├── store/                 # Global state management
│   └── authStore.ts       # Authentication state (Zustand)
│
├── types/                 # TypeScript definitions
│   ├── database.ts        # Generated from Supabase schema
│   └── app.ts             # App-specific types
│
└── utils/                 # Utility functions
    ├── constants.ts       # App constants
    ├── errors.ts          # Error handling
    ├── format.ts          # Formatting functions
    ├── validation.ts      # Input validation
    ├── performance.ts     # Performance optimizations
    ├── logger.ts          # Centralized logging
    ├── analytics.ts       # Event tracking
    └── offlineStorage.ts  # Offline data management
```

---

## 🏗️ Architecture Overview

### Data Flow

```
┌─────────────┐
│   Screen    │  ← User Interface (React Native)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│ Custom Hook │  ← Data Management (React Query)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Service   │  ← Business Logic
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Supabase   │  ← Backend (Database, Auth, Storage)
└─────────────┘
```

### Key Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | React Native + Expo | Cross-platform interface |
| **Navigation** | Expo Router | File-based routing |
| **State (Client)** | Zustand | Global state management |
| **State (Server)** | React Query | Data fetching & caching |
| **Backend** | Supabase | Database, Auth, Storage, Realtime |
| **Validation** | Zod | Schema validation |
| **Testing** | Jest + RTL | Unit & integration tests |

---

## 🎯 Code Patterns

### 1. Service Layer Pattern

All backend interactions go through services in `src/services/`:

```typescript
// ✅ GOOD: Use service functions
import { getBooks, createBook } from '@/services/books';

const books = await getBooks();
const newBook = await createBook(bookData);
```

```typescript
// ❌ BAD: Don't access Supabase directly in components
import { supabase } from '@/services/supabase';

const { data } = await supabase.from('books').select();
```

**Why?** Centralizes business logic, makes testing easier, and allows for caching.

---

### 2. React Query Hook Pattern

Use custom hooks for data fetching:

```typescript
// ✅ GOOD: Use custom hooks
import { useBooks, useCreateBook } from '@/hooks/useBooks';

function LibraryScreen() {
  const { data: books, isLoading, error } = useBooks();
  const { mutate: createBook } = useCreateBook();

  // Component logic...
}
```

**Why?** Automatic caching, refetching, and state management.

---

### 3. Error Handling Pattern

Always wrap components with ErrorBoundary:

```typescript
import { ErrorBoundary } from '@/components/ErrorBoundary';

export default function MyScreen() {
  return (
    <ErrorBoundary>
      <MyComponent />
    </ErrorBoundary>
  );
}
```

Use try-catch in async functions:

```typescript
try {
  await someAsyncOperation();
  logger.info('Operation successful');
} catch (error) {
  logger.error('Operation failed', error);
  // Show user-friendly error message
}
```

---

### 4. Validation Pattern

Use Zod schemas for validation:

```typescript
import { z } from 'zod';

const bookSchema = z.object({
  title: z.string().min(1).max(500),
  author: z.string().optional(),
  totalPages: z.number().positive().optional(),
});

// Validate data
const result = bookSchema.safeParse(data);
if (!result.success) {
  // Handle validation errors
  console.error(result.error.errors);
}
```

---

### 5. Performance Pattern

Use memoization and debouncing:

```typescript
import { debounce, memoize } from '@/utils/performance';

// Debounce search input
const debouncedSearch = debounce((query: string) => {
  searchBooks(query);
}, 300);

// Memoize expensive calculations
const calculateStats = memoize((data) => {
  // Expensive calculation
  return stats;
});
```

---

## 🎨 Adding New Features

### Step-by-Step Guide

#### 1. Create a Service Function

```typescript
// src/services/myFeature.ts
import { supabase } from './supabase';

export async function getMyData() {
  const { data, error } = await supabase
    .from('my_table')
    .select('*');

  if (error) throw error;
  return data;
}
```

#### 2. Create a React Query Hook

```typescript
// src/hooks/useMyFeature.ts
import { useQuery } from '@tanstack/react-query';
import { getMyData } from '@/services/myFeature';

export function useMyData() {
  return useQuery({
    queryKey: ['myData'],
    queryFn: getMyData,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
```

#### 3. Create a Screen Component

```typescript
// src/app/(tabs)/myFeature.tsx
import { useMyData } from '@/hooks/useMyFeature';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { EmptyState } from '@/components/EmptyState';

export default function MyFeatureScreen() {
  const { data, isLoading, error } = useMyData();

  if (isLoading) return <LoadingSpinner />;
  if (error) return <Text>Error: {error.message}</Text>;
  if (!data?.length) return <EmptyState title="No data" />;

  return (
    <FlatList
      data={data}
      renderItem={({ item }) => <MyCard item={item} />}
    />
  );
}
```

#### 4. Add Tests

```typescript
// src/__tests__/hooks/useMyFeature.test.ts
import { renderHook } from '@testing-library/react-native';
import { useMyData } from '@/hooks/useMyFeature';

describe('useMyData', () => {
  it('should fetch data', async () => {
    const { result, waitFor } = renderHook(() => useMyData());

    await waitFor(() => result.current.isSuccess);

    expect(result.current.data).toBeDefined();
  });
});
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
npm test

# Run in watch mode
npm run test:watch

# Generate coverage report
npm run test:coverage

# Run specific test file
npm test MyComponent.test.ts
```

### Writing Tests

```typescript
import { render, fireEvent } from '@testing-library/react-native';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  it('should render correctly', () => {
    const { getByText } = render(<MyComponent />);
    expect(getByText('Hello')).toBeTruthy();
  });

  it('should handle button press', () => {
    const mockFn = jest.fn();
    const { getByTestId } = render(<MyComponent onPress={mockFn} />);

    fireEvent.press(getByTestId('my-button'));
    expect(mockFn).toHaveBeenCalled();
  });
});
```

---

## ⚡ Performance

### Optimization Techniques

#### 1. Debounce User Input
```typescript
import { debounce } from '@/utils/performance';

const handleSearch = debounce((query: string) => {
  searchBooks(query);
}, 300);
```

#### 2. Memoize Expensive Calculations
```typescript
import { useMemo } from 'react';

const sortedBooks = useMemo(() => {
  return books.sort((a, b) => a.title.localeCompare(b.title));
}, [books]);
```

#### 3. Use React Query Caching
```typescript
// Data stays fresh for 5 minutes
queryKey: ['books'],
staleTime: 5 * 60 * 1000,
```

#### 4. Optimize Images
```typescript
<Image
  source={{ uri: coverUrl }}
  resizeMode="cover"
  style={{ width: 100, height: 150 }}
/>
```

#### 5. Virtual Lists for Large Data
```typescript
<FlatList
  data={items}
  renderItem={renderItem}
  windowSize={10}
  maxToRenderPerBatch={10}
  removeClippedSubviews={true}
/>
```

---

## 📋 Common Tasks

### Add a New Database Table

1. **Create migration:**
```bash
supabase migration new add_my_table
```

2. **Edit migration file:**
```sql
CREATE TABLE my_table (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users NOT NULL,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add RLS
ALTER TABLE my_table ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own data"
  ON my_table FOR SELECT
  USING (auth.uid() = user_id);
```

3. **Apply migration:**
```bash
supabase db reset
```

4. **Generate types:**
```bash
npm run supabase:gen-types
```

---

### Add Analytics Event

```typescript
import { analytics } from '@/utils/analytics';

// Track event
analytics.trackEvent('book_opened', {
  book_id: bookId,
  book_title: title,
});

// Track screen view
analytics.trackScreen('LibraryScreen');

// Track error
analytics.trackError(error, { context: 'book_upload' });
```

---

### Add Logging

```typescript
import { logger } from '@/utils/logger';

logger.debug('Debug message', { data });  // Development only
logger.info('Info message', { data });    // Always logged
logger.warn('Warning message', { data });  // Warning
logger.error('Error message', error);      // Error + remote logging
```

---

### Add Offline Support

```typescript
import { addToSyncQueue } from '@/utils/offlineStorage';

// Queue action for later sync
await addToSyncQueue({
  type: 'create',
  entity: 'annotation',
  data: annotationData,
});
```

---

## 🐛 Troubleshooting

### Common Issues

#### Module not found
```bash
# Clear cache and reinstall
rm -rf node_modules
npm install
npx expo start -c
```

#### Supabase connection failed
```bash
# Check Supabase is running
supabase status

# Verify .env file has correct keys
cat .env
```

#### TypeScript errors
```bash
# Regenerate database types
npm run supabase:gen-types

# Run type check
npm run type-check
```

#### Test failures
```bash
# Clear Jest cache
jest --clearCache

# Run specific test
npm test MyComponent.test.ts
```

---

## ✅ Best Practices

### Code Style

1. **Use TypeScript strictly**
   - No `any` types
   - Define interfaces for all data
   - Use generics when appropriate

2. **Follow naming conventions**
   - Components: `PascalCase` (e.g., `BookCard`)
   - Functions: `camelCase` (e.g., `getBooks`)
   - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_FILE_SIZE`)
   - Hooks: `use` prefix (e.g., `useBooks`)

3. **Document complex logic**
   ```typescript
   /**
    * Calculates reading progress percentage
    * @param currentPage - Current page number (1-indexed)
    * @param totalPages - Total pages in book
    * @returns Progress percentage (0-100)
    */
   function calculateProgress(currentPage: number, totalPages: number): number {
     return Math.round((currentPage / totalPages) * 100);
   }
   ```

4. **Keep components small**
   - One component per file
   - Extract complex logic to hooks
   - Max 200-300 lines per file

5. **Use path aliases**
   ```typescript
   // ✅ GOOD
   import { getBooks } from '@/services/books';
   
   // ❌ BAD
   import { getBooks } from '../../../services/books';
   ```

### Git Workflow

1. **Branch naming**
   - `feature/feature-name`
   - `fix/bug-description`
   - `docs/documentation-update`

2. **Commit messages**
   ```
   feat: add dark mode support
   fix: resolve PDF loading issue
   docs: update developer guide
   refactor: optimize book list rendering
   test: add tests for authentication
   ```

3. **Pull requests**
   - Write clear description
   - Link related issues
   - Request code review
   - Ensure CI passes

### Performance

1. **Lazy load screens**
2. **Optimize images**
3. **Use pagination for large lists**
4. **Cache API responses**
5. **Debounce expensive operations**

### Security

1. **Never commit sensitive data**
2. **Use environment variables**
3. **Validate all user input**
4. **Implement Row Level Security**
5. **Use signed URLs for files**

---

## 📚 Additional Resources

- **Expo Docs:** https://docs.expo.dev/
- **React Native:** https://reactnative.dev/
- **Supabase Docs:** https://supabase.com/docs
- **React Query:** https://tanstack.com/query/latest
- **TypeScript:** https://www.typescriptlang.org/docs/

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linter: `npm run lint`
6. Run tests: `npm test`
7. Commit with conventional message
8. Push and create pull request

---

## 💬 Getting Help

- **Documentation:** Check `docs/` folder
- **Code examples:** Browse `src/` for patterns
- **Issues:** Search existing issues first
- **Questions:** Ask in team chat

---

**Happy coding! 🚀**

Built with ❤️ by the BookFlow team
