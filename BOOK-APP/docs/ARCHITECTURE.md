# Technical Architecture
## BookFlow - Cross-Platform Book & PDF Editor

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Client Layer                         │
├──────────────┬──────────────┬─────────────────────────┤
│   iOS App    │ Android App  │      Web App            │
│ (React Native) (React Native) (React Native Web)       │
└──────┬───────┴──────┬───────┴──────┬─────────────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
            ┌─────────▼──────────┐
            │   API Gateway      │
            │  (Supabase REST)   │
            └─────────┬──────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│  Auth       │ │ Postgres │ │  Storage    │
│ (Supabase)  │ │  (RLS)   │ │ (Supabase)  │
└─────────────┘ └──────────┘ └─────────────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
            ┌─────────▼──────────┐
            │  Realtime (WS)     │
            │  Edge Functions    │
            └────────────────────┘
```

---

## 2. Frontend Architecture

### 2.1 Technology Choice: **React Native + Expo**

**Rationale:**
- ✅ Single codebase for iOS, Android, Web
- ✅ Expo provides excellent developer experience
- ✅ Rich ecosystem for PDF rendering (react-native-pdf, pdf.js)
- ✅ Supabase has official React Native support
- ✅ Over-the-air updates via EAS
- ⚠️ Alternative considered: Flutter (rejected due to web maturity & Supabase integration)

### 2.2 Frontend Stack

```typescript
// Core
- React Native 0.74+
- Expo SDK 51+
- TypeScript 5.0+
- Expo Router (file-based navigation)

// State Management
- Zustand (global state)
- React Query (server state, caching)
- AsyncStorage (persistence)

// PDF Rendering
- pdf.js (Web)
- react-native-pdf (iOS/Android)

// Annotations & Sync
- Yjs (CRDT for annotations)
- y-indexeddb (local persistence)
- y-websocket (sync provider)

// UI Components
- React Native Paper (Material Design)
- react-native-svg (drawings)
- react-native-gesture-handler (pan, zoom)

// Dev Tools
- ESLint + Prettier
- Jest + React Native Testing Library
- Detox (E2E testing)
```

### 2.3 Project Structure

```
src/
├── app/                    # Expo Router (file-based routing)
│   ├── (auth)/
│   │   ├── login.tsx
│   │   └── signup.tsx
│   ├── (tabs)/
│   │   ├── _layout.tsx    # Bottom tab navigator
│   │   ├── index.tsx      # Library
│   │   ├── search.tsx
│   │   ├── upload.tsx
│   │   └── profile.tsx
│   ├── reader/[id].tsx    # Dynamic route
│   └── _layout.tsx
│
├── components/
│   ├── ui/                # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   └── Input.tsx
│   └── features/          # Feature-specific components
│       ├── BookCard.tsx
│       ├── AnnotationToolbar.tsx
│       └── PDFViewer.tsx
│
├── features/              # Feature modules (business logic)
│   ├── auth/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types.ts
│   ├── library/
│   ├── reader/
│   ├── annotations/
│   └── sync/
│
├── services/              # API & external services
│   ├── supabase.ts
│   ├── api.ts
│   └── storage.ts
│
├── store/                 # Zustand stores
│   ├── authStore.ts
│   ├── libraryStore.ts
│   └── readerStore.ts
│
├── hooks/                 # Custom hooks
│   ├── useAuth.ts
│   ├── useBooks.ts
│   └── useAnnotations.ts
│
├── types/                 # TypeScript types
│   ├── database.ts        # Generated from Supabase
│   └── app.ts
│
└── utils/                 # Utilities
    ├── pdf.ts
    ├── sync.ts
    └── storage.ts
```

### 2.4 State Management Strategy

```typescript
// Global State (Zustand)
interface AppState {
  user: User | null;
  theme: 'light' | 'dark';
  selectedBook: Book | null;
  syncStatus: SyncStatus;
}

// Server State (React Query)
const { data: books } = useQuery({
  queryKey: ['books'],
  queryFn: fetchBooks,
  staleTime: 5 * 60 * 1000, // 5 minutes
});

// Local State (React useState)
const [selectedPage, setSelectedPage] = useState(1);
```

---

## 3. Backend Architecture

### 3.1 Supabase Services

**Supabase Auth**
- Email/password + OAuth (Google, Apple)
- JWT token management
- Session handling

**Postgres Database**
- Primary data store
- Full-text search (tsvector)
- Row Level Security (RLS)

**Supabase Storage**
- PDF file storage
- Signed URLs (1-hour expiry)
- 50GB free tier limit

**Supabase Realtime**
- WebSocket subscriptions
- Postgres CDC (Change Data Capture)
- Real-time annotation broadcast

**Edge Functions**
- OCR processing (Tesseract.js)
- PDF metadata extraction
- Thumbnail generation
- Webhook handlers

### 3.2 Database Schema

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users table (extends auth.users)
CREATE TABLE public.users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  preferences JSONB DEFAULT '{
    "theme": "light",
    "defaultView": "grid",
    "fontSize": 16
  }'::jsonb,
  storage_used_bytes BIGINT DEFAULT 0,
  storage_limit_bytes BIGINT DEFAULT 104857600, -- 100MB free tier
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Books table
CREATE TABLE public.books (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  owner_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Metadata
  title TEXT NOT NULL,
  authors TEXT[] DEFAULT '{}',
  description TEXT,
  language TEXT DEFAULT 'en',
  isbn TEXT,
  publisher TEXT,
  published_date DATE,
  
  -- File info
  file_type TEXT NOT NULL CHECK (file_type IN ('pdf', 'epub', 'mobi')),
  storage_path TEXT NOT NULL,  -- Supabase Storage path
  file_size_bytes BIGINT NOT NULL,
  page_count INTEGER,
  
  -- Source
  source TEXT DEFAULT 'uploaded' CHECK (source IN ('uploaded', 'api', 'purchased')),
  source_url TEXT,
  
  -- Search
  text_content TEXT,  -- Extracted text
  text_indexed TSVECTOR,  -- Full-text search index
  
  -- Metadata
  metadata_json JSONB DEFAULT '{}'::jsonb,
  thumbnail_url TEXT,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  last_opened_at TIMESTAMPTZ
);

-- Generate tsvector index automatically
CREATE FUNCTION books_text_index_trigger() RETURNS trigger AS $$
BEGIN
  NEW.text_indexed := to_tsvector('english', COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.text_content, ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER books_text_index_update
  BEFORE INSERT OR UPDATE ON books
  FOR EACH ROW
  EXECUTE FUNCTION books_text_index_trigger();

-- Annotations table
CREATE TABLE public.annotations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Location
  page INTEGER NOT NULL,
  coords JSONB NOT NULL,  -- {x, y, width, height} normalized (0-1)
  
  -- Annotation data
  type TEXT NOT NULL CHECK (type IN ('highlight', 'underline', 'strikeout', 'note', 'drawing', 'text')),
  color TEXT DEFAULT '#FFEB3B',
  content TEXT,  -- Note text or drawing path data
  selected_text TEXT,  -- Text content if applicable
  
  -- Sync metadata
  device_id TEXT,
  crdt_state JSONB,  -- Yjs CRDT state
  version BIGINT DEFAULT 1,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bookmarks table (reading progress)
CREATE TABLE public.bookmarks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  page INTEGER NOT NULL,
  scroll_offset REAL DEFAULT 0,  -- Percentage (0-1)
  location_json JSONB,  -- Additional positioning data
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(user_id, book_id)  -- One bookmark per user per book
);

-- Book edits/versions table
CREATE TABLE public.book_edits (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  edit_type TEXT NOT NULL CHECK (edit_type IN ('page_reorder', 'page_rotate', 'page_delete', 'page_add', 'text_edit', 'merge')),
  payload JSONB NOT NULL,  -- Edit details (e.g., {from: 5, to: 10})
  base_version BIGINT,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Devices table (for sync tracking)
CREATE TABLE public.devices (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  device_name TEXT,
  device_info JSONB,  -- OS, version, etc.
  last_seen_at TIMESTAMPTZ DEFAULT NOW(),
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Collections/Tags table
CREATE TABLE public.collections (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  name TEXT NOT NULL,
  color TEXT DEFAULT '#2196F3',
  icon TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(user_id, name)
);

-- Many-to-many: books <-> collections
CREATE TABLE public.book_collections (
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  collection_id UUID NOT NULL REFERENCES public.collections(id) ON DELETE CASCADE,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  
  PRIMARY KEY (book_id, collection_id)
);

-- Indexes
CREATE INDEX idx_books_owner ON books(owner_id);
CREATE INDEX idx_books_text_search ON books USING GIN(text_indexed);
CREATE INDEX idx_annotations_book_page ON annotations(book_id, page);
CREATE INDEX idx_annotations_user ON annotations(user_id);
CREATE INDEX idx_bookmarks_user_book ON bookmarks(user_id, book_id);
CREATE INDEX idx_book_edits_book ON book_edits(book_id);

-- Row Level Security (RLS) Policies

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.books ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bookmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.book_edits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.book_collections ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view own profile"
  ON public.users FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.users FOR UPDATE
  USING (auth.uid() = id);

-- Books policies
CREATE POLICY "Users can view own books"
  ON public.books FOR SELECT
  USING (auth.uid() = owner_id);

CREATE POLICY "Users can insert own books"
  ON public.books FOR INSERT
  WITH CHECK (auth.uid() = owner_id);

CREATE POLICY "Users can update own books"
  ON public.books FOR UPDATE
  USING (auth.uid() = owner_id);

CREATE POLICY "Users can delete own books"
  ON public.books FOR DELETE
  USING (auth.uid() = owner_id);

-- Annotations policies
CREATE POLICY "Users can view own annotations"
  ON public.annotations FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own annotations"
  ON public.annotations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own annotations"
  ON public.annotations FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own annotations"
  ON public.annotations FOR DELETE
  USING (auth.uid() = user_id);

-- Similar policies for other tables...

-- Functions & Triggers

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER books_updated_at
  BEFORE UPDATE ON books
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER annotations_updated_at
  BEFORE UPDATE ON annotations
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

-- Update storage usage on book upload/delete
CREATE OR REPLACE FUNCTION update_storage_usage()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE users
    SET storage_used_bytes = storage_used_bytes + NEW.file_size_bytes
    WHERE id = NEW.owner_id;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE users
    SET storage_used_bytes = GREATEST(0, storage_used_bytes - OLD.file_size_bytes)
    WHERE id = OLD.owner_id;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER books_storage_usage
  AFTER INSERT OR DELETE ON books
  FOR EACH ROW
  EXECUTE FUNCTION update_storage_usage();
```

### 3.3 Row Level Security Examples

```sql
-- Example: Shared books (future feature)
CREATE TABLE public.book_shares (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  shared_by UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  shared_with UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  permission TEXT NOT NULL CHECK (permission IN ('read', 'annotate', 'edit')),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Update books policy to include shared books
CREATE POLICY "Users can view owned or shared books"
  ON public.books FOR SELECT
  USING (
    auth.uid() = owner_id OR
    EXISTS (
      SELECT 1 FROM book_shares
      WHERE book_id = books.id
      AND shared_with = auth.uid()
    )
  );
```

---

## 4. API Design

### 4.1 REST Endpoints

**Base URL:** `https://your-project.supabase.co/rest/v1`

```typescript
// Authentication
POST   /auth/v1/signup                    # Create account
POST   /auth/v1/token?grant_type=password # Login
POST   /auth/v1/logout                    # Logout
POST   /auth/v1/recover                   # Password reset

// Books
GET    /books                             # List user's books
GET    /books?select=*,annotations(*)     # Include annotations
POST   /books                             # Create book entry
GET    /books/:id                         # Get book details
PATCH  /books/:id                         # Update metadata
DELETE /books/:id                         # Delete book

// Annotations
GET    /annotations?book_id=eq.:id        # Get book annotations
POST   /annotations                       # Create annotation
PATCH  /annotations/:id                   # Update annotation
DELETE /annotations/:id                   # Delete annotation

// Bookmarks
GET    /bookmarks?book_id=eq.:id&user_id=eq.:uid
POST   /bookmarks                         # Upsert bookmark
```

### 4.2 Custom Edge Functions

```typescript
// supabase/functions/process-upload/index.ts
Deno.serve(async (req) => {
  const { bookId } = await req.json();
  
  // 1. Download PDF from Storage
  // 2. Extract metadata (title, author, page count)
  // 3. Generate thumbnail (first page)
  // 4. Extract text for search indexing
  // 5. Update books table
  
  return new Response(JSON.stringify({ success: true }));
});

// supabase/functions/ocr-document/index.ts
Deno.serve(async (req) => {
  const { bookId, pages } = await req.json();
  
  // 1. Download specified pages
  // 2. Run Tesseract OCR
  // 3. Update text_content in books table
  
  return new Response(JSON.stringify({ jobId: '...' }));
});

// supabase/functions/search-external/index.ts
Deno.serve(async (req) => {
  const { query } = await req.json();
  
  // Proxy to Google Books API or Open Library
  const results = await fetch(`https://www.googleapis.com/books/v1/volumes?q=${query}`);
  
  return new Response(JSON.stringify(results));
});
```

---

## 5. Sync Architecture

### 5.1 CRDT Implementation (Yjs)

```typescript
// Client-side
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';

// Create Yjs document for book annotations
const ydoc = new Y.Doc();
const annotations = ydoc.getMap('annotations');

// Connect to Supabase Realtime as sync provider
const provider = new WebsocketProvider(
  'wss://your-project.supabase.co/realtime/v1',
  `book-${bookId}`,
  ydoc,
  { params: { token: supabaseToken } }
);

// Local annotation changes automatically sync
annotations.set('annot-123', {
  type: 'highlight',
  page: 5,
  coords: { x: 0.1, y: 0.2, w: 0.5, h: 0.03 },
  color: '#FFEB3B',
});

// Listen for remote changes
annotations.observe((event) => {
  event.changes.keys.forEach((change, key) => {
    if (change.action === 'add' || change.action === 'update') {
      updateUIWithAnnotation(key, annotations.get(key));
    }
  });
});
```

### 5.2 Sync Flow Diagram

```
┌─────────────┐                    ┌──────────────┐
│  Client A   │                    │  Client B    │
└──────┬──────┘                    └──────┬───────┘
       │                                  │
       │ 1. Create annotation            │
       │    (saved locally)               │
       │                                  │
       │ 2. Send delta ───────────────┐  │
       │                              │  │
       │                              ▼  │
       │                        ┌─────────────┐
       │                        │  Supabase   │
       │                        │  Realtime   │
       │                        └──────┬──────┘
       │                               │
       │ 3. Ack ◄─────────────────────┤
       │                               │
       │                               │ 4. Broadcast
       │                               │
       │                               └──────────────► Client B
       │                                              │
       │                                              │ 5. Merge CRDT
       │                                              │    Update UI
```

### 5.3 Offline Queue

```typescript
interface PendingOperation {
  id: string;
  type: 'create' | 'update' | 'delete';
  table: 'annotations' | 'bookmarks' | 'book_edits';
  payload: any;
  timestamp: number;
  retries: number;
}

class SyncQueue {
  private queue: PendingOperation[] = [];
  
  async add(operation: PendingOperation) {
    this.queue.push(operation);
    await this.persist();
    this.processQueue();
  }
  
  private async processQueue() {
    if (!navigator.onLine) return;
    
    for (const op of this.queue) {
      try {
        await this.syncOperation(op);
        this.queue = this.queue.filter(o => o.id !== op.id);
      } catch (error) {
        op.retries++;
        if (op.retries > 5) {
          // Move to failed queue
          this.handleFailedOperation(op);
        }
      }
    }
    
    await this.persist();
  }
}
```

---

## 6. PDF Rendering Strategy

### 6.1 Platform-Specific Rendering

**Web:**
```typescript
// Using pdf.js
import * as pdfjsLib from 'pdfjs-dist';

const loadPDF = async (url: string) => {
  const pdf = await pdfjsLib.getDocument(url).promise;
  const page = await pdf.getPage(pageNumber);
  
  const viewport = page.getViewport({ scale: 1.5 });
  const canvas = document.getElementById('pdf-canvas');
  const context = canvas.getContext('2d');
  
  await page.render({ canvasContext: context, viewport }).promise;
};
```

**Mobile (iOS/Android):**
```typescript
// Using react-native-pdf
import Pdf from 'react-native-pdf';

<Pdf
  source={{ uri: pdfUrl }}
  onLoadComplete={(numberOfPages) => {
    console.log(`Pages: ${numberOfPages}`);
  }}
  onPageChanged={(page) => {
    saveBookmark(bookId, page);
  }}
  style={styles.pdf}
/>
```

### 6.2 Annotation Overlay

```typescript
// SVG overlay for annotations
const AnnotationLayer = ({ annotations, pageWidth, pageHeight }) => {
  return (
    <Svg
      width={pageWidth}
      height={pageHeight}
      style={{ position: 'absolute', top: 0, left: 0 }}
    >
      {annotations.map(annot => {
        if (annot.type === 'highlight') {
          return (
            <Rect
              key={annot.id}
              x={annot.coords.x * pageWidth}
              y={annot.coords.y * pageHeight}
              width={annot.coords.width * pageWidth}
              height={annot.coords.height * pageHeight}
              fill={annot.color}
              opacity={0.3}
            />
          );
        }
        // Handle other annotation types...
      })}
    </Svg>
  );
};
```

---

## 7. Security Architecture

### 7.1 Authentication Flow

```typescript
// Supabase Auth with JWT
const signUp = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
  });
  
  if (error) throw error;
  
  // JWT stored in secure storage
  // iOS: Keychain, Android: KeyStore, Web: httpOnly cookie
  return data.session;
};

// Auto-refresh token
supabase.auth.onAuthStateChange((event, session) => {
  if (event === 'TOKEN_REFRESHED') {
    // Update stored token
  }
});
```

### 7.2 File Access Control

```typescript
// Generate signed URL with short expiry
const getFileUrl = async (bookId: string) => {
  const { data, error } = await supabase
    .storage
    .from('books')
    .createSignedUrl(`${userId}/${bookId}.pdf`, 3600); // 1 hour
  
  return data.signedUrl;
};

// Storage bucket policies
CREATE POLICY "Users can upload own files"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'books' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Users can read own files"
  ON storage.objects FOR SELECT
  USING (
    bucket_id = 'books' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );
```

---

## 8. Performance Optimizations

### 8.1 Caching Strategy

```typescript
// React Query caching
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,      // 5 minutes
      cacheTime: 30 * 60 * 1000,     // 30 minutes
      refetchOnWindowFocus: false,
    },
  },
});

// Persistent cache (AsyncStorage)
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import { createAsyncStoragePersister } from '@tanstack/query-async-storage-persister';
import AsyncStorage from '@react-native-async-storage/async-storage';

const asyncStoragePersister = createAsyncStoragePersister({
  storage: AsyncStorage,
});
```

### 8.2 Virtual Scrolling (Large PDFs)

```typescript
// Only render visible pages
import { FlatList } from 'react-native';

<FlatList
  data={Array.from({ length: pageCount }, (_, i) => i + 1)}
  renderItem={({ item }) => <PDFPage pageNumber={item} />}
  keyExtractor={(item) => `page-${item}`}
  windowSize={3}  // Render 3 pages before/after visible
  initialNumToRender={1}
  maxToRenderPerBatch={2}
/>
```

---

## 9. Deployment Architecture

```
┌────────────────────────────────────────────────┐
│                                                │
│  Expo Application Services (EAS)               │
│  ├─ EAS Build (iOS/Android builds)            │
│  ├─ EAS Submit (App Store deployment)         │
│  └─ EAS Update (OTA updates)                  │
│                                                │
└────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────┐
│  Vercel (Web Hosting)                          │
│  └─ Next.js SSR + Static Export                │
└────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────┐
│  Supabase Cloud                                │
│  ├─ Postgres Database                          │
│  ├─ Storage (CDN)                              │
│  ├─ Auth Service                               │
│  ├─ Realtime (WebSockets)                      │
│  └─ Edge Functions (Deno Deploy)               │
└────────────────────────────────────────────────┘
```

---

## 10. Monitoring & Observability

```typescript
// Error tracking (Sentry)
import * as Sentry from '@sentry/react-native';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: __DEV__ ? 'development' : 'production',
  tracesSampleRate: 0.2,
});

// Analytics (PostHog)
import PostHog from 'posthog-react-native';

const posthog = new PostHog(API_KEY, {
  host: 'https://app.posthog.com',
});

// Track events
posthog.capture('book_uploaded', {
  book_id: bookId,
  file_size: fileSize,
  page_count: pageCount,
});
```

---

**Next:** See [API.md](./API.md) for detailed API documentation and [WIREFRAMES.md](./WIREFRAMES.md) for UI specifications.
