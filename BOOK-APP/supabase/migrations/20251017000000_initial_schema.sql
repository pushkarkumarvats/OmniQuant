-- BookFlow Database Schema
-- Migration: Initial setup
-- Created: 2025-10-17

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS "pgcrypto"; -- For encryption functions

-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE public.users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  
  -- Preferences
  preferences JSONB DEFAULT jsonb_build_object(
    'theme', 'light',
    'defaultView', 'grid',
    'fontSize', 16,
    'autoSync', true,
    'offlineMode', false
  ),
  
  -- Storage quota
  storage_used_bytes BIGINT DEFAULT 0,
  storage_limit_bytes BIGINT DEFAULT 104857600, -- 100MB free tier
  
  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- BOOKS TABLE
-- ============================================================================

CREATE TABLE public.books (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  owner_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Core metadata
  title TEXT NOT NULL,
  authors TEXT[] DEFAULT '{}',
  description TEXT,
  language TEXT DEFAULT 'en',
  isbn TEXT,
  publisher TEXT,
  published_date DATE,
  
  -- File information
  file_type TEXT NOT NULL CHECK (file_type IN ('pdf', 'epub', 'mobi')),
  storage_path TEXT NOT NULL UNIQUE,
  file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes > 0),
  page_count INTEGER CHECK (page_count > 0),
  
  -- Source information
  source TEXT DEFAULT 'uploaded' CHECK (source IN ('uploaded', 'api', 'purchased', 'shared')),
  source_url TEXT,
  source_metadata JSONB DEFAULT '{}'::jsonb,
  
  -- Full-text search
  text_content TEXT,
  text_indexed TSVECTOR,
  
  -- Visual
  thumbnail_url TEXT,
  cover_color TEXT DEFAULT '#1976D2',
  
  -- Additional metadata
  metadata_json JSONB DEFAULT '{}'::jsonb,
  tags TEXT[] DEFAULT '{}',
  
  -- Status
  processing_status TEXT DEFAULT 'completed' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
  processing_error TEXT,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  last_opened_at TIMESTAMPTZ
);

-- ============================================================================
-- ANNOTATIONS TABLE
-- ============================================================================

CREATE TABLE public.annotations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Location within book
  page INTEGER NOT NULL CHECK (page > 0),
  
  -- Coordinates (normalized 0-1)
  coords JSONB NOT NULL,
  /* Expected structure:
     {
       "x": 0.0-1.0,
       "y": 0.0-1.0,
       "width": 0.0-1.0,
       "height": 0.0-1.0
     }
  */
  
  -- Annotation type
  type TEXT NOT NULL CHECK (type IN ('highlight', 'underline', 'strikeout', 'note', 'drawing', 'text', 'comment')),
  
  -- Visual properties
  color TEXT DEFAULT '#FFEB3B',
  opacity REAL DEFAULT 0.3 CHECK (opacity >= 0 AND opacity <= 1),
  thickness INTEGER DEFAULT 2 CHECK (thickness > 0),
  
  -- Content
  selected_text TEXT,      -- For text-based annotations
  content TEXT,            -- For notes, comments
  drawing_data JSONB,      -- For freehand drawings (SVG paths)
  
  -- Sync metadata
  device_id TEXT,
  crdt_state JSONB,        -- Yjs CRDT state
  version BIGINT DEFAULT 1,
  synced_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Ensure coordinates are valid
  CONSTRAINT valid_coords CHECK (
    coords ? 'x' AND 
    coords ? 'y' AND
    (coords->>'x')::float >= 0 AND (coords->>'x')::float <= 1 AND
    (coords->>'y')::float >= 0 AND (coords->>'y')::float <= 1
  )
);

-- ============================================================================
-- BOOKMARKS TABLE (Reading Progress)
-- ============================================================================

CREATE TABLE public.bookmarks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Position
  page INTEGER NOT NULL CHECK (page > 0),
  scroll_offset REAL DEFAULT 0 CHECK (scroll_offset >= 0 AND scroll_offset <= 1),
  
  -- Additional location data
  location_json JSONB DEFAULT '{}'::jsonb,
  /* Example:
     {
       "zoom": 1.5,
       "viewMode": "single",
       "rotation": 0
     }
  */
  
  -- Progress percentage
  progress_percent REAL GENERATED ALWAYS AS (
    CASE 
      WHEN page IS NOT NULL AND EXISTS (
        SELECT 1 FROM books b WHERE b.id = book_id AND b.page_count > 0
      )
      THEN (page::float / (SELECT page_count FROM books WHERE id = book_id)) * 100
      ELSE 0
    END
  ) STORED,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- One bookmark per user per book
  UNIQUE(user_id, book_id)
);

-- ============================================================================
-- BOOK EDITS / VERSIONS TABLE
-- ============================================================================

CREATE TABLE public.book_edits (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Edit type
  edit_type TEXT NOT NULL CHECK (edit_type IN (
    'page_reorder', 
    'page_rotate', 
    'page_delete', 
    'page_add', 
    'text_edit', 
    'merge',
    'split'
  )),
  
  -- Edit details
  payload JSONB NOT NULL,
  /* Examples:
     page_reorder: {"from": 5, "to": 10}
     page_rotate: {"pages": [1, 2, 3], "degrees": 90}
     page_delete: {"pages": [5, 6]}
  */
  
  -- Versioning
  base_version BIGINT,
  result_storage_path TEXT,  -- Path to modified file
  
  -- Status
  status TEXT DEFAULT 'completed' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  error_message TEXT,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- DEVICES TABLE (Sync Tracking)
-- ============================================================================

CREATE TABLE public.devices (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Device information
  device_name TEXT,
  device_type TEXT CHECK (device_type IN ('ios', 'android', 'web', 'desktop')),
  device_info JSONB DEFAULT '{}'::jsonb,
  /* Example:
     {
       "os": "iOS",
       "osVersion": "17.0",
       "appVersion": "1.0.0",
       "model": "iPhone 15 Pro"
     }
  */
  
  -- Status
  last_seen_at TIMESTAMPTZ DEFAULT NOW(),
  is_active BOOLEAN DEFAULT true,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- COLLECTIONS TABLE (Folders/Tags)
-- ============================================================================

CREATE TABLE public.collections (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Collection metadata
  name TEXT NOT NULL,
  description TEXT,
  color TEXT DEFAULT '#2196F3',
  icon TEXT,  -- Icon identifier (e.g., 'folder', 'bookmark', 'star')
  
  -- Organization
  parent_id UUID REFERENCES public.collections(id) ON DELETE CASCADE,
  sort_order INTEGER DEFAULT 0,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Unique name per user
  UNIQUE(user_id, name)
);

-- ============================================================================
-- BOOK_COLLECTIONS JOIN TABLE
-- ============================================================================

CREATE TABLE public.book_collections (
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  collection_id UUID NOT NULL REFERENCES public.collections(id) ON DELETE CASCADE,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  
  PRIMARY KEY (book_id, collection_id)
);

-- ============================================================================
-- BOOK SHARES TABLE (Future: Collaboration)
-- ============================================================================

CREATE TABLE public.book_shares (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  book_id UUID NOT NULL REFERENCES public.books(id) ON DELETE CASCADE,
  shared_by UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  shared_with UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  
  -- Permissions
  permission TEXT NOT NULL DEFAULT 'read' CHECK (permission IN ('read', 'annotate', 'edit')),
  
  -- Status
  status TEXT DEFAULT 'active' CHECK (status IN ('active', 'revoked', 'expired')),
  expires_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Prevent duplicate shares
  UNIQUE(book_id, shared_with)
);

-- ============================================================================
-- SYNC EVENTS TABLE (Audit Log)
-- ============================================================================

CREATE TABLE public.sync_events (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  device_id UUID REFERENCES public.devices(id) ON DELETE SET NULL,
  
  -- Event details
  event_type TEXT NOT NULL CHECK (event_type IN (
    'annotation_create',
    'annotation_update',
    'annotation_delete',
    'bookmark_update',
    'book_upload',
    'book_delete',
    'sync_conflict'
  )),
  resource_type TEXT NOT NULL CHECK (resource_type IN ('book', 'annotation', 'bookmark', 'collection')),
  resource_id UUID,
  
  -- Additional data
  payload JSONB DEFAULT '{}'::jsonb,
  
  -- Timestamp
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Books indexes
CREATE INDEX idx_books_owner ON books(owner_id);
CREATE INDEX idx_books_created_at ON books(created_at DESC);
CREATE INDEX idx_books_last_opened ON books(last_opened_at DESC NULLS LAST);
CREATE INDEX idx_books_text_search ON books USING GIN(text_indexed);
CREATE INDEX idx_books_title_trgm ON books USING GIN(title gin_trgm_ops);
CREATE INDEX idx_books_tags ON books USING GIN(tags);

-- Annotations indexes
CREATE INDEX idx_annotations_book_page ON annotations(book_id, page);
CREATE INDEX idx_annotations_user ON annotations(user_id);
CREATE INDEX idx_annotations_created_at ON annotations(created_at DESC);
CREATE INDEX idx_annotations_type ON annotations(type);

-- Bookmarks indexes
CREATE INDEX idx_bookmarks_user_book ON bookmarks(user_id, book_id);
CREATE INDEX idx_bookmarks_updated ON bookmarks(updated_at DESC);

-- Book edits indexes
CREATE INDEX idx_book_edits_book ON book_edits(book_id, created_at DESC);

-- Devices indexes
CREATE INDEX idx_devices_user ON devices(user_id);
CREATE INDEX idx_devices_last_seen ON devices(last_seen_at DESC);

-- Collections indexes
CREATE INDEX idx_collections_user ON collections(user_id);
CREATE INDEX idx_collections_parent ON collections(parent_id);

-- Book collections indexes
CREATE INDEX idx_book_collections_book ON book_collections(book_id);
CREATE INDEX idx_book_collections_collection ON book_collections(collection_id);

-- Sync events indexes
CREATE INDEX idx_sync_events_user ON sync_events(user_id, created_at DESC);
CREATE INDEX idx_sync_events_resource ON sync_events(resource_type, resource_id);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function: Update books text search index
CREATE OR REPLACE FUNCTION books_text_index_trigger()
RETURNS TRIGGER AS $$
BEGIN
  NEW.text_indexed := to_tsvector('english', 
    COALESCE(NEW.title, '') || ' ' ||
    COALESCE(array_to_string(NEW.authors, ' '), '') || ' ' ||
    COALESCE(NEW.description, '') || ' ' ||
    COALESCE(NEW.text_content, '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function: Update user storage usage
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
  ELSIF TG_OP = 'UPDATE' AND NEW.file_size_bytes != OLD.file_size_bytes THEN
    UPDATE users
    SET storage_used_bytes = storage_used_bytes - OLD.file_size_bytes + NEW.file_size_bytes
    WHERE id = NEW.owner_id;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Function: Full-text search books
CREATE OR REPLACE FUNCTION search_books(search_query TEXT, user_id_filter UUID, result_limit INTEGER DEFAULT 10)
RETURNS TABLE(
  book_id UUID,
  title TEXT,
  authors TEXT[],
  thumbnail_url TEXT,
  match_snippet TEXT,
  rank REAL
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    b.id AS book_id,
    b.title,
    b.authors,
    b.thumbnail_url,
    ts_headline('english', b.text_content, query, 'MaxWords=20, MinWords=10') AS match_snippet,
    ts_rank(b.text_indexed, query) AS rank
  FROM books b,
       to_tsquery('english', search_query) query
  WHERE b.text_indexed @@ query
    AND b.owner_id = user_id_filter
  ORDER BY rank DESC
  LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Batch create annotations
CREATE OR REPLACE FUNCTION batch_create_annotations(annotations_data JSONB)
RETURNS SETOF annotations AS $$
DECLARE
  annotation_record JSONB;
  new_annotation annotations;
BEGIN
  FOR annotation_record IN SELECT * FROM jsonb_array_elements(annotations_data)
  LOOP
    INSERT INTO annotations (
      book_id,
      user_id,
      page,
      coords,
      type,
      color,
      selected_text,
      content,
      device_id
    ) VALUES (
      (annotation_record->>'book_id')::UUID,
      auth.uid(),
      (annotation_record->>'page')::INTEGER,
      annotation_record->'coords',
      annotation_record->>'type',
      COALESCE(annotation_record->>'color', '#FFEB3B'),
      annotation_record->>'selected_text',
      annotation_record->>'content',
      annotation_record->>'device_id'
    )
    RETURNING * INTO new_annotation;
    
    RETURN NEXT new_annotation;
  END LOOP;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at triggers
CREATE TRIGGER users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER books_updated_at
  BEFORE UPDATE ON books
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER annotations_updated_at
  BEFORE UPDATE ON annotations
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER bookmarks_updated_at
  BEFORE UPDATE ON bookmarks
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER collections_updated_at
  BEFORE UPDATE ON collections
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER devices_updated_at
  BEFORE UPDATE ON devices
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

-- Text search index trigger
CREATE TRIGGER books_text_index_update
  BEFORE INSERT OR UPDATE ON books
  FOR EACH ROW
  EXECUTE FUNCTION books_text_index_trigger();

-- Storage usage triggers
CREATE TRIGGER books_storage_usage
  AFTER INSERT OR DELETE OR UPDATE ON books
  FOR EACH ROW
  EXECUTE FUNCTION update_storage_usage();

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.books ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bookmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.book_edits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.book_collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.book_shares ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sync_events ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view own profile"
  ON public.users FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.users FOR UPDATE
  USING (auth.uid() = id);

-- Books policies
CREATE POLICY "Users can view owned or shared books"
  ON public.books FOR SELECT
  USING (
    auth.uid() = owner_id OR
    EXISTS (
      SELECT 1 FROM book_shares
      WHERE book_id = books.id
      AND shared_with = auth.uid()
      AND status = 'active'
      AND (expires_at IS NULL OR expires_at > NOW())
    )
  );

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
CREATE POLICY "Users can view annotations on accessible books"
  ON public.annotations FOR SELECT
  USING (
    user_id = auth.uid() OR
    EXISTS (
      SELECT 1 FROM books b
      LEFT JOIN book_shares bs ON b.id = bs.book_id
      WHERE b.id = annotations.book_id
      AND (
        b.owner_id = auth.uid() OR
        (bs.shared_with = auth.uid() AND bs.status = 'active')
      )
    )
  );

CREATE POLICY "Users can insert own annotations"
  ON public.annotations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own annotations"
  ON public.annotations FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own annotations"
  ON public.annotations FOR DELETE
  USING (auth.uid() = user_id);

-- Bookmarks policies
CREATE POLICY "Users can view own bookmarks"
  ON public.bookmarks FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own bookmarks"
  ON public.bookmarks FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own bookmarks"
  ON public.bookmarks FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own bookmarks"
  ON public.bookmarks FOR DELETE
  USING (auth.uid() = user_id);

-- Book edits policies
CREATE POLICY "Users can view edits on own books"
  ON public.book_edits FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own book edits"
  ON public.book_edits FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Devices policies
CREATE POLICY "Users can view own devices"
  ON public.devices FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own devices"
  ON public.devices FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own devices"
  ON public.devices FOR UPDATE
  USING (auth.uid() = user_id);

-- Collections policies
CREATE POLICY "Users can view own collections"
  ON public.collections FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own collections"
  ON public.collections FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Book collections policies (inherit from books and collections)
CREATE POLICY "Users can view book collections for accessible books"
  ON public.book_collections FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM books b
      WHERE b.id = book_id AND b.owner_id = auth.uid()
    )
  );

CREATE POLICY "Users can manage book collections"
  ON public.book_collections FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM books b
      WHERE b.id = book_id AND b.owner_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM books b
      WHERE b.id = book_id AND b.owner_id = auth.uid()
    )
  );

-- Sync events policies
CREATE POLICY "Users can view own sync events"
  ON public.sync_events FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own sync events"
  ON public.sync_events FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- ============================================================================
-- SAMPLE DATA (Optional - for development)
-- ============================================================================

-- Uncomment to insert sample data
/*
-- Insert sample user (after auth.users entry exists)
INSERT INTO public.users (id, email, full_name)
VALUES (
  'sample-user-uuid',
  'demo@bookflow.app',
  'Demo User'
);
*/

-- ============================================================================
-- GRANTS
-- ============================================================================

-- Grant access to authenticated users
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- Grant access to service role (for edge functions)
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;
