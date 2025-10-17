# Role-Specific Implementation Prompts
## BookFlow - Specialized Handoff Documents

---

## 📋 For Product Manager (PM)

**Prompt:**

You are a product manager for BookFlow, a cross-platform mobile (Android & iOS) and web application that enables users to upload/import books and PDFs, read them, annotate and edit where feasible, search across library (local + via external book API), and sync everything across devices. The backend uses Supabase (Auth, Postgres, Realtime, Storage).

**Your Task:**

Produce a prioritized product backlog with:

1. **User Stories** - Detailed user stories for MVP features including:
   - Authentication (email/password + OAuth)
   - File upload and storage
   - PDF viewer with annotations
   - Sync across devices
   - Search (local full-text + external API)
   - Offline support

2. **Acceptance Criteria** - Clear, testable criteria for each story

3. **8-Week Sprint Plan** - Break down MVP into 4 sprints with:
   - Sprint goals
   - Story points
   - Dependencies
   - Risk mitigation

4. **Success Metrics**:
   - **Acquisition**: Download/signup targets
   - **Activation**: Books uploaded per new user
   - **Engagement**: DAU/MAU ratio, annotations created
   - **Retention**: D1, D7, D30 retention rates
   - **Revenue**: Free-to-paid conversion rate

5. **Risk Register**:
   - Technical risks (PDF rendering performance, sync conflicts)
   - Business risks (low user adoption, competitive pressure)
   - Operational risks (Supabase scaling limits)
   - Mitigation strategies for each

6. **Dependencies**:
   - External APIs (Google Books, Supabase)
   - Platform approvals (App Store, Play Store)
   - Third-party SDKs (potential paid PDF libraries)

**Output Format:**
- Prioritized backlog in CSV or Jira-compatible format
- Sprint plan in Gantt chart or table
- Metrics dashboard mockup (KPIs to track)

---

## 💻 For Mobile Engineer (React Native / Flutter)

**Prompt:**

You are a senior mobile engineer tasked with building BookFlow, a cross-platform app (iOS, Android, Web) for reading and annotating PDFs with cloud sync.

**Stack Decision:**
Compare React Native + Expo + React Native for Web vs. Flutter + Flutter Web:
- Justify recommendation based on:
  - PDF rendering capabilities (pdf.js, react-native-pdf vs native PDF renderers)
  - Code sharing between mobile and web (>80% target)
  - Supabase SDK support
  - Development velocity
  - Team skillset (assume TypeScript familiarity)

**Your Task:**

1. **Technical Implementation Plan**:
   - Recommended tech stack with justification
   - Project structure (folder organization, module boundaries)
   - State management approach (Zustand, Redux, Bloc, etc.)
   - Navigation architecture (Expo Router, React Navigation, GoRouter)

2. **PDF Rendering Strategy**:
   - Web: pdf.js integration (canvas rendering, virtualization for large files)
   - Mobile: react-native-pdf or native PDFKit/PdfRenderer
   - Performance optimization (lazy loading, caching, memory management)

3. **Annotation Layer**:
   - SVG/Canvas overlay for highlights, drawings, notes
   - Coordinate normalization (0-1 scale for cross-device compatibility)
   - Touch gesture handling (pan, zoom, long-press select text)

4. **Offline Support**:
   - Local persistence (SQLite, AsyncStorage, IndexedDB)
   - Sync queue implementation (pending operations with retry logic)
   - Conflict resolution (CRDT for annotations, last-write-wins for bookmarks)

5. **Code Examples**:
   ```typescript
   // Example: PDF viewer component structure
   <PDFViewer
     url={signedUrl}
     page={currentPage}
     onPageChange={handlePageChange}
     annotations={pageAnnotations}
     onAnnotationCreate={handleAnnotationCreate}
   >
     <AnnotationLayer
       annotations={pageAnnotations}
       mode={annotationMode}
       onSelect={handleAnnotationSelect}
     />
   </PDFViewer>
   ```

6. **Network Layer**:
   - Supabase client setup (auth, realtime subscriptions)
   - React Query integration (caching, optimistic updates)
   - File upload with progress tracking (chunked uploads for large files)

7. **Testing Strategy**:
   - Unit tests (Jest)
   - Component tests (React Native Testing Library)
   - E2E tests (Detox for mobile, Playwright for web)

**Deliverables:**
- Architecture diagram (components, data flow)
- Sample code for core features
- Performance benchmarks (target: <2s PDF load time for 100-page doc)

---

## ⚙️ For Backend Engineer

**Prompt:**

You are a backend engineer building the API and database for BookFlow using Supabase (Postgres, Realtime, Storage, Edge Functions).

**Your Task:**

1. **Database Schema**:
   - Provide complete SQL table definitions for:
     - `users`, `books`, `annotations`, `bookmarks`, `book_edits`, `devices`, `collections`
   - Indexes for performance (text search, annotation lookups by book/page)
   - Triggers (auto-update timestamps, storage quota calculation)

2. **Row Level Security (RLS) Policies**:
   ```sql
   -- Example: Users can only view/edit their own annotations
   CREATE POLICY "Users view own annotations"
     ON annotations FOR SELECT
     USING (auth.uid() = user_id);
   
   -- Shared books: allow viewing if shared
   CREATE POLICY "Users view shared books"
     ON books FOR SELECT
     USING (
       owner_id = auth.uid() OR
       EXISTS (SELECT 1 FROM book_shares WHERE book_id = books.id AND shared_with = auth.uid())
     );
   ```

3. **Realtime Sync Architecture**:
   - WebSocket channel design (`book:${bookId}:annotations`)
   - Broadcast annotation changes to all subscribed clients
   - Delta-based updates (only send changed fields)
   - Client subscription pattern:
   ```typescript
   supabase
     .channel(`book:${bookId}`)
     .on('postgres_changes', { event: '*', schema: 'public', table: 'annotations' }, handleChange)
     .subscribe();
   ```

4. **Edge Functions** (Deno Deploy):
   ```typescript
   // supabase/functions/process-upload/index.ts
   Deno.serve(async (req) => {
     const { bookId } = await req.json();
     
     // 1. Download PDF from Storage
     const fileData = await downloadFromStorage(bookId);
     
     // 2. Extract metadata (pdf-lib or pdf.js in Deno)
     const metadata = await extractPdfMetadata(fileData);
     
     // 3. Generate thumbnail (first page as JPEG)
     const thumbnail = await generateThumbnail(fileData);
     
     // 4. Extract text for search indexing
     const text = await extractText(fileData);
     
     // 5. Update books table
     await supabase.from('books').update({
       title: metadata.title,
       page_count: metadata.pageCount,
       thumbnail_url: thumbnail.url,
       text_content: text,
     }).eq('id', bookId);
     
     return new Response(JSON.stringify({ success: true }));
   });
   ```

5. **OCR Job Queue**:
   - Async job processing (use Supabase Edge Functions + external queue like BullMQ if needed)
   - Tesseract.js for OCR or Google Cloud Vision API
   - Update `books.text_content` with OCR results

6. **Full-Text Search**:
   ```sql
   -- Using Postgres tsvector
   SELECT id, title, ts_rank(text_indexed, query) AS rank,
          ts_headline('english', text_content, query) AS snippet
   FROM books, to_tsquery('english', 'machine & learning') query
   WHERE text_indexed @@ query
   ORDER BY rank DESC
   LIMIT 10;
   ```

7. **API Endpoints** (REST via PostgREST):
   - GET `/books` - List user's books with filters
   - POST `/books` - Create book entry after upload
   - GET `/annotations?book_id=eq.{id}` - Get annotations for book
   - POST `/rpc/batch_create_annotations` - Batch insert for offline sync

8. **Storage Buckets**:
   - `books` bucket (private, 200MB file limit, PDF/ePub MIME types)
   - `thumbnails` bucket (public, 5MB limit, image MIME types)
   - Signed URL generation (1-hour expiry)

**Deliverables:**
- Complete SQL migration file
- RLS policy examples
- Edge Function implementations
- API request/response JSON samples

---

## 🎨 For UX/UI Designer

**Prompt:**

You are a UX/UI designer creating the interface for BookFlow, a mobile-first (iOS/Android) and web-responsive PDF reading and annotation app.

**Your Task:**

1. **Wireframes** for key screens:
   - **Login/Signup** - Email/password + OAuth (Google, Apple)
   - **Library** - Grid/list view, search bar, sort/filter options
   - **Reader** - PDF canvas, annotation toolbar, page slider, thumbnail drawer
   - **Annotations List** - Grouped by page, filterable by type
   - **Upload** - File picker, drag-drop (web), URL import, external search
   - **Profile/Settings** - Storage usage, sync status, theme toggle

2. **Annotation Toolbar** (floating UI in reader):
   ```
   ┌─────────────────────────────────┐
   │ 🖊  💡  ✏️  🗑  T  ↩ ↪         │
   └─────────────────────────────────┘
   ```
   Icons: Highlight, Note, Draw, Eraser, Text, Undo, Redo
   - Color picker for highlights (Yellow, Green, Blue, Pink, Orange)
   - Thickness selector for drawing (Fine, Medium, Thick)

3. **Sync State Indicators**:
   - Green dot: Synced
   - Orange dot: Pending sync
   - Red dot: Sync error
   - Sync panel (modal) showing pending operations and errors

4. **Empty States**:
   - Empty library: "Upload Your First Book" CTA
   - No annotations: "Start highlighting to remember key points"
   - No search results: "Try different keywords"

5. **Loading States**:
   - Skeleton screens for book cards
   - Upload progress bar with percentage and estimated time
   - Page loading spinner in PDF viewer

6. **Error States**:
   - Toast notifications (auto-dismiss after 3s)
   - Inline error messages with retry actions
   - Full-screen error for critical failures

7. **Theme Specifications**:
   - **Light**: White background, dark text, #1976D2 primary
   - **Dark**: #121212 background, white text, #42A5F5 primary
   - **Sepia**: #F4ECD8 background for reading comfort

8. **Style Guide**:
   - **Typography**: System fonts (SF Pro iOS, Roboto Android)
     - H1: 32px Bold
     - H2: 24px Bold
     - Body: 16px Regular
     - Caption: 12px Regular
   - **Spacing**: 4, 8, 16, 24, 32, 48px scale
   - **Border Radius**: 4px (inputs), 8px (cards), 16px (modals)
   - **Shadows**: Elevation small/medium/large

9. **Accessibility**:
   - Minimum touch target: 44x44pt
   - Color contrast: WCAG AA (4.5:1 for text)
   - Screen reader labels for all interactive elements
   - Focus indicators for keyboard navigation (web)

10. **Responsive Breakpoints**:
    - Mobile: <768px (single column, bottom nav)
    - Tablet: 768-1024px (two columns, side drawer)
    - Desktop: >1024px (multi-column grid, sidebar always visible)

**Deliverables:**
- High-fidelity mockups in Figma
- Component library (buttons, inputs, cards)
- Prototype with interaction flows
- Style guide document

---

## 🧪 For QA / Test Engineer

**Prompt:**

You are a QA engineer responsible for testing BookFlow MVP (mobile + web PDF reader with sync).

**Your Task:**

1. **Test Plan** covering:
   - **Functional Testing**: All features work as specified
   - **Integration Testing**: Supabase API, file upload, sync
   - **Performance Testing**: PDF load time, sync latency
   - **Security Testing**: Authentication, RLS policies, file access
   - **Compatibility Testing**: iOS 15+, Android 10+, modern browsers

2. **Test Cases** (examples):

   **TC-001: User Sign Up**
   - Steps:
     1. Open app
     2. Tap "Sign Up"
     3. Enter email, password (8+ chars), full name
     4. Tap "Create Account"
   - Expected: Success, navigate to library, user profile created in DB
   - Negative: Invalid email → error message, weak password → warning

   **TC-012: Upload PDF**
   - Steps:
     1. Tap "Upload" button
     2. Select PDF from device (50MB file)
     3. Observe progress
   - Expected: Upload completes, thumbnail generated, book appears in library
   - Performance: Upload completes in <30s on 10Mbps connection

   **TC-025: Create Highlight Annotation**
   - Steps:
     1. Open book
     2. Long-press text to select
     3. Tap "Highlight" in color picker
     4. Choose yellow color
   - Expected: Text highlighted, annotation saved, appears on other devices within 2s

   **TC-030: Sync Across Devices**
   - Steps:
     1. Device A: Create annotation on page 10
     2. Device B: Open same book
   - Expected: Annotation visible on Device B within 2s
   - Conflict: Both devices create annotation offline → merge without data loss

   **TC-040: Full-Text Search**
   - Steps:
     1. Tap search icon
     2. Enter query "machine learning"
     3. Tap result
   - Expected: Results displayed in <1s, tapping result opens book to matching page

   **TC-050: Offline Mode**
   - Steps:
     1. Enable airplane mode
     2. Open previously cached book
     3. Create annotation
     4. Re-enable network
   - Expected: Book opens, annotation queued, syncs when online

3. **Security Tests**:
   - **Auth-001**: Invalid JWT → 401 Unauthorized
   - **RLS-001**: User A cannot access User B's books (403 Forbidden)
   - **Storage-001**: Signed URL expires after 1 hour
   - **Upload-001**: Reject files > 200MB or non-PDF MIME types

4. **Performance Tests**:
   - **PDF Load**: 100-page document loads in <2s
   - **Sync Latency**: Annotation sync completes in <2s (good network)
   - **Search Speed**: Library search returns in <500ms for 1000 books

5. **Regression Test Checklist**:
   - [ ] All auth flows (signup, login, logout, password reset)
   - [ ] File upload and download
   - [ ] PDF rendering (zoom, scroll, page navigation)
   - [ ] All annotation types (highlight, note, draw)
   - [ ] Sync (create, update, delete annotations)
   - [ ] Search (local + external)
   - [ ] Offline mode and reconnect
   - [ ] Theme toggle (light/dark)

6. **Bug Report Template**:
   ```
   **Title**: [Component] Brief description
   **Severity**: Critical | High | Medium | Low
   **Steps to Reproduce**:
   1. ...
   2. ...
   **Expected**: ...
   **Actual**: ...
   **Environment**: iOS 17.0, iPhone 15 Pro, App v1.0.0
   **Screenshots**: [Attach]
   **Logs**: [Attach console output]
   ```

**Deliverables:**
- Comprehensive test plan document
- Test case spreadsheet (100+ cases)
- Automated test scripts (Jest, Detox, Playwright)
- Performance benchmarks report

---

## 🚀 Deployment & DevOps

**Prompt:**

Set up CI/CD pipeline for BookFlow (React Native + Expo) with:

1. **GitHub Actions Workflow**:
   - Lint and type-check on every PR
   - Run unit tests
   - Build iOS/Android apps via EAS Build
   - Deploy web to Vercel
   - Deploy Edge Functions to Supabase

2. **Environment Management**:
   - Dev, Staging, Production Supabase projects
   - Environment-specific .env files
   - Secrets management (GitHub Secrets for API keys)

3. **Mobile Release**:
   - EAS Build profiles (development, preview, production)
   - App Store Connect / Google Play Console integration
   - Automated version bumping
   - Beta testing via TestFlight / Internal Testing

4. **Monitoring**:
   - Sentry for crash reporting
   - PostHog for analytics
   - Supabase logs for backend errors
   - Uptime monitoring for API

**Deliverable:**
- `.github/workflows/ci.yml`
- `eas.json` build configuration
- Deployment runbook

---

## 📊 Sample Mock Data

**Mock "Book" Object (JSON):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "owner_id": "user-uuid-123",
  "title": "Principles of Modern Physics",
  "authors": ["A. Author", "B. Writer"],
  "description": "Comprehensive introduction to modern physics concepts including quantum mechanics, relativity, and particle physics.",
  "language": "en",
  "isbn": "978-1-2345-6789-0",
  "publisher": "OpenBooks Publishing",
  "published_date": "2023-05-15",
  "file_type": "pdf",
  "storage_path": "user-uuid-123/550e8400-e29b-41d4-a716-446655440000.pdf",
  "file_size_bytes": 42134567,
  "page_count": 432,
  "source": "uploaded",
  "source_url": null,
  "source_metadata": {},
  "text_content": "Chapter 1: Introduction to Quantum Mechanics...",
  "thumbnail_url": "https://storage.url/thumbnails/550e8400-thumb.jpg",
  "cover_color": "#1976D2",
  "metadata_json": {
    "subject": "Physics",
    "keywords": ["quantum", "relativity", "modern physics"]
  },
  "tags": ["physics", "textbook"],
  "processing_status": "completed",
  "created_at": "2025-10-10T08:30:00Z",
  "updated_at": "2025-10-17T10:30:00Z",
  "last_opened_at": "2025-10-17T09:15:00Z"
}
```

**Mock "Annotation" Object:**
```json
{
  "id": "annot-uuid-456",
  "book_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user-uuid-123",
  "page": 12,
  "coords": {
    "x": 0.15,
    "y": 0.3,
    "width": 0.6,
    "height": 0.03
  },
  "type": "highlight",
  "color": "#FFEB3B",
  "opacity": 0.3,
  "thickness": 2,
  "selected_text": "The wave-particle duality is a fundamental concept in quantum mechanics.",
  "content": "Important concept - review for exam",
  "drawing_data": null,
  "device_id": "device-abc-123",
  "crdt_state": null,
  "version": 1,
  "synced_at": "2025-10-17T10:20:00Z",
  "created_at": "2025-10-17T10:20:00Z",
  "updated_at": "2025-10-17T10:20:00Z"
}
```

---

## 🎯 Single Comprehensive Prompt (All-in-One)

**For Windsurf AI or any development assistant:**

Build a cross-platform mobile (Android & iOS) and web app named **"BookFlow"** that enables users to upload/import books and PDFs, read them, annotate and edit where feasible, search across library (local + via external book API), and sync everything across devices. Use **Supabase** for auth, Postgres, Realtime and Storage as the default backend.

### Provide:

1. **Prioritized MVP feature list** with acceptance criteria and 8-week sprint plan

2. **Full technical architecture**:
   - Frontend stack recommendation (React Native + Expo vs Flutter) with justification
   - Backend APIs and Postgres schema (SQL with RLS policies)
   - Realtime sync architecture and offline strategy
   - Exact API endpoints with sample request/response payloads

3. **Detailed Supabase table definitions**:
   - `users`, `books`, `annotations`, `book_versions`, `devices`
   - Example RLS policies and index suggestions

4. **Sync & conflict resolution**:
   - CRDT approach (Automerge/Yjs) for annotations or practical fallback
   - Client+server flow diagrams
   - Sample JSON deltas

5. **PDF editing strategy** and recommended libraries/SDKs:
   - pdf.js, pdf-lib, PSPDFKit, PDFTron with trade-offs
   - When to use paid SDKs
   - Support for annotations, highlights, freehand drawing, sticky notes, text edits, page reordering, merge/split

6. **OCR plan**:
   - Tesseract vs cloud OCR
   - Where to run (client WASM / serverless)
   - How to index text for search (tsvector vs Algolia)

7. **Security & privacy checklist**:
   - Signed URLs, encryption at rest, RLS, GDPR export/delete, virus scanning
   - Cost/licensing considerations for commercial PDF SDKs

8. **Detailed UI/UX components**:
   - Wireframe descriptions
   - Theme (light/dark)
   - Accessibility considerations
   - Copy for key states (uploading/sync error/conflict)

9. **Runnable sample DB schema** (SQL), sample API endpoints (REST + WebSocket channels), sample annotation JSON payloads

10. **Roadmap** with milestones, risks, and metrics (DAU, retention, conversion)

### Also produce specialized prompts for:
- PM (prioritized backlog, sprint plan, metrics)
- Mobile Engineer (tech stack, architecture, code samples)
- Backend Engineer (schema, RLS, Edge Functions, sync)
- UX Designer (wireframes, components, style guide)
- QA Engineer (test plan, test cases, security/performance tests)

Keep everything highly actionable with concrete examples and code snippets.

---

**End of Role Prompts Document**
