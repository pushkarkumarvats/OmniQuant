# Product Requirements Document (PRD)
## BookFlow - Mobile & Web PDF/Book Reader, Editor & Sync

**Version:** 1.0  
**Date:** October 2025  
**Status:** Draft  

---

## 1. Executive Summary

### 1.1 Product Vision
BookFlow empowers readers to manage, annotate, and edit their digital libraries across all devices with professional-grade PDF editing tools and seamless cloud synchronization.

### 1.2 Business Objectives
- Capture market share in the mobile PDF annotation space (competing with Xodo, Adobe Acrobat Mobile)
- Generate recurring revenue through freemium subscription model
- Build engaged user base of 100K MAU within 12 months
- Achieve 15% free-to-paid conversion rate

### 1.3 Success Metrics
| Metric | Target (3 months) | Target (12 months) |
|--------|-------------------|-------------------|
| MAU | 10,000 | 100,000 |
| DAU/MAU Ratio | 30% | 40% |
| D30 Retention | 25% | 35% |
| Avg Annotations/User/Week | 15 | 30 |
| Free→Paid Conversion | 8% | 15% |
| Avg Session Duration | 8 min | 12 min |

---

## 2. User Personas

### 2.1 Sarah - Graduate Student
- **Age:** 24
- **Needs:** Annotate research papers, organize literature reviews, search across papers
- **Pain Points:** Switching between laptop and phone loses annotations; can't easily search handwritten notes
- **Goals:** Quick highlighting during commute, detailed notes on desktop, everything synced

### 2.2 Michael - Corporate Lawyer
- **Age:** 38
- **Needs:** Review contracts, add comments, share annotated docs with colleagues
- **Pain Points:** Adobe Acrobat too expensive for whole team; cloud storage security concerns
- **Goals:** Professional PDF editing, secure storage, offline access during flights

### 2.3 Emma - Self-Published Author
- **Age:** 31
- **Needs:** Review manuscript PDFs, make edits, track changes
- **Pain Points:** Desktop-only tools limit flexibility; version control is manual
- **Goals:** Edit on the go, version history, export to different formats

### 2.4 James - Casual Reader
- **Age:** 42
- **Needs:** Read ebooks and PDFs, bookmark progress, night reading mode
- **Pain Points:** Current apps don't sync reading position; too many features he doesn't need
- **Goals:** Simple reading experience, works offline, remembers where he left off

---

## 3. MVP Feature Specifications

### 3.1 Authentication & Onboarding

#### 3.1.1 User Stories
- **US-001:** As a new user, I want to sign up with email/password so I can create an account
- **US-002:** As a user, I want to sign in with Google/Apple so I can skip manual registration
- **US-003:** As a user, I want to reset my password so I can regain access if I forget it
- **US-004:** As a new user, I want to see an onboarding tutorial so I understand key features

#### 3.1.2 Acceptance Criteria
- [ ] Email/password signup with validation (min 8 chars, password strength indicator)
- [ ] Social OAuth (Google, Apple) integration
- [ ] Email verification flow with resend option
- [ ] Password reset via email with secure token
- [ ] 3-screen onboarding tutorial (swipeable, skippable)
- [ ] Auto-login on app relaunch (secure token storage)

#### 3.1.3 Technical Notes
- Use Supabase Auth for all auth flows
- Store JWT in secure storage (Keychain iOS, Keystore Android, localStorage web)
- Implement refresh token rotation

---

### 3.2 Library Management

#### 3.2.1 User Stories
- **US-010:** As a user, I want to see all my books in a grid/list view so I can browse my library
- **US-011:** As a user, I want to sort books by title/author/date so I can find books quickly
- **US-012:** As a user, I want to search my library so I can find specific books
- **US-013:** As a user, I want to see reading progress on each book so I know where I left off
- **US-014:** As a user, I want to organize books with tags/collections so I can categorize them

#### 3.2.2 Acceptance Criteria
- [ ] Grid view with thumbnail, title, author, progress bar
- [ ] List view with compact layout
- [ ] Toggle between grid/list with preference saved
- [ ] Sort options: Recent, Title (A-Z), Author, Date Added
- [ ] Search bar filters library in real-time
- [ ] Long-press context menu: Delete, Share, Move to Collection
- [ ] Empty state with "Upload Your First Book" CTA
- [ ] Pull-to-refresh to sync latest changes

#### 3.2.3 UI Components
```
LibraryScreen
├── SearchBar (filterable)
├── SortDropdown (Recent | Title | Author | Date)
├── ViewToggle (Grid | List)
└── BookList/BookGrid
    └── BookCard
        ├── Thumbnail (generated on upload)
        ├── Title (2 lines max, ellipsis)
        ├── Author (1 line)
        ├── ProgressBar (0-100%)
        └── ContextMenu (⋮)
```

---

### 3.3 File Upload & Storage

#### 3.3.1 User Stories
- **US-020:** As a user, I want to upload PDFs from my device so I can read them in the app
- **US-021:** As a user, I want to import PDFs from URL so I can add remote files
- **US-022:** As a user, I want to see upload progress so I know when it's complete
- **US-023:** As a user, I want to import books from external API so I can discover new content

#### 3.3.2 Acceptance Criteria
- [ ] File picker for local PDF selection (supports multi-select)
- [ ] Drag-and-drop upload on web
- [ ] URL import with validation
- [ ] Progress indicator (0-100%) with cancel option
- [ ] Automatic metadata extraction (title, author, page count)
- [ ] Thumbnail generation (first page preview)
- [ ] File size limit: 200MB (configurable)
- [ ] Supported formats: PDF (MVP), ePub/MOBI (post-MVP)
- [ ] External API search (Google Books / Open Library) with "Add to Library" button

#### 3.3.3 Technical Flow
1. User selects file → Client validates (type, size)
2. Client requests signed upload URL from API
3. Client uploads directly to Supabase Storage (chunked, resumable)
4. Client notifies API of completion
5. Server processes file (extract metadata, generate thumbnail, extract text for search)
6. Server creates `books` table entry
7. Client receives book ID, navigates to reader

---

### 3.4 PDF Viewer & Reader

#### 3.4.1 User Stories
- **US-030:** As a user, I want to view PDF pages so I can read the content
- **US-031:** As a user, I want to navigate pages with swipe/scroll so it feels natural
- **US-032:** As a user, I want to zoom in/out so I can read small text
- **US-033:** As a user, I want to toggle light/dark mode so I can read comfortably
- **US-034:** As a user, I want to see page thumbnails so I can jump to specific pages
- **US-035:** As a user, I want to reflow text so I can read on small screens

#### 3.4.2 Acceptance Criteria
- [ ] Smooth page rendering (pdf.js web, react-native-pdf mobile)
- [ ] Horizontal swipe to change pages (mobile) or scroll (web)
- [ ] Pinch-to-zoom with smooth animation
- [ ] Page slider at bottom (1-indexed display)
- [ ] Thumbnail strip (collapsible drawer)
- [ ] Theme toggle: Light, Dark, Sepia
- [ ] Font size adjustment (reflow mode only)
- [ ] Single/double page layout (tablet/web)
- [ ] Full-screen mode (hides toolbars)
- [ ] Reading position auto-saved every 5 seconds

#### 3.4.3 UI Layout
```
ReaderScreen
├── TopBar
│   ├── BackButton
│   ├── BookTitle
│   └── BookmarkButton
├── AnnotationToolbar (floating)
│   ├── HighlightButton
│   ├── UnderlineButton
│   ├── DrawButton
│   ├── NoteButton
│   └── EraserButton
├── PDFCanvas (main view)
│   └── AnnotationOverlay (SVG/Canvas layer)
├── PageSlider (bottom)
│   ├── CurrentPage input
│   ├── Slider (1 to N)
│   └── TotalPages label
└── ThumbnailDrawer (side)
    └── ThumbnailList
```

---

### 3.5 Annotations

#### 3.5.1 User Stories
- **US-040:** As a user, I want to highlight text so I can mark important passages
- **US-041:** As a user, I want to choose highlight color so I can categorize notes
- **US-042:** As a user, I want to add text notes so I can write my thoughts
- **US-043:** As a user, I want to draw freehand so I can sketch diagrams
- **US-044:** As a user, I want to erase annotations so I can correct mistakes
- **US-045:** As a user, I want to see all my annotations in a list so I can review them

#### 3.5.2 Annotation Types

**Highlight**
- User selects text (long-press on mobile, click-drag on web)
- Color picker appears (Yellow, Green, Blue, Pink, Orange)
- Highlight saved with coordinates, color, extracted text
- Tapping highlight shows popup: Edit Color | Add Note | Delete

**Underline/Strikeout**
- Same selection flow as highlight
- Style option: Underline | Strikeout | Squiggly

**Freehand Drawing**
- Activate pen tool, choose color & thickness
- Touch/mouse draws path (captured as SVG path or coordinates array)
- Smooth stroke rendering with pressure sensitivity (if supported)

**Sticky Note**
- Tap location → note icon appears
- Modal opens for text input
- Note displays as icon on page, expands on tap

**Text Comment**
- Long-press text → "Add Comment" option
- Inline comment anchored to text selection

#### 3.5.3 Acceptance Criteria
- [ ] All annotation types functional on mobile & web
- [ ] Color picker with 5+ preset colors
- [ ] Pen thickness selector (Fine, Medium, Thick)
- [ ] Undo/Redo for annotation actions (last 20 operations)
- [ ] Annotations persist locally immediately
- [ ] Annotations sync to server within 2 seconds (online)
- [ ] Annotations appear on other devices in real-time
- [ ] Annotation list view: sortable by page, date, type
- [ ] Tap annotation in list → jumps to page & highlights annotation

#### 3.5.4 Data Model (JSON)
```json
{
  "id": "uuid-1234",
  "book_id": "book-uuid",
  "user_id": "user-uuid",
  "page": 12,
  "type": "highlight",
  "coords": {
    "x": 0.12,
    "y": 0.45,
    "width": 0.6,
    "height": 0.03
  },
  "color": "#FFEB3B",
  "text": "extracted text content",
  "note": "optional user comment",
  "created_at": "2025-10-17T10:30:00Z",
  "updated_at": "2025-10-17T10:30:00Z",
  "device_id": "device-abc",
  "version": 1
}
```

---

### 3.6 Real-Time Sync

#### 3.6.1 User Stories
- **US-050:** As a user, I want my annotations to sync across devices so I can switch devices seamlessly
- **US-051:** As a user, I want to see sync status so I know when changes are saved
- **US-052:** As a user, I want offline changes to sync automatically when I reconnect

#### 3.6.2 Sync Architecture

**Client-Side:**
1. User creates annotation → saved to local SQLite/IndexedDB
2. Mark as "pending sync" (status icon: orange dot)
3. Background sync worker attempts upload
4. On success → mark as "synced" (green dot)
5. On failure → retry with exponential backoff (show red dot)

**Server-Side:**
1. API receives annotation delta
2. Validate user permission (RLS)
3. Apply CRDT merge (Yjs) if conflicts exist
4. Persist to Postgres
5. Broadcast to Supabase Realtime channel
6. Return ack with canonical version

**Real-Time Subscription:**
```javascript
// Subscribe to book's annotation channel
supabase
  .channel(`book:${bookId}:annotations`)
  .on('postgres_changes', {
    event: '*',
    schema: 'public',
    table: 'annotations',
    filter: `book_id=eq.${bookId}`
  }, (payload) => {
    // Update local state with remote change
    mergeRemoteAnnotation(payload.new);
  })
  .subscribe();
```

#### 3.6.3 Conflict Resolution
- **Annotations (CRDT):** Yjs automatically merges concurrent edits
- **Reading Position:** Last-write-wins (timestamp-based)
- **Page Edits (reorder/delete):** Version-based, manual merge UI if conflict detected

#### 3.6.4 Acceptance Criteria
- [ ] Annotations sync within 2 seconds on good connection
- [ ] Sync status indicator: Green (synced), Orange (pending), Red (error)
- [ ] Offline queue stores up to 1000 pending operations
- [ ] Automatic retry on reconnect (with visual feedback)
- [ ] Manual "Retry Sync" button in error state
- [ ] Conflict notification with option to view changes

---

### 3.7 Search

#### 3.7.1 User Stories
- **US-060:** As a user, I want to search within a book so I can find specific passages
- **US-061:** As a user, I want to search across my entire library so I can find any content
- **US-062:** As a user, I want to search external book APIs so I can discover new books

#### 3.7.2 Search Types

**In-Book Search**
- Search bar in reader view
- Highlights matches on current page
- "Next/Previous" buttons to navigate results
- Shows result count (e.g., "3 of 47 results")

**Library-Wide Search**
- Searches indexed text across all books
- Results show: book title, matched snippet, page number
- Click result → opens book to that page with highlight

**External API Search**
- Search Google Books / Open Library
- Results show: cover, title, author, description, ISBN
- "Add to Library" button (if PDF available)
- "Preview" button (opens external link)

#### 3.7.3 Technical Implementation
- Extract text during upload (pdf.js `getTextContent()`)
- Store in `books.text_indexed` column (tsvector)
- Postgres full-text search with ranking:
```sql
SELECT * FROM books
WHERE text_indexed @@ to_tsquery('modern & physics')
ORDER BY ts_rank(text_indexed, to_tsquery('modern & physics')) DESC;
```

#### 3.7.4 Acceptance Criteria
- [ ] In-book search returns results in <500ms
- [ ] Library search returns results in <1s
- [ ] Search supports partial words (e.g., "phys" matches "physics")
- [ ] Case-insensitive matching
- [ ] Search history (last 10 queries)
- [ ] External API results cached for 24 hours

---

### 3.8 Page Management

#### 3.8.1 User Stories
- **US-070:** As a user, I want to reorder pages so I can reorganize content
- **US-071:** As a user, I want to rotate pages so I can fix orientation
- **US-072:** As a user, I want to delete pages so I can remove unwanted content

#### 3.8.2 Acceptance Criteria
- [ ] Thumbnail view with drag-to-reorder
- [ ] Rotate button (90° increments)
- [ ] Multi-select for bulk operations
- [ ] Delete with confirmation dialog
- [ ] Undo operation (version history)
- [ ] Changes create new book version (non-destructive)

---

### 3.9 Offline Support

#### 3.9.1 User Stories
- **US-080:** As a user, I want to read books offline so I can use the app without internet
- **US-081:** As a user, I want my offline annotations to sync when I reconnect

#### 3.9.2 Acceptance Criteria
- [ ] Last 5 opened books cached locally
- [ ] Manual "Download for Offline" option
- [ ] Offline indicator in UI (airplane icon)
- [ ] Annotations saved locally, queued for sync
- [ ] Graceful degradation: disable cloud-only features offline
- [ ] Automatic sync on reconnect with progress indicator

---

## 4. Non-Functional Requirements

### 4.1 Performance
- PDF load time: <2s for 100-page document
- Annotation latency: <50ms (local), <2s (sync)
- Search response time: <1s
- App launch time: <3s

### 4.2 Scalability
- Support books up to 5000 pages
- Handle 10,000+ books per user
- 100,000 concurrent users (server)

### 4.3 Accessibility
- WCAG 2.1 AA compliance
- Screen reader support (VoiceOver, TalkBack)
- Keyboard navigation (web)
- High contrast mode
- Font scaling up to 200%

### 4.4 Security
- All data encrypted in transit (TLS 1.3)
- At-rest encryption (Supabase default)
- Signed URLs expire in 1 hour
- Row Level Security on all tables
- No sensitive data in logs

### 4.5 Privacy
- GDPR-compliant data export
- Account deletion removes all user data within 30 days
- No tracking without consent
- Optional E2E encryption (post-MVP)

---

## 5. Out of Scope (Post-MVP)

- Desktop native apps (Electron)
- Advanced OCR (handwriting recognition)
- Real-time collaborative editing
- Video/audio annotations
- Form filling & digital signatures
- DRM-protected content
- Enterprise SSO (SAML)
- API for third-party integrations

---

## 6. Success Criteria & Launch Readiness

### 6.1 MVP Launch Checklist
- [ ] All US-001 to US-081 acceptance criteria met
- [ ] 100+ hours of QA testing
- [ ] <0.1% crash rate
- [ ] <5s average API response time
- [ ] Security audit passed
- [ ] App store review guidelines compliance
- [ ] Privacy policy & Terms of Service published
- [ ] Customer support infrastructure ready

### 6.2 Post-Launch Metrics (30 days)
- 10,000+ downloads
- 30% D1 retention
- 4.0+ star rating (App Store/Play Store)
- <1% churn rate
- 50+ paying subscribers

---

## 7. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PDF.js performance on large files | High | Medium | Implement virtual scrolling, page caching |
| CRDT sync conflicts | Medium | Low | Extensive testing, fallback to manual merge |
| Supabase scaling limits | High | Low | Monitor usage, plan migration to dedicated |
| App store rejection | Medium | Medium | Follow guidelines, prepare appeals |
| Low user adoption | High | Medium | Beta testing, marketing campaign |

---

## 8. Dependencies & Assumptions

### 8.1 External Dependencies
- Supabase availability (99.9% SLA)
- Google Books API rate limits
- App store approval timelines
- Payment processor (Stripe) integration

### 8.2 Assumptions
- Users have stable internet for sync (or understand offline limitations)
- Target devices: iPhone 11+, Android 10+, modern browsers
- Users accept freemium model with 100MB free storage limit

---

**Next Steps:**
1. Review & approve PRD with stakeholders
2. Create detailed wireframes (see WIREFRAMES.md)
3. Finalize technical architecture (see ARCHITECTURE.md)
4. Begin sprint planning
