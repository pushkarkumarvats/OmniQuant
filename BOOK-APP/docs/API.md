# API Specification
## BookFlow REST API & WebSocket Protocol

**Base URL:** `https://<your-project>.supabase.co`  
**API Version:** v1  
**Authentication:** Bearer JWT tokens

---

## Authentication

All API requests (except auth endpoints) require a valid JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### POST /auth/v1/signup
Create a new user account.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "data": {
    "full_name": "John Doe"
  }
}
```

**Response (201):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "...",
  "user": {
    "id": "uuid-1234",
    "email": "user@example.com",
    "created_at": "2025-10-17T10:00:00Z"
  }
}
```

### POST /auth/v1/token?grant_type=password
Login with email/password.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

**Response (200):**
```json
{
  "access_token": "...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "...",
  "user": { ... }
}
```

### POST /auth/v1/token?grant_type=refresh_token
Refresh access token.

**Request:**
```json
{
  "refresh_token": "your-refresh-token"
}
```

---

## Books

### GET /rest/v1/books
List user's books with pagination and filtering.

**Query Parameters:**
- `select` (string): Fields to return (default: `*`)
- `order` (string): Sort order (e.g., `created_at.desc`)
- `limit` (int): Max results (default: 20, max: 100)
- `offset` (int): Pagination offset
- `title` (string): Filter by title (case-insensitive)
- `authors` (string[]): Filter by authors

**Example Request:**
```
GET /rest/v1/books?select=*&order=last_opened_at.desc&limit=20
```

**Response (200):**
```json
[
  {
    "id": "book-uuid-1",
    "owner_id": "user-uuid",
    "title": "Introduction to Algorithms",
    "authors": ["Thomas H. Cormen", "Charles E. Leiserson"],
    "description": "Comprehensive algorithms textbook",
    "language": "en",
    "isbn": "978-0262033848",
    "file_type": "pdf",
    "storage_path": "user-uuid/book-uuid-1.pdf",
    "file_size_bytes": 45678901,
    "page_count": 1312,
    "source": "uploaded",
    "thumbnail_url": "https://storage.url/thumbnails/book-uuid-1.jpg",
    "created_at": "2025-10-10T08:30:00Z",
    "updated_at": "2025-10-17T10:30:00Z",
    "last_opened_at": "2025-10-17T09:15:00Z"
  }
]
```

### POST /rest/v1/books
Create a new book entry (after file upload).

**Request:**
```json
{
  "title": "My Research Paper",
  "authors": ["Author Name"],
  "file_type": "pdf",
  "storage_path": "user-uuid/book-uuid-2.pdf",
  "file_size_bytes": 2345678,
  "page_count": 25,
  "source": "uploaded"
}
```

**Response (201):**
```json
{
  "id": "book-uuid-2",
  "owner_id": "user-uuid",
  "title": "My Research Paper",
  ...
}
```

### GET /rest/v1/books/:id
Get single book details.

**Response (200):**
```json
{
  "id": "book-uuid-1",
  "owner_id": "user-uuid",
  "title": "Introduction to Algorithms",
  ...
}
```

### PATCH /rest/v1/books/:id
Update book metadata.

**Request:**
```json
{
  "title": "Updated Title",
  "last_opened_at": "2025-10-17T11:00:00Z"
}
```

**Response (200):**
```json
{
  "id": "book-uuid-1",
  "title": "Updated Title",
  ...
}
```

### DELETE /rest/v1/books/:id
Delete a book (also deletes file from storage).

**Response (204):** No content

---

## Annotations

### GET /rest/v1/annotations
Get annotations for a book.

**Query Parameters:**
- `book_id` (required): Filter by book ID (e.g., `book_id=eq.book-uuid-1`)
- `page` (optional): Filter by page (e.g., `page=eq.5`)
- `type` (optional): Filter by type (e.g., `type=eq.highlight`)
- `order` (string): Sort order (default: `page.asc,created_at.asc`)

**Example:**
```
GET /rest/v1/annotations?book_id=eq.book-uuid-1&page=eq.5
```

**Response (200):**
```json
[
  {
    "id": "annot-uuid-1",
    "book_id": "book-uuid-1",
    "user_id": "user-uuid",
    "page": 5,
    "type": "highlight",
    "coords": {
      "x": 0.12,
      "y": 0.45,
      "width": 0.6,
      "height": 0.03
    },
    "color": "#FFEB3B",
    "selected_text": "Important concept here",
    "content": null,
    "device_id": "device-abc",
    "version": 1,
    "created_at": "2025-10-17T10:20:00Z",
    "updated_at": "2025-10-17T10:20:00Z"
  },
  {
    "id": "annot-uuid-2",
    "book_id": "book-uuid-1",
    "user_id": "user-uuid",
    "page": 5,
    "type": "note",
    "coords": { "x": 0.8, "y": 0.1, "width": 0.05, "height": 0.05 },
    "color": "#FFC107",
    "selected_text": null,
    "content": "Remember to review this section",
    "device_id": "device-abc",
    "version": 1,
    "created_at": "2025-10-17T10:25:00Z",
    "updated_at": "2025-10-17T10:25:00Z"
  }
]
```

### POST /rest/v1/annotations
Create a new annotation.

**Request:**
```json
{
  "book_id": "book-uuid-1",
  "page": 12,
  "type": "highlight",
  "coords": {
    "x": 0.15,
    "y": 0.3,
    "width": 0.5,
    "height": 0.025
  },
  "color": "#4CAF50",
  "selected_text": "This is the highlighted text",
  "device_id": "device-xyz"
}
```

**Response (201):**
```json
{
  "id": "annot-uuid-3",
  "book_id": "book-uuid-1",
  "user_id": "user-uuid",
  "page": 12,
  ...
}
```

### PATCH /rest/v1/annotations/:id
Update an existing annotation.

**Request:**
```json
{
  "color": "#2196F3",
  "content": "Added a note to this highlight"
}
```

**Response (200):**
```json
{
  "id": "annot-uuid-3",
  "color": "#2196F3",
  "content": "Added a note to this highlight",
  ...
}
```

### DELETE /rest/v1/annotations/:id
Delete an annotation.

**Response (204):** No content

---

## Bookmarks

### GET /rest/v1/bookmarks
Get bookmarks (reading progress).

**Query Parameters:**
- `book_id` (required): Filter by book
- `user_id` (optional): Filter by user (defaults to current user)

**Example:**
```
GET /rest/v1/bookmarks?book_id=eq.book-uuid-1
```

**Response (200):**
```json
[
  {
    "id": "bookmark-uuid-1",
    "book_id": "book-uuid-1",
    "user_id": "user-uuid",
    "page": 47,
    "scroll_offset": 0.35,
    "location_json": {
      "zoom": 1.5,
      "viewMode": "single"
    },
    "created_at": "2025-10-15T08:00:00Z",
    "updated_at": "2025-10-17T10:45:00Z"
  }
]
```

### POST /rest/v1/bookmarks
Create or update bookmark (upsert).

**Request:**
```json
{
  "book_id": "book-uuid-1",
  "page": 48,
  "scroll_offset": 0.12
}
```

**Response (201 or 200):**
```json
{
  "id": "bookmark-uuid-1",
  "book_id": "book-uuid-1",
  "user_id": "user-uuid",
  "page": 48,
  "scroll_offset": 0.12,
  ...
}
```

---

## Storage

### POST /storage/v1/object/books/:path
Upload a PDF file.

**Request:**
```
POST /storage/v1/object/books/user-uuid/book-uuid-1.pdf
Content-Type: application/pdf
Body: <binary PDF data>
```

**Response (200):**
```json
{
  "Key": "books/user-uuid/book-uuid-1.pdf",
  "Bucket": "books"
}
```

### GET /storage/v1/object/sign/books/:path
Create a signed URL for file access.

**Query Parameters:**
- `expiresIn` (int): Expiry time in seconds (default: 3600)

**Example:**
```
GET /storage/v1/object/sign/books/user-uuid/book-uuid-1.pdf?expiresIn=3600
```

**Response (200):**
```json
{
  "signedURL": "https://storage.url/books/user-uuid/book-uuid-1.pdf?token=..."
}
```

### DELETE /storage/v1/object/books/:path
Delete a file from storage.

**Response (200):**
```json
{
  "message": "Successfully deleted"
}
```

---

## Edge Functions

### POST /functions/v1/process-upload
Process uploaded PDF (extract metadata, generate thumbnail).

**Request:**
```json
{
  "book_id": "book-uuid-1",
  "storage_path": "user-uuid/book-uuid-1.pdf"
}
```

**Response (200):**
```json
{
  "success": true,
  "metadata": {
    "title": "Extracted Title",
    "authors": ["Author Name"],
    "page_count": 150
  },
  "thumbnail_url": "https://storage.url/thumbnails/book-uuid-1.jpg",
  "text_length": 45678
}
```

### POST /functions/v1/ocr-document
Request OCR processing for scanned PDF.

**Request:**
```json
{
  "book_id": "book-uuid-1",
  "pages": [1, 2, 3],
  "language": "eng"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "job-uuid-1",
  "status": "queued",
  "estimated_completion": "2025-10-17T11:00:00Z"
}
```

### GET /functions/v1/search-external
Search external book APIs (Google Books, Open Library).

**Query Parameters:**
- `q` (string): Search query
- `source` (string): API source (google | openlibrary)
- `limit` (int): Max results (default: 10)

**Example:**
```
GET /functions/v1/search-external?q=algorithms&source=google&limit=5
```

**Response (200):**
```json
{
  "results": [
    {
      "id": "external-book-1",
      "title": "Introduction to Algorithms",
      "authors": ["Thomas H. Cormen"],
      "publisher": "MIT Press",
      "published_date": "2009",
      "isbn": "978-0262033848",
      "description": "Comprehensive algorithms textbook...",
      "cover_url": "https://covers.openlibrary.org/b/isbn/978-0262033848-L.jpg",
      "pdf_url": null,
      "preview_link": "https://books.google.com/books?id=..."
    }
  ],
  "total": 47
}
```

---

## Realtime (WebSocket)

### Connection

**URL:** `wss://<your-project>.supabase.co/realtime/v1/websocket`

**Authentication:**
```json
{
  "event": "phx_join",
  "topic": "realtime:public:annotations",
  "payload": {
    "config": {
      "broadcast": { "self": false },
      "presence": { "key": "" }
    },
    "access_token": "your-jwt-token"
  },
  "ref": "1"
}
```

### Subscribe to Annotations Channel

**Topic:** `realtime:public:annotations:book_id=eq.{book_id}`

**Message Format:**
```json
{
  "event": "postgres_changes",
  "payload": {
    "data": {
      "id": "annot-uuid-4",
      "book_id": "book-uuid-1",
      "type": "highlight",
      "page": 15,
      ...
    },
    "commit_timestamp": "2025-10-17T10:50:00Z",
    "eventType": "INSERT"
  }
}
```

**Event Types:**
- `INSERT` - New annotation created
- `UPDATE` - Annotation modified
- `DELETE` - Annotation deleted

### Subscribe to Bookmark Changes

**Topic:** `realtime:public:bookmarks:book_id=eq.{book_id}`

---

## Batch Operations

### POST /rest/v1/rpc/batch_create_annotations
Create multiple annotations in a single request.

**Request:**
```json
{
  "annotations": [
    {
      "book_id": "book-uuid-1",
      "page": 10,
      "type": "highlight",
      "coords": { "x": 0.1, "y": 0.2, "width": 0.5, "height": 0.03 },
      "color": "#FFEB3B"
    },
    {
      "book_id": "book-uuid-1",
      "page": 11,
      "type": "note",
      "coords": { "x": 0.8, "y": 0.1, "width": 0.05, "height": 0.05 },
      "content": "Review this"
    }
  ]
}
```

**Response (200):**
```json
{
  "created": [
    { "id": "annot-uuid-5", ... },
    { "id": "annot-uuid-6", ... }
  ]
}
```

---

## Full-Text Search

### GET /rest/v1/rpc/search_books
Search across all books' text content.

**Request:**
```json
{
  "search_query": "machine learning",
  "limit": 10
}
```

**Response (200):**
```json
[
  {
    "book_id": "book-uuid-1",
    "title": "Introduction to Machine Learning",
    "match_snippet": "...fundamentals of machine learning algorithms...",
    "page": 5,
    "rank": 0.95
  }
]
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error message",
  "code": "error_code",
  "details": "Additional context",
  "hint": "Suggested fix"
}
```

**Common Error Codes:**
- `401` - Unauthorized (invalid or expired token)
- `403` - Forbidden (no permission)
- `404` - Not found
- `422` - Unprocessable entity (validation error)
- `429` - Rate limit exceeded
- `500` - Internal server error

**Example:**
```json
{
  "error": "You do not have permission to access this book",
  "code": "PGRST301",
  "details": "Row Level Security policy violation",
  "hint": "Check book ownership or sharing permissions"
}
```

---

## Rate Limits

- **Anonymous:** 60 requests/minute
- **Authenticated:** 600 requests/minute
- **File uploads:** 10 uploads/hour (free tier)

**Rate limit headers:**
```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 583
X-RateLimit-Reset: 1697548800
```

---

## Pagination

Use `Range` header for pagination:

**Request:**
```
GET /rest/v1/books
Range: 0-19
```

**Response:**
```
Content-Range: 0-19/147
```

---

## CORS

Allowed origins configured in Supabase dashboard.

**Response headers:**
```
Access-Control-Allow-Origin: https://yourapp.com
Access-Control-Allow-Methods: GET, POST, PATCH, DELETE
Access-Control-Allow-Headers: Authorization, Content-Type
```

---

**Next:** See [WIREFRAMES.md](./WIREFRAMES.md) for UI/UX specifications.
