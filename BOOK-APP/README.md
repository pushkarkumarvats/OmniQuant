# BookFlow - Cross-Platform Book & PDF Editor

**Mobile (Android/iOS) + Web PDF/Book Reader, Editor & Sync Platform**

## 🎯 Elevator Pitch

BookFlow is a cross-platform application that allows users to import, read, annotate, and edit books and PDFs across mobile and web devices with seamless cloud synchronization. Built with React Native (Expo) for mobile and web, powered by Supabase for backend services.

## 🎨 Target Users

- **Students** - Annotate textbooks, research papers
- **Researchers** - Organize and search academic literature
- **Professionals** - Review contracts, reports, and documentation
- **Casual Readers** - Manage personal book collections
- **Small Publishers/Self-Publishers** - Review and edit manuscripts

## ✨ MVP Features (Phase 1)

### Core Functionality
- ✅ **Authentication** - Email/password + OAuth via Supabase Auth
- ✅ **PDF Upload** - From device or URL, stored in Supabase Storage
- ✅ **Library Management** - List/grid view, metadata display
- ✅ **PDF Viewer** - Cross-platform rendering (pdf.js web, native mobile)
- ✅ **Annotations**
  - Highlight text
  - Underline/strikeout
  - Freehand drawing
  - Sticky notes & comments
- ✅ **Search** - Full-text search across library + external book API integration
- ✅ **Real-time Sync** - Annotations, bookmarks, reading position across devices
- ✅ **Themes** - Dark/Light mode, font scaling
- ✅ **Offline Support** - Read cached documents, sync when online
- ✅ **Page Management** - Reorder, rotate, delete pages
- ✅ **Security** - HTTPS, signed URLs, RLS policies

## 🚀 Post-MVP Features (Phase 2+)

- Advanced PDF text editing (PSPDFKit/PDFTron integration)
- Merge/split/convert PDFs
- Real-time collaborative annotations
- OCR for scanned documents
- Multi-format support (ePub, MOBI)
- Subscription/IAP monetization
- Analytics & A/B testing

## 🏗️ Tech Stack

### Frontend
- **React Native + Expo** - Cross-platform (iOS, Android, Web)
- **React Navigation** - Navigation
- **Zustand** - State management
- **React Query** - Server state & caching
- **pdf.js** - Web PDF rendering
- **react-native-pdf** - Native mobile rendering
- **Yjs/Automerge** - CRDT for annotation sync

### Backend
- **Supabase** - Auth, Postgres DB, Storage, Realtime
- **Supabase Edge Functions** - Serverless compute (OCR, conversions)
- **Postgres Full-Text Search** - tsvector indexing

### DevOps
- **GitHub Actions** - CI/CD
- **EAS Build** - Mobile app builds
- **Vercel/Netlify** - Web hosting
- **Sentry** - Error monitoring
- **PostHog** - Analytics

## 📁 Project Structure

```
bookflow/
├── docs/                      # Documentation
│   ├── PRD.md                # Product Requirements
│   ├── ARCHITECTURE.md       # Technical Architecture
│   ├── API.md               # API Specification
│   └── WIREFRAMES.md        # UI/UX Designs
├── supabase/                 # Supabase configuration
│   ├── migrations/          # Database migrations
│   ├── functions/           # Edge Functions
│   └── config.toml          # Supabase config
├── src/
│   ├── app/                 # Expo Router pages
│   ├── components/          # Reusable components
│   ├── features/            # Feature modules
│   │   ├── auth/
│   │   ├── library/
│   │   ├── reader/
│   │   ├── annotations/
│   │   └── sync/
│   ├── hooks/               # Custom hooks
│   ├── services/            # API & business logic
│   ├── store/               # State management
│   ├── types/               # TypeScript types
│   └── utils/               # Utilities
├── assets/                   # Images, fonts
└── package.json
```

## 🗄️ Database Schema

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed schema.

**Core Tables:**
- `users` - User profiles & preferences
- `books` - Book/PDF metadata
- `annotations` - User annotations (CRDT deltas)
- `book_edits` - Page-level edits & versions
- `bookmarks` - Reading positions
- `devices` - Device registry for sync

## 🔄 Sync Strategy

- **CRDT-based** - Yjs for annotations (deterministic merge)
- **Optimistic Updates** - Apply locally, sync in background
- **Conflict Resolution** - Automatic for annotations, version history for major edits
- **Real-time** - Supabase Realtime WebSocket subscriptions

## 🔐 Security & Privacy

- HTTPS everywhere
- Signed URLs (short expiry) for file access
- Row Level Security (RLS) on all tables
- Encryption at rest (Supabase/cloud provider)
- Optional E2E encryption for private libraries
- Virus scanning on uploads
- GDPR/CCPA compliance (export/delete)

## 📊 Success Metrics

- **DAU/MAU** - Daily/Monthly active users
- **Retention** - D1, D7, D30 retention rates
- **Engagement** - Books uploaded, annotations created per user
- **Conversion** - Free → Paid tier conversion
- **Performance** - PDF load time, sync latency

## 🛣️ Roadmap

### Week 0-2: Foundation
- Product spec finalization
- Wireframes & design system
- Tech stack validation
- Supabase project setup

### Week 3-6: Auth & Storage
- User authentication flow
- File upload/download
- Basic library UI
- Supabase integration

### Week 7-10: Reader & Annotations
- PDF viewer implementation
- Annotation tools (highlight, notes, draw)
- Local persistence
- Annotation UI

### Week 11-14: Sync
- Supabase Realtime integration
- CRDT implementation
- Sync status UI
- Conflict handling

### Week 15-18: Search & Offline
- Full-text search
- External book API integration
- OCR pipeline (serverless)
- Offline caching

### Week 19-22: Advanced Features
- Page management (reorder/rotate/delete)
- Export/merge basics
- QA & bug fixes

### Week 23-26: Launch Prep
- Performance optimization
- Paid SDK evaluation (optional)
- App store submission
- Marketing & launch

## 🚦 Getting Started

See [docs/SETUP.md](docs/SETUP.md) for development setup instructions.

## 📄 License

MIT License - See LICENSE file for details
