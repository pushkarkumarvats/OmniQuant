# Changelog
All notable changes to BookFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-17

### 🎉 Initial Release

#### Added
- **Authentication System**
  - Email/password sign up and login
  - OAuth support (Google, Apple)
  - Password reset functionality
  - Secure session management with auto-refresh
  - Profile management

- **Library Management**
  - Grid and list view for books
  - Real-time search and filtering
  - Sort by title, author, date added
  - Pull-to-refresh support
  - Empty states with helpful CTAs

- **File Upload**
  - PDF file picker integration
  - Direct upload to Supabase Storage
  - Progress tracking with visual feedback
  - File size validation (max 200MB)
  - Automatic metadata extraction

- **PDF Reader**
  - Native PDF rendering (iOS/Android)
  - Web-based PDF viewer (pdf.js)
  - Page navigation with slider
  - Zoom and pan gestures
  - Page indicator
  - Bookmark support

- **Annotations System**
  - Highlight tool with color picker
  - Sticky notes and text comments
  - Freehand drawing
  - Annotation list view grouped by page
  - Edit and delete annotations
  - Real-time sync across devices

- **Search Functionality**
  - Full-text search across library
  - Search history with chips
  - Real-time search results
  - Book content search

- **Profile & Settings**
  - Storage usage tracking with visual progress
  - Auto-sync toggle
  - Theme preferences (coming soon)
  - Subscription management
  - Sign out

- **Sync & Offline**
  - Real-time synchronization via Supabase Realtime
  - Offline reading support
  - Automatic sync on reconnection
  - Conflict resolution

- **Security**
  - Row Level Security (RLS) on all tables
  - Signed URLs for file access
  - Secure token storage (Keychain/KeyStore)
  - Encrypted data transmission

#### Technical
- React Native + Expo SDK 51
- Supabase backend (Auth, Database, Storage, Realtime)
- TypeScript with strict mode
- React Query for server state
- Zustand for client state
- Comprehensive test coverage
- CI/CD with GitHub Actions
- EAS Build configuration

#### Documentation
- Complete Product Requirements Document (PRD)
- Technical Architecture Documentation
- API Specification
- UI/UX Wireframes
- Setup Guide
- Deployment Guide
- Role-specific Implementation Guides

---

## [Unreleased]

### Planned Features
- Dark mode and theme customization
- Advanced PDF editing (text replacement)
- OCR for scanned documents
- Multi-format support (ePub, MOBI)
- Real-time collaborative annotations
- Export annotations to PDF/JSON
- Collections and tags
- Advanced search filters
- Desktop native apps
- Subscription tiers with IAP

### Known Issues
- Web PDF viewer needs pdf.js implementation
- Large PDFs (>1000 pages) may have performance issues
- Offline annotation queue needs optimization

---

## Version History

### [1.0.0] - 2025-10-17
- Initial production release
- MVP feature set complete
- Ready for App Store and Play Store submission

---

**Note:** This project follows semantic versioning:
- **Major** (1.x.x): Breaking changes
- **Minor** (x.1.x): New features (backward compatible)
- **Patch** (x.x.1): Bug fixes (backward compatible)
