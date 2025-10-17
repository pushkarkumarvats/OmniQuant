# Development Setup Guide
## BookFlow - Getting Started

---

## Prerequisites

### Required Software
- **Node.js**: 18.x or higher ([Download](https://nodejs.org/))
- **npm**: 9.x or higher (comes with Node.js)
- **Git**: Latest version
- **Supabase CLI**: Latest version
- **Expo CLI**: Will be installed via npm

### Platform-Specific Requirements

**iOS Development:**
- macOS required
- Xcode 15+ ([Mac App Store](https://apps.apple.com/app/xcode/id497799835))
- CocoaPods: `sudo gem install cocoapods`
- iOS Simulator or physical iOS device

**Android Development:**
- Android Studio ([Download](https://developer.android.com/studio))
- Android SDK (API 33+)
- Java JDK 17
- Android Emulator or physical Android device

**Web Development:**
- Modern browser (Chrome, Firefox, Safari, Edge)

---

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/bookflow.git
cd bookflow
```

### 2. Install Dependencies

```bash
npm install
```

This will install all required packages including:
- React Native & Expo
- Supabase client
- PDF libraries
- UI components

### 3. Install Supabase CLI

```bash
npm install -g supabase
```

Or with Homebrew (macOS):
```bash
brew install supabase/tap/supabase
```

### 4. Set Up Supabase Locally

```bash
# Start Supabase local instance (Docker required)
supabase start

# This will output:
# - API URL (http://localhost:54321)
# - Anon key
# - Service role key
```

**Save these credentials** - you'll need them for environment variables.

### 5. Run Database Migrations

```bash
supabase db reset
```

This applies all migrations in `supabase/migrations/`.

### 6. Configure Environment Variables

Create `.env` file in project root:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Supabase (Local Development)
EXPO_PUBLIC_SUPABASE_URL=http://localhost:54321
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-from-supabase-start

# Supabase (Production - uncomment when deploying)
# EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
# EXPO_PUBLIC_SUPABASE_ANON_KEY=your-production-anon-key

# Google Books API (optional for external search)
EXPO_PUBLIC_GOOGLE_BOOKS_API_KEY=your-google-books-api-key

# Sentry (Error Monitoring)
SENTRY_DSN=your-sentry-dsn

# Environment
NODE_ENV=development
```

### 7. Generate TypeScript Types from Supabase

```bash
npm run supabase:gen-types
```

This creates `src/types/database.ts` with your database schema types.

---

## Running the App

### Development Server

```bash
npm start
```

This starts Expo Dev Server. You'll see:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Metro waiting on exp://192.168.1.100:19000            │
│   QR Code                                               │
│                                                         │
│   › Press a │ open Android                             │
│   › Press i │ open iOS simulator                       │
│   › Press w │ open web                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Platform-Specific Commands

**iOS:**
```bash
npm run ios

# Or with specific simulator
npx expo run:ios --device "iPhone 15 Pro"
```

**Android:**
```bash
npm run android

# Or with specific emulator
npx expo run:android --device emulator-5554
```

**Web:**
```bash
npm run web
```

Opens browser at http://localhost:19006

---

## Development Workflow

### 1. Start Supabase

```bash
supabase start
```

Runs Postgres, Realtime, Storage locally via Docker.

### 2. Start Expo Dev Server

```bash
npm start
```

### 3. Open on Device/Simulator

- **iOS**: Press `i` or scan QR with Camera app
- **Android**: Press `a` or scan QR with Expo Go app
- **Web**: Press `w`

### 4. Make Changes

Edit files in `src/` - changes hot reload automatically.

### 5. View Logs

- Metro logs in terminal
- Device logs: `npx expo logs:android` or `npx expo logs:ios`
- Web console: Browser DevTools

---

## Database Management

### View Database

Supabase Studio (local): http://localhost:54323

### Create New Migration

```bash
supabase migration new migration_name
```

Edit file in `supabase/migrations/`, then:

```bash
supabase db reset  # Apply migration
```

### Seed Database

Create `supabase/seed.sql`:

```sql
-- Insert test data
INSERT INTO public.users (id, email, full_name)
VALUES (
  'test-user-uuid',
  'test@bookflow.app',
  'Test User'
);
```

Apply:
```bash
supabase db reset
```

### Inspect Database

```bash
# Connect to Postgres
supabase db connect

# Then run SQL
SELECT * FROM books;
```

---

## Testing

### Run Unit Tests

```bash
npm test
```

### Run Tests in Watch Mode

```bash
npm run test:watch
```

### Generate Coverage Report

```bash
npm run test:coverage
```

### E2E Tests (Future)

```bash
# Install Detox
npm install -g detox-cli

# Build app
detox build -c ios.sim.debug

# Run tests
detox test -c ios.sim.debug
```

---

## Code Quality

### Linting

```bash
npm run lint
```

Fix auto-fixable issues:
```bash
npm run lint -- --fix
```

### Formatting

```bash
npm run format
```

Check formatting:
```bash
npm run format -- --check
```

### Type Checking

```bash
npm run type-check
```

---

## Building for Production

### iOS

```bash
# Install EAS CLI
npm install -g eas-cli

# Login to Expo
eas login

# Configure project
eas build:configure

# Build
eas build --platform ios --profile production
```

### Android

```bash
eas build --platform android --profile production
```

### Web

```bash
npm run build:web

# Output in web-build/
# Deploy to Vercel/Netlify
```

---

## Supabase Cloud Setup (Production)

### 1. Create Supabase Project

Go to [Supabase Dashboard](https://app.supabase.com/)
- Click "New Project"
- Set project name, database password
- Choose region (close to users)

### 2. Link Local to Cloud

```bash
supabase link --project-ref your-project-ref
```

### 3. Push Migrations

```bash
supabase db push
```

### 4. Update Environment Variables

Update `.env` with production credentials:

```env
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-production-anon-key
```

### 5. Configure Storage Buckets

In Supabase Dashboard → Storage:
- Create `books` bucket (private)
- Create `thumbnails` bucket (public)
- Set size limits, MIME types

### 6. Configure Authentication

In Supabase Dashboard → Authentication:
- Enable Email provider
- Configure redirect URLs
- Add OAuth providers (Google, Apple)
- Set JWT expiry

### 7. Deploy Edge Functions

```bash
supabase functions deploy process-upload
supabase functions deploy ocr-document
supabase functions deploy search-external
```

---

## Troubleshooting

### Metro Bundle Error

```bash
# Clear cache
npx expo start -c
```

### iOS Pods Install Fails

```bash
cd ios
pod install --repo-update
cd ..
```

### Android Build Fails

```bash
cd android
./gradlew clean
cd ..
```

### Supabase Connection Issues

Check:
1. Supabase running: `supabase status`
2. Correct URL/keys in `.env`
3. Network firewall not blocking localhost

### Type Errors After Schema Change

```bash
npm run supabase:gen-types
npm run type-check
```

---

## Project Structure

```
bookflow/
├── src/                    # Application code
│   ├── app/               # Expo Router pages
│   ├── components/        # Reusable components
│   ├── features/          # Feature modules
│   ├── services/          # API services
│   ├── store/             # State management
│   ├── hooks/             # Custom hooks
│   ├── utils/             # Utilities
│   └── types/             # TypeScript types
├── supabase/              # Backend configuration
│   ├── migrations/        # Database migrations
│   ├── functions/         # Edge Functions
│   └── config.toml        # Supabase config
├── assets/                # Images, fonts, etc.
├── docs/                  # Documentation
├── .env                   # Environment variables (gitignored)
├── app.json              # Expo configuration
├── package.json          # Dependencies
└── tsconfig.json         # TypeScript config
```

---

## Environment-Specific Notes

### Local Development
- Supabase runs on localhost
- Fast iteration with hot reload
- Use Expo Go app for quick testing

### Staging
- Supabase staging project
- TestFlight (iOS) / Internal testing (Android)
- Staging API keys

### Production
- Supabase production project
- App Store / Play Store
- Production API keys
- Analytics enabled

---

## Useful Commands Cheat Sheet

```bash
# Development
npm start                    # Start Expo dev server
npm run ios                  # Run on iOS
npm run android              # Run on Android
npm run web                  # Run on web

# Supabase
supabase start               # Start local Supabase
supabase stop                # Stop local Supabase
supabase status              # Check status
supabase db reset            # Reset DB (apply migrations)
supabase migration new NAME  # Create migration

# Code Quality
npm run lint                 # Lint code
npm run format               # Format code
npm run type-check           # Check types
npm test                     # Run tests

# Build
eas build --platform ios     # Build iOS
eas build --platform android # Build Android
npm run build:web            # Build web

# Deploy
eas submit                   # Submit to app stores
supabase functions deploy    # Deploy edge functions
```

---

## Next Steps

1. ✅ Set up development environment
2. ✅ Run app locally
3. 📝 Read [ARCHITECTURE.md](./ARCHITECTURE.md) for technical details
4. 📝 Review [PRD.md](./PRD.md) for feature specifications
5. 🎨 Check [WIREFRAMES.md](./WIREFRAMES.md) for UI designs
6. 💻 Start building features!

---

## Getting Help

- **Documentation**: [docs/](./docs)
- **Expo Docs**: https://docs.expo.dev
- **Supabase Docs**: https://supabase.com/docs
- **React Native Docs**: https://reactnative.dev
- **Issue Tracker**: GitHub Issues
- **Team Chat**: Slack #bookflow-dev

---

**Happy Coding! 🚀**
