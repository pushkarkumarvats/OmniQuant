# BookFlow - Command Reference
Quick reference for common development and deployment commands

---

## 🚀 Development

### Setup
```bash
# Install dependencies
npm install

# Start Supabase (requires Docker)
supabase start

# Generate TypeScript types from database
npm run supabase:gen-types

# Copy environment variables
cp .env.example .env
# Then edit .env with your Supabase keys
```

### Running the App
```bash
# Start Expo dev server
npm start

# Run on iOS simulator
npm run ios

# Run on Android emulator
npm run android

# Run on web
npm run web

# Clear cache and restart
npx expo start -c
```

---

## 🧪 Testing & Quality

### Code Quality
```bash
# Run linter
npm run lint

# Fix linting issues
npm run lint -- --fix

# Format code
npm run format

# Check formatting
npm run format -- --check

# Type check
npm run type-check
```

### Testing
```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Generate coverage report
npm run test:coverage

# Run specific test file
npm test src/__tests__/store/authStore.test.ts
```

---

## 🗄️ Database

### Supabase Local
```bash
# Start Supabase
supabase start

# Stop Supabase
supabase stop

# Check status
supabase status

# Reset database (reapply migrations)
supabase db reset

# View logs
supabase logs

# Open Studio in browser
open http://localhost:54323
```

### Migrations
```bash
# Create new migration
supabase migration new migration_name

# Apply migrations to local
supabase db reset

# Push migrations to production
supabase db push

# Pull changes from production
supabase db pull
```

### Type Generation
```bash
# Generate TypeScript types from schema
npm run supabase:gen-types

# Or directly
supabase gen types typescript --local > src/types/database.ts
```

---

## 📦 Building

### Web
```bash
# Build for web
npm run build:web

# Output will be in: web-build/
```

### Mobile (EAS)
```bash
# Install EAS CLI globally
npm install -g eas-cli

# Login to Expo
eas login

# Configure EAS (first time)
eas build:configure

# Build for iOS (preview)
eas build --platform ios --profile preview

# Build for iOS (production)
eas build --platform ios --profile production

# Build for Android (preview)
eas build --platform android --profile preview

# Build for Android (production)
eas build --platform android --profile production

# Build for both platforms
eas build --platform all --profile production

# Check build status
eas build:list
```

---

## 🚀 Deployment

### Web (Vercel)
```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy to preview
vercel

# Deploy to production
vercel --prod

# Check deployment status
vercel ls
```

### Mobile App Stores
```bash
# Submit iOS to App Store
eas submit --platform ios

# Submit Android to Play Store
eas submit --platform android

# Check submission status
eas submit:list
```

### Supabase Production
```bash
# Link to production project
supabase link --project-ref your-project-ref

# Push migrations to production
supabase db push

# Deploy edge function
supabase functions deploy function-name

# Set secrets for functions
supabase secrets set SECRET_NAME=value

# List all secrets
supabase secrets list
```

---

## 🔧 Troubleshooting

### Clear Caches
```bash
# Clear Expo cache
npx expo start -c

# Clear npm cache
npm cache clean --force

# Clear watchman (macOS)
watchman watch-del-all

# Clear Metro bundler
rm -rf $TMPDIR/metro-*
rm -rf $TMPDIR/haste-map-*

# Reinstall dependencies
rm -rf node_modules
npm install
```

### iOS Specific
```bash
# Clean iOS build
cd ios
rm -rf Pods Podfile.lock
cd ..

# Reinstall pods
cd ios
pod install
cd ..

# Reset iOS simulator
npx expo run:ios --device
```

### Android Specific
```bash
# Clean Android build
cd android
./gradlew clean
cd ..

# List Android devices
adb devices

# Reverse port for local Supabase
adb reverse tcp:54321 tcp:54321
```

### Database Issues
```bash
# Reset local database
supabase db reset

# Check database connection
supabase db ping

# View database logs
supabase logs db

# Manually connect to database
supabase db connect
```

---

## 📊 Monitoring

### View Logs
```bash
# Expo logs (mobile)
npx expo logs:android
npx expo logs:ios

# Supabase logs
supabase logs api
supabase logs db
supabase logs auth
```

### Performance
```bash
# Analyze bundle size
npx expo-size

# Run Lighthouse (web)
npx lighthouse http://localhost:19006 --view

# Profile app performance
npx react-devtools
```

---

## 🔐 Environment Management

### Environment Files
```bash
# Development
.env

# Production
.env.production

# Switch between environments
# Edit .env to use production values
```

### Secrets
```bash
# View environment variables
echo $EXPO_PUBLIC_SUPABASE_URL

# Set secret in Vercel
vercel env add EXPO_PUBLIC_SUPABASE_URL

# Set secret in EAS
eas secret:create --name EXPO_PUBLIC_SUPABASE_URL --value "your-value"

# List EAS secrets
eas secret:list
```

---

## 📝 Git Commands

### Common Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Stage changes
git add .

# Commit with message
git commit -m "feat: add new feature"

# Push to remote
git push origin feature/new-feature

# Create pull request (on GitHub)

# Merge to main
git checkout main
git merge feature/new-feature

# Tag release
git tag v1.0.0
git push origin v1.0.0
```

### Commit Message Convention
```bash
# Format: <type>: <description>

# Types:
feat: New feature
fix: Bug fix
docs: Documentation changes
style: Code style changes
refactor: Code refactoring
test: Adding tests
chore: Maintenance tasks

# Examples:
git commit -m "feat: add dark mode"
git commit -m "fix: resolve PDF loading issue"
git commit -m "docs: update setup guide"
```

---

## 🔄 Updates

### Update Dependencies
```bash
# Check for outdated packages
npm outdated

# Update all dependencies
npm update

# Update Expo SDK
npx expo upgrade

# Update specific package
npm install package-name@latest

# Check for security vulnerabilities
npm audit

# Fix vulnerabilities
npm audit fix
```

---

## 🎨 Asset Management

### Generate App Icons
```bash
# Place icon.png (1024x1024) in assets/
# Run build and Expo will generate all sizes

# Or manually with ImageMagick
convert icon.png -resize 512x512 icon-512.png
```

### Optimize Images
```bash
# Install imagemin
npm install -g imagemin-cli

# Optimize PNG
imagemin assets/*.png --out-dir=assets/optimized

# Optimize JPG
imagemin assets/*.jpg --plugin=mozjpeg --out-dir=assets/optimized
```

---

## 📱 Testing on Physical Devices

### iOS
```bash
# Start dev server
npm start

# Scan QR code with Camera app
# Or use Expo Go app
```

### Android
```bash
# Start dev server
npm start

# Scan QR code with Expo Go app

# Or install directly
npx expo run:android
```

---

## 🔍 Debugging

### React Native Debugger
```bash
# Open Chrome DevTools
# Press 'j' in Expo CLI

# Or use React DevTools
npm install -g react-devtools
react-devtools
```

### Network Debugging
```bash
# Use Flipper
npx react-native flipper

# Or use Reactotron
npm install --save-dev reactotron-react-native
```

---

## 📚 Documentation

### Generate Docs
```bash
# TypeDoc for API documentation
npm install -g typedoc
typedoc --out docs src/

# Compodoc for Angular-style docs
npm install -g @compodoc/compodoc
compodoc -p tsconfig.json
```

---

## Quick Reference Card

```bash
# MOST USED COMMANDS

# Development
npm start              # Start dev server
npm run ios            # Run iOS
npm run android        # Run Android
npm run web            # Run web

# Database
supabase start         # Start Supabase
supabase db reset      # Reset database
npm run supabase:gen-types  # Generate types

# Quality
npm run lint           # Check code
npm test               # Run tests
npm run type-check     # Check types

# Build
eas build --platform ios --profile production
eas build --platform android --profile production
npm run build:web

# Deploy
vercel --prod          # Deploy web
eas submit --platform ios
eas submit --platform android
```

---

## 🆘 Need Help?

- 📖 Full documentation: `docs/` folder
- 🚀 Quick start: `QUICK_START.md`
- 🔧 Setup guide: `docs/SETUP.md`
- 🚢 Deployment: `DEPLOYMENT.md`
- ❓ Issues: Check GitHub Issues

---

**Bookmark this file for quick reference!** 🔖
