# BookFlow - Deployment Guide
**Complete guide to deploy BookFlow to production**

---

## 🚀 Pre-Deployment Checklist

### 1. Environment Setup
- [ ] Supabase production project created
- [ ] All environment variables configured
- [ ] OAuth credentials obtained (Google, Apple)
- [ ] API keys secured (Google Books, Sentry, PostHog)
- [ ] Domain name configured (if using custom domain)

### 2. Code Quality
- [ ] All tests passing (`npm test`)
- [ ] Linting passes (`npm run lint`)
- [ ] Type checking passes (`npm run type-check`)
- [ ] No console.log statements in production code
- [ ] All TODOs resolved or documented

### 3. Security
- [ ] Row Level Security policies tested
- [ ] File upload size limits configured
- [ ] Rate limiting implemented
- [ ] Signed URLs have proper expiry
- [ ] Sensitive data encrypted

---

## 📋 Step-by-Step Deployment

### Phase 1: Supabase Production Setup

#### 1.1 Create Supabase Project
```bash
# Go to https://app.supabase.com/
# Click "New Project"
# Name: bookflow-prod
# Database Password: <generate-strong-password>
# Region: <closest-to-users>
```

#### 1.2 Apply Database Migrations
```bash
# Link local to production
supabase link --project-ref <your-project-ref>

# Push migrations
supabase db push

# Verify tables created
# Check in Supabase Dashboard → Database → Tables
```

#### 1.3 Configure Storage Buckets
```bash
# In Supabase Dashboard → Storage:
# 1. Create "books" bucket (private)
#    - Max file size: 200MB
#    - Allowed MIME types: application/pdf
#
# 2. Create "thumbnails" bucket (public)
#    - Max file size: 5MB
#    - Allowed MIME types: image/jpeg, image/png, image/webp
```

#### 1.4 Set Up Authentication
```bash
# In Supabase Dashboard → Authentication → Providers:
# 1. Enable Email provider
# 2. Configure Email templates
# 3. Add OAuth providers:
#    - Google: Add Client ID & Secret
#    - Apple: Add Service ID & Key
# 4. Set redirect URLs:
#    - https://yourapp.com/auth/callback
#    - https://yourdomain.com/*
```

#### 1.5 Deploy Edge Functions
```bash
# Deploy all functions
supabase functions deploy process-upload
supabase functions deploy ocr-document
supabase functions deploy search-external

# Set environment variables for functions
supabase secrets set GOOGLE_BOOKS_API_KEY=<your-key>
```

---

### Phase 2: Web Deployment (Vercel)

#### 2.1 Install Vercel CLI
```bash
npm install -g vercel
```

#### 2.2 Build Web App
```bash
npm run build:web
```

#### 2.3 Deploy to Vercel
```bash
# Login
vercel login

# Deploy
cd web-build
vercel --prod

# Or configure automatic deployment:
# 1. Push to GitHub
# 2. Import project in Vercel dashboard
# 3. Set environment variables in Vercel:
#    - EXPO_PUBLIC_SUPABASE_URL
#    - EXPO_PUBLIC_SUPABASE_ANON_KEY
#    - SENTRY_DSN
#    - POSTHOG_API_KEY
```

#### 2.4 Configure Custom Domain (Optional)
```bash
# In Vercel dashboard:
# Settings → Domains → Add Domain
# Add DNS records as instructed
```

---

### Phase 3: Mobile App Deployment

#### 3.1 Install EAS CLI
```bash
npm install -g eas-cli
```

#### 3.2 Configure EAS
```bash
# Login to Expo
eas login

# Initialize EAS
eas build:configure
```

#### 3.3 Update Production Environment
Edit `.env.production`:
```env
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-production-anon-key
SENTRY_DSN=your-production-sentry-dsn
POSTHOG_API_KEY=your-production-posthog-key
NODE_ENV=production
```

#### 3.4 Build iOS App
```bash
# Build for iOS
eas build --platform ios --profile production

# Download IPA when complete
# Upload to App Store Connect via Transporter or:
eas submit --platform ios
```

**iOS App Store Requirements:**
- [ ] App Icon (1024x1024px)
- [ ] Screenshots (all required sizes)
- [ ] Privacy Policy URL
- [ ] App Description & Keywords
- [ ] App Store Connect account ($99/year)
- [ ] Apple Developer Program membership

#### 3.5 Build Android App
```bash
# Build for Android
eas build --platform android --profile production

# Submit to Play Store
eas submit --platform android
```

**Android Play Store Requirements:**
- [ ] App Icon (512x512px)
- [ ] Feature Graphic (1024x500px)
- [ ] Screenshots (at least 2)
- [ ] Privacy Policy URL
- [ ] App Description
- [ ] Google Play Console account ($25 one-time)

---

### Phase 4: Monitoring & Analytics Setup

#### 4.1 Sentry (Error Monitoring)
```bash
# Install Sentry
npm install --save sentry-expo

# Configure in app
# Update src/app/_layout.tsx to initialize Sentry
```

```typescript
import * as Sentry from 'sentry-expo';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  enableInExpoDevelopment: false,
  debug: false,
});
```

#### 4.2 PostHog (Analytics)
```bash
# Install PostHog
npm install posthog-react-native

# Track events
posthog.capture('book_uploaded', {
  book_id: bookId,
  file_size: fileSize,
});
```

---

### Phase 5: Production Configuration

#### 5.1 Environment Variables
Create `.env.production`:
```env
# Supabase
EXPO_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# External APIs
EXPO_PUBLIC_GOOGLE_BOOKS_API_KEY=your-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
POSTHOG_API_KEY=your-posthog-key

# Feature Flags
EXPO_PUBLIC_ENABLE_OCR=true
EXPO_PUBLIC_ENABLE_COLLABORATION=false

# Limits
EXPO_PUBLIC_MAX_FILE_SIZE=209715200
EXPO_PUBLIC_FREE_TIER_STORAGE=104857600
```

#### 5.2 Update app.json
```json
{
  "expo": {
    "name": "BookFlow",
    "slug": "bookflow",
    "version": "1.0.0",
    "ios": {
      "bundleIdentifier": "com.yourcompany.bookflow",
      "buildNumber": "1"
    },
    "android": {
      "package": "com.yourcompany.bookflow",
      "versionCode": 1
    },
    "extra": {
      "eas": {
        "projectId": "your-eas-project-id"
      }
    }
  }
}
```

---

## 🔧 Post-Deployment Tasks

### 1. Verify Functionality
- [ ] Sign up new user
- [ ] Upload PDF
- [ ] View PDF in reader
- [ ] Create annotations
- [ ] Test sync across devices
- [ ] Test search
- [ ] Test offline mode

### 2. Performance Testing
```bash
# Use Lighthouse for web
npx lighthouse https://yourapp.com --view

# Check bundle size
npx expo-size

# Monitor API response times in Supabase Dashboard
```

### 3. Set Up Monitoring Alerts
```bash
# Supabase Dashboard → Settings → Notifications
# - Database usage > 80%
# - API errors > 100/hour
# - Storage usage > 80%

# Sentry Dashboard → Alerts
# - Error rate > 1%
# - New error types

# Vercel Dashboard → Monitoring
# - Response time > 1s
# - Failed requests
```

---

## 📊 Production Monitoring

### Key Metrics to Track

**User Metrics:**
- DAU/MAU ratio
- New signups per day
- Retention (D1, D7, D30)
- Churn rate

**Performance Metrics:**
- API response time (p50, p95, p99)
- PDF load time
- Sync latency
- Error rate

**Business Metrics:**
- Books uploaded per user
- Annotations created per user
- Free → Paid conversion
- Storage usage per user

### Monitoring Tools
- **Supabase Dashboard**: Database & API metrics
- **Sentry**: Error tracking & performance
- **PostHog**: User analytics & funnels
- **Vercel Analytics**: Web performance
- **Google Analytics**: User behavior (optional)

---

## 🔄 Continuous Deployment

### GitHub Actions Workflow
The `.github/workflows/ci.yml` file is configured to:
1. Run tests on every PR
2. Deploy web to Vercel on main branch
3. Build mobile apps on main branch (optional)

### Manual Deployment Commands
```bash
# Web
npm run build:web
vercel --prod

# iOS
eas build --platform ios --profile production
eas submit --platform ios

# Android
eas build --platform android --profile production
eas submit --platform android
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Supabase Connection Failed**
- Verify `.env` has correct URL and keys
- Check RLS policies allow access
- Ensure JWT token is valid

**2. File Upload Fails**
- Check storage bucket permissions
- Verify file size under limit (200MB)
- Check MIME type is allowed

**3. App Store Rejection**
- Review App Store guidelines
- Ensure privacy policy is accessible
- Add required usage descriptions in Info.plist

**4. Build Errors**
```bash
# Clear cache
npx expo start -c

# Reinstall dependencies
rm -rf node_modules
npm install

# Check for type errors
npm run type-check
```

---

## 🔐 Security Best Practices

1. **Never commit secrets**
   - Use `.env` files (gitignored)
   - Use GitHub Secrets for CI/CD
   - Rotate keys periodically

2. **Keep dependencies updated**
   ```bash
   npm audit
   npm audit fix
   npm outdated
   ```

3. **Monitor for vulnerabilities**
   - Enable Dependabot alerts
   - Review Sentry errors
   - Check Supabase logs

4. **Implement rate limiting**
   - Add rate limits to API endpoints
   - Use Supabase edge functions for validation

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks
- **Weekly**: Review error logs, check metrics
- **Monthly**: Update dependencies, review user feedback
- **Quarterly**: Security audit, performance optimization

### Rollback Procedure
```bash
# Vercel (web)
vercel rollback

# Mobile (via EAS)
# Publish OTA update with previous version
eas update --branch production --message "Rollback"
```

---

## ✅ Deployment Checklist Summary

### Pre-Launch
- [ ] All features tested
- [ ] Environment variables set
- [ ] Database migrations applied
- [ ] Storage buckets configured
- [ ] OAuth providers configured
- [ ] Monitoring tools integrated

### Launch
- [ ] Web deployed to production
- [ ] iOS app submitted to App Store
- [ ] Android app submitted to Play Store
- [ ] DNS configured (if custom domain)
- [ ] SSL certificates active

### Post-Launch
- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Respond to user feedback
- [ ] Plan next features

---

## 🎉 You're Ready to Launch!

**Estimated Time to Production:**
- Supabase setup: 2-3 hours
- Web deployment: 1 hour
- Mobile deployment: 4-6 hours (including app store review)
- Testing & monitoring: 2-3 hours

**Total: 1-2 days for complete deployment**

---

**Questions?**
- Review [README.md](./README.md) for project overview
- Check [SETUP.md](./docs/SETUP.md) for development setup
- Consult [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for technical details

**Good luck with your launch! 🚀**
