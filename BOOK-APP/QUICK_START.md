# BookFlow - Quick Start Guide
## Get Running in 5 Minutes

---

## Prerequisites Check

```bash
# Verify Node.js (need 18+)
node --version

# Verify npm
npm --version

# Install Supabase CLI
npm install -g supabase

# Verify Docker is running (required for Supabase local)
docker --version
```

---

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (2 min)

```bash
cd BOOK-APP
npm install
```

### Step 2: Start Supabase (1 min)

```bash
supabase start
```

**Copy the output:**
```
API URL: http://localhost:54321
anon key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
service_role key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Step 3: Configure Environment (30 sec)

```bash
cp .env.example .env
```

Edit `.env` and paste your keys:
```env
EXPO_PUBLIC_SUPABASE_URL=http://localhost:54321
EXPO_PUBLIC_SUPABASE_ANON_KEY=<paste-anon-key-here>
```

### Step 4: Generate Types (30 sec)

```bash
npm run supabase:gen-types
```

### Step 5: Run App (1 min)

```bash
npm start
```

Press:
- **`w`** - Open in web browser
- **`i`** - Open iOS simulator (macOS only)
- **`a`** - Open Android emulator

---

## 📱 Access Points

Once running:

- **Web App:** http://localhost:19006
- **Supabase Studio:** http://localhost:54323
- **API Docs:** http://localhost:54321/rest/v1/

---

## 🧪 Test the App

1. **Sign Up:**
   - Open web app
   - Click "Sign Up"
   - Enter: `test@bookflow.app` / `password123`
   - Create account

2. **View Database:**
   - Open Supabase Studio (http://localhost:54323)
   - Navigate to Table Editor
   - See your user in `users` table

3. **Explore API:**
   - Open http://localhost:54321/rest/v1/books
   - Should see `[]` (empty array - no books yet)

---

## 🛠️ Development Commands

```bash
# Start development server
npm start

# Run on specific platform
npm run ios          # iOS simulator
npm run android      # Android emulator
npm run web          # Web browser

# Code quality
npm run lint         # Check code style
npm run format       # Auto-format code
npm run type-check   # Check TypeScript

# Testing
npm test             # Run unit tests
npm run test:watch   # Watch mode

# Database
supabase status      # Check Supabase status
supabase db reset    # Reset database (reapply migrations)
supabase stop        # Stop Supabase

# Build for production
npm run build:web    # Build web app
eas build --platform ios       # Build iOS (requires EAS account)
eas build --platform android   # Build Android
```

---

## 📂 Project Structure Overview

```
BOOK-APP/
├── docs/                   # 📚 Complete documentation
│   ├── PRD.md             # Product requirements
│   ├── ARCHITECTURE.md    # Technical design
│   ├── API.md             # API reference
│   ├── WIREFRAMES.md      # UI/UX specs
│   └── SETUP.md           # Detailed setup guide
│
├── supabase/              # 🗄️ Backend configuration
│   ├── migrations/        # Database schema
│   └── config.toml        # Supabase config
│
├── src/                   # 💻 Application code
│   ├── app/              # Pages (Expo Router)
│   ├── services/         # API clients
│   ├── store/            # State management
│   ├── types/            # TypeScript types
│   └── ...               # Components, hooks, utils
│
├── package.json          # Dependencies
├── .env                  # Environment variables (create this!)
└── README.md             # Project overview
```

---

## 🐛 Common Issues

### "Supabase CLI not found"
```bash
npm install -g supabase
```

### "Docker not running"
```bash
# Start Docker Desktop
# Then: supabase start
```

### "Port already in use"
```bash
# Stop Supabase
supabase stop

# Kill processes on ports
lsof -ti:54321 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :54321   # Windows (then kill PID)
```

### "Cannot find module '@supabase/supabase-js'"
```bash
npm install
```

### "Module not found: Error: Can't resolve '@/'"
```bash
# Clear Metro cache
npx expo start -c
```

---

## 🎯 What to Build Next

**Immediate Tasks:**
1. ✅ Foundation complete
2. 🚧 Finish auth screens (signup, forgot password)
3. 🚧 Build library screen
4. 🚧 Implement file upload
5. 🚧 Create PDF viewer
6. 🚧 Add annotation tools
7. 🚧 Set up sync

**See `/docs/PRD.md` for full roadmap**

---

## 📖 Documentation

| Doc | What's Inside |
|-----|---------------|
| **README.md** | Project overview & features |
| **SETUP.md** | Detailed installation guide |
| **ARCHITECTURE.md** | Database schema, API design, sync strategy |
| **API.md** | All API endpoints with examples |
| **WIREFRAMES.md** | UI mockups & component specs |
| **PRD.md** | User stories & acceptance criteria |
| **PROJECT_SUMMARY.md** | Complete project status |

---

## 💡 Quick Tips

**Hot Reload:**
- Save any file in `src/` → app reloads automatically
- No need to restart server

**View Logs:**
- Metro bundler logs in terminal
- Web: Browser DevTools console
- Mobile: `npx expo logs:android` or `npx expo logs:ios`

**Database Changes:**
```bash
# Create new migration
supabase migration new add_feature

# Edit supabase/migrations/<timestamp>_add_feature.sql
# Then apply:
supabase db reset
```

**Type Safety:**
```bash
# After changing database schema:
npm run supabase:gen-types
```

---

## 🔗 Resources

- **Expo Docs:** https://docs.expo.dev
- **Supabase Docs:** https://supabase.com/docs
- **React Native:** https://reactnative.dev
- **TypeScript:** https://www.typescriptlang.org

---

## ✅ Verification Checklist

- [ ] Node.js 18+ installed
- [ ] Docker running
- [ ] Supabase started successfully
- [ ] `.env` file created with correct keys
- [ ] `npm install` completed without errors
- [ ] `npm start` launches Expo dev server
- [ ] Web app opens at localhost:19006
- [ ] Can create account on login screen
- [ ] Supabase Studio accessible at localhost:54323

**All checked?** You're ready to develop! 🎉

---

## 🆘 Need Help?

1. Check `/docs/SETUP.md` for detailed troubleshooting
2. Review error messages carefully
3. Search GitHub Issues
4. Ask your team

---

**Now start building! 🚀**

```bash
npm start
```
