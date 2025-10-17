# UI/UX Wireframes & Design Specifications
## BookFlow - Mobile & Web Interface

---

## Design System

### Color Palette

**Light Theme:**
```
Primary:     #1976D2 (Blue)
Secondary:   #FF6F00 (Orange)
Success:     #4CAF50 (Green)
Warning:     #FFC107 (Amber)
Error:       #F44336 (Red)
Background:  #FFFFFF
Surface:     #F5F5F5
Text:        #212121
TextSecondary: #757575
```

**Dark Theme:**
```
Primary:     #42A5F5 (Light Blue)
Secondary:   #FFB74D (Light Orange)
Success:     #66BB6A (Light Green)
Warning:     #FFCA28 (Light Amber)
Error:       #EF5350 (Light Red)
Background:  #121212
Surface:     #1E1E1E
Text:        #FFFFFF
TextSecondary: #B0B0B0
```

**Annotation Colors:**
```
Highlight Yellow:  #FFEB3B
Highlight Green:   #4CAF50
Highlight Blue:    #2196F3
Highlight Pink:    #E91E63
Highlight Orange:  #FF9800
Note Yellow:       #FFC107
Underline Red:     #F44336
Strikeout Gray:    #9E9E9E
```

### Typography

```
Font Family: System Default (San Francisco iOS, Roboto Android, -apple-system web)

Heading 1:   32px, Bold, 1.2 line-height
Heading 2:   24px, Bold, 1.3 line-height
Heading 3:   20px, SemiBold, 1.4 line-height
Body Large:  16px, Regular, 1.5 line-height
Body:        14px, Regular, 1.5 line-height
Body Small:  12px, Regular, 1.4 line-height
Caption:     10px, Medium, 1.2 line-height
Button:      14px, SemiBold, 1.0 line-height
```

### Spacing System

```
xs:  4px
sm:  8px
md:  16px
lg:  24px
xl:  32px
xxl: 48px
```

### Border Radius

```
Small:  4px  (buttons, inputs)
Medium: 8px  (cards)
Large:  16px (modals, sheets)
Round:  9999px (chips, avatars)
```

### Shadows

```
Small:  0 1px 3px rgba(0,0,0,0.12)
Medium: 0 2px 8px rgba(0,0,0,0.15)
Large:  0 4px 16px rgba(0,0,0,0.18)
```

---

## Screen Layouts

### 1. Authentication Screens

#### 1.1 Login Screen
```
┌─────────────────────────────────────┐
│                                     │
│            [Logo]                   │
│         BookFlow                    │
│                                     │
│    Welcome Back                     │
│    Sign in to continue              │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Email                         │ │
│  │ [email@example.com]           │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Password                      │ │
│  │ [••••••••••]          [👁]    │ │
│  └───────────────────────────────┘ │
│                                     │
│           Forgot Password?          │
│                                     │
│  ┌───────────────────────────────┐ │
│  │      Sign In                  │ │
│  └───────────────────────────────┘ │
│                                     │
│        ───── OR ─────               │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  [G] Continue with Google     │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  [🍎] Continue with Apple     │ │
│  └───────────────────────────────┘ │
│                                     │
│    Don't have an account? Sign Up   │
│                                     │
└─────────────────────────────────────┘
```

**Components:**
- `TextInput` (email, password)
- `Button` (primary, secondary, outlined)
- `Link` (forgot password, sign up)
- `Divider` (OR separator)

---

### 2. Library Screen (Home)

#### 2.1 Grid View
```
┌─────────────────────────────────────┐
│  [≡]  Library        [🔍] [⋮]       │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │ [🔍] Search books...        │   │
│  └─────────────────────────────┘   │
│                                     │
│  [Recent ▼] [Grid/List] [Sort]     │
│                                     │
│  ┌─────────┐  ┌─────────┐          │
│  │ [Cover] │  │ [Cover] │          │
│  │         │  │         │          │
│  │ Title   │  │ Title   │          │
│  │ Author  │  │ Author  │          │
│  │ ████ 75%│  │ ██░░ 25%│          │
│  └─────────┘  └─────────┘          │
│                                     │
│  ┌─────────┐  ┌─────────┐          │
│  │ [Cover] │  │ [Cover] │          │
│  │         │  │         │          │
│  │ Title   │  │ Title   │          │
│  │ Author  │  │ Author  │          │
│  │ ░░░░  0%│  │ ████100%│          │
│  └─────────┘  └─────────┘          │
│                                     │
├─────────────────────────────────────┤
│ [📚] [🔍] [➕] [👤]                 │
└─────────────────────────────────────┘
```

**Components:**
- `SearchBar` (filterable)
- `Dropdown` (filters: Recent, All Books, Collections)
- `ToggleButton` (Grid/List view)
- `BookCard` (thumbnail, title, author, progress bar)
- `BottomNavigation` (Library, Search, Upload, Profile)

#### 2.2 List View
```
┌─────────────────────────────────────┐
│  [≡]  Library        [🔍] [⋮]       │
├─────────────────────────────────────┤
│  [Recent ▼] [Grid/List] [Sort]     │
│                                     │
│  ┌───┬─────────────────────────┬─┐ │
│  │[T]│ Introduction to ML      │⋮│ │
│  │ h │ Author Name             │ │ │
│  │ m │ 150 pages • 75% read    │ │ │
│  └───┴─────────────────────────┴─┘ │
│                                     │
│  ┌───┬─────────────────────────┬─┐ │
│  │[T]│ React Native Guide      │⋮│ │
│  │ h │ John Doe                │ │ │
│  │ m │ 240 pages • 25% read    │ │ │
│  └───┴─────────────────────────┴─┘ │
│                                     │
│  ┌───┬─────────────────────────┬─┐ │
│  │[T]│ Design Patterns         │⋮│ │
│  │ h │ Gang of Four            │ │ │
│  │ m │ 395 pages • Unread      │ │ │
│  └───┴─────────────────────────┴─┘ │
│                                     │
└─────────────────────────────────────┘
```

---

### 3. PDF Reader Screen

```
┌─────────────────────────────────────┐
│  [←] Book Title           [🔖] [⋮]  │ Top Bar
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 🖊  💡  ✏️  🗑  T  ↩ ↪      │   │ Annotation Toolbar
│  └─────────────────────────────┘   │
│                                     │
│  ╔═══════════════════════════════╗ │
│  ║  PDF Content Area             ║ │
│  ║                               ║ │
│  ║  This is the main text content║ │
│  ║  that the user is reading.    ║ │
│  ║  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀         ║ │ Highlight overlay
│  ║  highlighted text here        ║ │
│  ║  ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔         ║ │
│  ║                               ║ │
│  ║  More content...              ║ │
│  ║                          [📝] ║ │ Sticky note icon
│  ║                               ║ │
│  ║  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~║ │ Freehand drawing
│  ║                               ║ │
│  ╚═══════════════════════════════╝ │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ [<] ━━━━●━━━━ [>]  12 / 250│   │ Page Slider
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

**Annotation Toolbar Icons:**
- 🖊 Highlight
- 💡 Note
- ✏️ Draw
- 🗑 Eraser
- T Text
- ↩ Undo
- ↪ Redo

**Interaction States:**

**Highlight Selection:**
```
┌─────────────────────────────────────┐
│  Selected: "important concept"      │
│                                     │
│  ┌──────────────────────────────┐  │
│  │ 🟡 Yellow                     │  │
│  │ 🟢 Green                      │  │
│  │ 🔵 Blue                       │  │
│  │ 🔴 Pink                       │  │
│  │ 🟠 Orange                     │  │
│  ├──────────────────────────────┤  │
│  │ ✏️ Add Note                   │  │
│  │ 🗑️ Delete                     │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Note Popup:**
```
┌─────────────────────────────────────┐
│  Add Note                      [×]  │
│                                     │
│  ┌───────────────────────────────┐ │
│  │                               │ │
│  │ Type your note here...        │ │
│  │                               │ │
│  │                               │ │
│  └───────────────────────────────┘ │
│                                     │
│  Color: 🟡 🟢 🔵 🔴 🟠             │
│                                     │
│         [Cancel]    [Save]          │
└─────────────────────────────────────┘
```

---

### 4. Annotations List Screen

```
┌─────────────────────────────────────┐
│  [←] Annotations            [Filter]│
├─────────────────────────────────────┤
│  [All] [Highlights] [Notes] [Draws] │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Page 12              Oct 17 │   │
│  │ ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀         │   │
│  │ "This is highlighted text"  │   │
│  │ ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔         │   │
│  │ 💭 My note: Important!      │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Page 45              Oct 16 │   │
│  │ ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀         │   │
│  │ "Another key concept here"  │   │
│  │ ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔         │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Page 78              Oct 15 │   │
│  │ 📝 Note                     │   │
│  │ "Review this section later" │   │
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

**Tap annotation → navigates to that page in reader**

---

### 5. Upload Screen

```
┌─────────────────────────────────────┐
│  [←] Upload Book                    │
├─────────────────────────────────────┤
│                                     │
│         ┌──────────────┐            │
│         │              │            │
│         │   📄 + 📁    │            │
│         │              │            │
│         │ Tap to select│            │
│         │ or drag file │            │
│         │              │            │
│         └──────────────┘            │
│                                     │
│         Supported formats:          │
│         PDF, ePub (coming soon)     │
│         Max size: 200MB             │
│                                     │
│  ───────────── OR ─────────────     │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 🔗 Import from URL          │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 🔍 Search Book Database     │   │
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

**Upload Progress:**
```
┌─────────────────────────────────────┐
│  Uploading...                       │
├─────────────────────────────────────┤
│                                     │
│  📄 Introduction to ML.pdf          │
│                                     │
│  ████████████████░░░░░░  67%        │
│                                     │
│  15.2 MB / 22.8 MB                  │
│  Estimated: 12 seconds              │
│                                     │
│           [Cancel Upload]           │
│                                     │
└─────────────────────────────────────┘
```

---

### 6. Search Screen

```
┌─────────────────────────────────────┐
│  [🔍] Search                   [×]  │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │ Search your library...      │   │
│  └─────────────────────────────┘   │
│                                     │
│  [My Books] [External]              │
│                                     │
│  Recent Searches:                   │
│  • machine learning                 │
│  • design patterns                  │
│  • algorithms                       │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  Search Results (24)                │
│                                     │
│  📘 Introduction to ML              │
│  Page 45: "...fundamentals of       │
│  machine learning algorithms..."    │
│                                     │
│  📗 Deep Learning                   │
│  Page 12: "...neural networks and   │
│  machine learning techniques..."    │
│                                     │
└─────────────────────────────────────┘
```

**External Search Results:**
```
┌─────────────────────────────────────┐
│  External Database                  │
├─────────────────────────────────────┤
│  ┌───┬─────────────────────────┬─┐ │
│  │[C]│ Deep Learning           │+│ │
│  │ o │ Ian Goodfellow          │ │ │
│  │ v │ MIT Press, 2016         │ │ │
│  │ e │ 800 pages               │ │ │
│  │ r │ [Preview]               │ │ │
│  └───┴─────────────────────────┴─┘ │
│                                     │
│  ┌───┬─────────────────────────┬─┐ │
│  │[C]│ Pattern Recognition     │+│ │
│  │ o │ Christopher Bishop      │ │ │
│  │ v │ Springer, 2006          │ │ │
│  │ e │ 738 pages               │ │ │
│  │ r │ [Preview]               │ │ │
│  └───┴─────────────────────────┴─┘ │
└─────────────────────────────────────┘
```

---

### 7. Profile/Settings Screen

```
┌─────────────────────────────────────┐
│  [←] Profile                        │
├─────────────────────────────────────┤
│         ┌─────────┐                 │
│         │  [👤]   │                 │
│         └─────────┘                 │
│       John Doe                      │
│    john@example.com                 │
│                                     │
│  ──────────────────────────────     │
│                                     │
│  Storage: 45.2 MB / 100 MB          │
│  ████████░░░░░░░░░░  45%           │
│                                     │
│  ──────────────────────────────     │
│                                     │
│  📚 My Library                      │
│  └─ 24 books                        │
│                                     │
│  🎨 Appearance                      │
│  └─ Light Mode                      │
│                                     │
│  🔔 Notifications                   │
│  └─ Enabled                         │
│                                     │
│  ☁️ Sync & Backup                   │
│  └─ Auto-sync: On                   │
│                                     │
│  💾 Offline Storage                 │
│  └─ 5 books downloaded              │
│                                     │
│  💳 Subscription                    │
│  └─ Free Plan                       │
│                                     │
│  ⚙️ Advanced Settings               │
│                                     │
│  📄 Privacy Policy                  │
│                                     │
│  🚪 Log Out                         │
│                                     │
└─────────────────────────────────────┘
```

---

### 8. Sync Status Panel

```
┌─────────────────────────────────────┐
│  Sync Status                   [×]  │
├─────────────────────────────────────┤
│                                     │
│  ●  All changes synced              │
│      Last sync: 2 minutes ago       │
│                                     │
│  ──────────────────────────────     │
│                                     │
│  Recent Activity:                   │
│                                     │
│  ✅ Annotation added (Page 12)      │
│     1 minute ago                    │
│                                     │
│  ✅ Bookmark updated                │
│     2 minutes ago                   │
│                                     │
│  ✅ Book uploaded                   │
│     10 minutes ago                  │
│                                     │
│  Synced across 3 devices:           │
│  • iPhone 15 Pro                    │
│  • iPad Air                         │
│  • Web Browser                      │
│                                     │
│         [Force Sync Now]            │
│                                     │
└─────────────────────────────────────┘
```

**Sync Error State:**
```
┌─────────────────────────────────────┐
│  ⚠️ Sync Issues                     │
├─────────────────────────────────────┤
│                                     │
│  ⚠️  2 pending changes              │
│      Unable to sync                 │
│                                     │
│  ──────────────────────────────     │
│                                     │
│  Failed Items:                      │
│                                     │
│  ❌ Annotation on Page 45           │
│     Error: Network timeout          │
│     [Retry]                         │
│                                     │
│  ❌ Book upload: React Guide.pdf    │
│     Error: File too large           │
│     [Remove from queue]             │
│                                     │
│         [Retry All]                 │
│                                     │
└─────────────────────────────────────┘
```

---

### 9. Page Management (Thumbnail View)

```
┌─────────────────────────────────────┐
│  [←] Edit Pages            [Save]   │
├─────────────────────────────────────┤
│  [Reorder] [Rotate] [Delete]        │
│                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐           │
│  │  1  │ │  2  │ │  3  │ [☑]       │
│  │ [T] │ │ [T] │ │ [T] │           │
│  │  h  │ │  h  │ │  h  │           │
│  │  m  │ │  m  │ │  m  │           │
│  │  b] │ │  b] │ │  b] │           │
│  └─────┘ └─────┘ └─────┘           │
│                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐           │
│  │  4  │ │  5  │ │  6  │           │
│  │ [T] │ │ [T] │ │ [T] │           │
│  │  h  │ │  h  │ │  h  │           │
│  │  m  │ │  m  │ │  m  │           │
│  │  b] │ │  b] │ │  b] │           │
│  └─────┘ └─────┘ └─────┘           │
│                                     │
└─────────────────────────────────────┘
```

**Actions:**
- Drag thumbnails to reorder
- Tap rotate icon to rotate 90°
- Select multiple and delete
- Changes create new version (non-destructive)

---

## Responsive Breakpoints

### Mobile (< 768px)
- Single column layout
- Bottom navigation
- Full-screen modals
- Touch-optimized controls (44x44pt minimum)

### Tablet (768px - 1024px)
- Two-column layout for library
- Side drawer navigation
- Modal sheets (not full screen)
- Dual-page reader mode

### Web (> 1024px)
- Multi-column library grid (3-4 columns)
- Sidebar navigation always visible
- Keyboard shortcuts enabled
- Hover states on interactive elements

---

## Accessibility Specifications

### Screen Reader Labels
```typescript
// Example labels
<Button accessibilityLabel="Upload new book">
  <Icon name="plus" />
</Button>

<BookCard 
  accessibilityLabel={`${title} by ${author}, ${progress}% read`}
  accessibilityHint="Tap to open book"
/>
```

### Focus Order
1. Navigation elements
2. Primary actions
3. Content items
4. Secondary actions
5. Footer elements

### Keyboard Shortcuts (Web)
```
Ctrl/Cmd + K:    Search
Ctrl/Cmd + U:    Upload
Ctrl/Cmd + N:    New annotation
Ctrl/Cmd + Z:    Undo
Ctrl/Cmd + Y:    Redo
Left/Right Arrow: Previous/Next page
Esc:             Close modal
```

---

## Animation & Transitions

### Page Transitions
- Duration: 300ms
- Easing: ease-in-out
- Type: slide (mobile), fade (web)

### Button Press
- Scale: 0.95
- Duration: 100ms

### Modal Appearance
- Slide up from bottom (mobile)
- Fade + scale from center (web)
- Duration: 250ms

### Loading States
- Skeleton screens for content
- Shimmer animation (1.5s loop)
- Progress bars for uploads

---

## Component Library

### Buttons
```
Primary:    Filled, Primary color
Secondary:  Outlined, Primary color
Tertiary:   Text only
Destructive: Filled, Error color
Disabled:   Gray, 40% opacity
```

### Cards
```
Elevation: Medium shadow
Border Radius: 8px
Padding: 16px
Background: Surface color
```

### Inputs
```
Height: 48px
Border: 1px solid (outline)
Border Radius: 4px
Focus: 2px primary border
Error: 2px error border
```

### Bottom Sheets
```
Max Height: 90% viewport
Border Radius: 16px top corners
Drag handle: 32px wide, 4px tall
Backdrop: rgba(0,0,0,0.5)
```

---

## Empty States

### Empty Library
```
┌─────────────────────────────────────┐
│                                     │
│            📚                       │
│                                     │
│      No books yet                   │
│                                     │
│  Start building your library        │
│  by uploading your first book       │
│                                     │
│  ┌───────────────────────────────┐ │
│  │    Upload Your First Book     │ │
│  └───────────────────────────────┘ │
│                                     │
│         or                          │
│                                     │
│     Search Book Database            │
│                                     │
└─────────────────────────────────────┘
```

### No Annotations
```
┌─────────────────────────────────────┐
│                                     │
│            ✏️                       │
│                                     │
│    No annotations yet               │
│                                     │
│  Start highlighting and adding      │
│  notes to remember key points       │
│                                     │
└─────────────────────────────────────┘
```

### No Search Results
```
┌─────────────────────────────────────┐
│                                     │
│            🔍                       │
│                                     │
│  No results for "query text"        │
│                                     │
│  Try different keywords or          │
│  check your spelling                │
│                                     │
└─────────────────────────────────────┘
```

---

## Loading States

### Skeleton Book Card
```
┌─────────┐
│░░░░░░░░░│
│░░░░░░░░░│ (shimmer animation)
│░░░░░░░░░│
│░░░░░░░░░│
│████░░░░ │
└─────────┘
```

### Inline Loader
```
  ⟳ Loading...
```

### Full Page Loader
```
┌─────────────────────────────────────┐
│                                     │
│                                     │
│              ⟳                      │
│          Loading...                 │
│                                     │
│                                     │
└─────────────────────────────────────┘
```

---

## Error States

### Inline Error
```
┌───────────────────────────────┐
│ ⚠️ Failed to upload file       │
│ Network connection lost        │
│               [Retry] [Dismiss]│
└───────────────────────────────┘
```

### Toast Notification
```
┌─────────────────────────────────────┐
│  ✅ Book uploaded successfully!     │
└─────────────────────────────────────┘
(Auto-dismiss after 3 seconds)
```

---

**Next Steps:**
1. Review designs with stakeholders
2. Create high-fidelity mockups in Figma
3. Build component library in Storybook
4. Implement screens in React Native

**Design Tools:**
- Figma (design & prototyping)
- Storybook (component documentation)
- Zeplin (developer handoff)
