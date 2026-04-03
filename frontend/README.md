# Voice Clone Studio - Frontend

A sleek, production-ready UI for the Voice Clone Studio desktop application. Built with **Next.js 16**, **React 19**, **Tailwind CSS**, and **Framer Motion** for a developer-inspired dark aesthetic.

## 🎨 Design System

- **Theme**: Deep dark mode (`#09090b` background)
- **Typography**: Inter (UI), JetBrains Mono (logs/code)
- **Accent Colors**: Amber (primary), Emerald (success), Rose (error), Violet (info)
- **Components**: Glassmorphism cards with 1px subtle borders

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/                          # Next.js App Router
│   │   ├── layout.tsx                # Root layout
│   │   ├── page.tsx                  # Dashboard
│   │   ├── session-builder/          # New training form (wizard)
│   │   ├── my-models/                # Models grid
│   │   ├── inference/                # Voice conversion
│   │   ├── training/[id]/            # Active training monitor
│   │   ├── curation/[id]/            # Segment review
│   │   └── settings/                 # Settings panel
│   ├── components/
│   │   ├── AppShell.tsx              # Main layout wrapper
│   │   ├── Sidebar.tsx               # Navigation sidebar
│   │   ├── screens/                  # Full-page views
│   │   │   ├── Dashboard.tsx
│   │   │   ├── SessionBuilder.tsx
│   │   │   ├── MyModels.tsx
│   │   │   ├── Inference.tsx
│   │   │   ├── ActiveTraining.tsx
│   │   │   ├── Curation.tsx
│   │   │   └── Settings.tsx
│   │   └── ui/                       # Reusable components
│   │       ├── StatusBadge.tsx
│   │       └── StatCard.tsx
│   ├── lib/
│   │   └── mockData.ts               # Mock data for all screens
│   └── styles/
│       └── globals.css               # Global Tailwind styles
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── next.config.js
└── postcss.config.js
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- pnpm (recommended) or npm

### Installation

```bash
cd frontend
pnpm install
```

### Development

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
pnpm build
pnpm start
```

## 📱 Screens Overview

### 1. **Dashboard** (`/`)
- Quick stats: GPU VRAM, CUDA status, active sessions
- Active training sessions with progress bars
- Recent models grid
- Quick access to new training

### 2. **Session Builder** (`/session-builder`)
- Multi-step wizard: Files → Config → Review
- File upload (audio/video)
- Model naming and advanced mode toggle
- Review and start training

### 3. **Active Training** (`/training/[id]`)
- Real-time epoch progress
- Loss graph (using Recharts)
- Terminal log streaming
- Pause/Resume/Stop controls
- Checkpoint selector

### 4. **My Models** (`/my-models`)
- Grid of all trained models
- Status badges (done, paused, training)
- Local vs cloud indicators
- Download weights for synced models
- Quick convert action

### 5. **Segment Curation** (`/curation/[id]`)
- Audio segment list with waveforms
- Approve/reject individual segments
- Bulk approve/reject
- Quality guidelines
- Proceeds to hyperparameter config

### 6. **Inference** (`/inference`)
- Model selector dropdown
- Audio file upload
- Pitch shift slider (-12 to +12 semitones)
- Advanced options (index rate, protect breathiness)
- Conversion output player

### 7. **Settings** (`/settings`)
- System information display
- Supabase cloud sync toggle
- Training preferences (auto-save, notifications)
- Storage & disk usage visualization
- About & links

## 🎯 Key Features

### Mock Data
All screens render with realistic mock data:
- 4 sample voice models
- 3 active training sessions
- 5 audio segments for curation
- System stats and logs

### Responsive Design
- Mobile-first approach
- Flexbox-based layouts (Tailwind)
- Grid for model cards
- Collapsible sections on mobile

### Accessibility
- Semantic HTML
- ARIA roles where needed
- Keyboard-navigable forms
- Screen reader friendly

### Dark Mode
- Fixed dark theme (no toggle needed yet)
- Consistent contrast ratios
- Glassmorphism effect with subtle transparency

## 🔌 API Integration Points

Currently using mock data. Ready to integrate with backend:

### Endpoints to connect
- `POST /api/sessions` - Create training session
- `GET /api/sessions` - List sessions
- `GET /api/training/{id}` - Get training progress
- `POST /api/inference` - Run voice conversion
- `GET /api/models` - Fetch trained models

## 📚 Technologies

| Tech | Version | Purpose |
|------|---------|---------|
| Next.js | 16 | Framework |
| React | 19 | UI library |
| Tailwind CSS | 3.4 | Styling |
| Recharts | 2.10 | Loss graphs |
| Framer Motion | 10.16 | Animations |
| Wavesurfer.js | 7.7 | Audio visualization |
| SWR | 2.2 | Data fetching (ready) |
| Lucide React | 0.263 | Icons |

## 🎨 Customization

### Colors
Edit `tailwind.config.js` to modify accent colors:
```js
accent: {
  amber: "#fbbf24",
  emerald: "#10b981",
  rose: "#f43f5e",
  violet: "#a78bfa",
}
```

### Typography
Fonts are imported in `src/styles/globals.css` via Google Fonts.

### Card Styling
All cards use the `.card-base` Tailwind utility class for consistency.

## 🧪 Mock Data Structure

See `src/lib/mockData.ts` for:
- `TrainingSession` interface
- `VoiceModel` interface
- `AudioSegment` interface
- Sample data arrays: `mockSessions`, `mockModels`, `mockSegments`

## 📝 Notes

- All pages render independently with mock data
- No external API calls needed yet
- Forms are functional but don't persist data
- Live training simulation in ActiveTraining (epoch auto-increments)
- Loss graph uses synthetic data

## 🚢 Deployment

Ready to deploy to Vercel, Netlify, or any Node.js host:

```bash
pnpm build
pnpm start
```

## 📄 License

Part of Voice Clone Studio project.
