# Novelist UI

Frontend for the Novelist dashboard.

## Requirements

- Node.js 20+
- Backend API running at `http://127.0.0.1:8000` (default)

## Development

```bash
npm install
npm run dev
```

The app runs on `http://localhost:5173`.

## Build

```bash
npm run build
```

## API base URL

By default, the UI points to:

- `http://127.0.0.1:8000` during local dev
- same-origin in production

Override with:

```bash
VITE_API_BASE_URL=https://your-api.example.com
```
