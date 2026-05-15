# ProtoAGI Admin (React)

Локальна адмінка для перегляду стану памʼяті, цілей, конфліктів і
Telegram-чатів. Vite + React 19 + TypeScript + Tailwind 4. Не SSR — це
звичайний SPA, який Python admin server (`protoagi admin`) роздає як
статичні файли.

## Розробка

```powershell
# 1) Поставити npm deps (один раз).
cd src/protoagi/admin_panel/web
npm install

# 2) Запустити Python admin на :8765 (роздає API + DB).
cd ../../../..
$env:PYTHONPATH = "src"
python -m protoagi admin

# 3) Запустити Vite dev server на :5173 (з HMR і proxy на :8765).
cd src/protoagi/admin_panel/web
npm run dev
```

Відкрий <http://127.0.0.1:5173>. API-запити проксяться на Python.

## Продакшн білд

```powershell
cd src/protoagi/admin_panel/web
npm run build
```

Це згенерує `dist/`, який Python server автоматично знаходить (через
`Path(__file__).parent / "web" / "dist"`) і роздає при наступному
запуску `protoagi admin`. Відкривати потім <http://127.0.0.1:8765>.

## Структура

```
web/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── src/
│   ├── main.tsx         # React mount + Router
│   ├── App.tsx          # layout shell
│   ├── index.css        # @import "tailwindcss"
│   ├── lib/
│   │   └── api.ts       # typed admin API client
│   ├── components/
│   │   ├── Sidebar.tsx  # бічна шторка з persona + nav + health
│   │   └── Page.tsx     # PageHeader / Card / Pill / IconButton / ...
│   └── pages/
│       ├── OverviewPage.tsx
│       ├── MemoryPage.tsx
│       ├── GoalsPage.tsx
│       ├── ConflictsPage.tsx
│       └── ChatsPage.tsx
```

Сторінки незалежні: кожна сама ходить у API і тримає локальний стейт.
Жодного глобального store — додаси якщо станет тісно.

## Routing

Маршрути:

| Шлях | Сторінка |
|------|----------|
| `/` | Огляд (health summary картки) |
| `/memory` | Память: фільтри, edit/pin/delete |
| `/goals` | Цілі: відкриті/виконані/покинуті |
| `/conflicts` | Суперечності: review unresolved, manual resolve |
| `/chats` | Telegram чати + reasoning log по chat_id |

`*` (будь-що інше) редіректить на `/`. SPA fallback на серверній
стороні (`/some/path` повертає `index.html`) — щоб refresh не зламав
client-side routing.
