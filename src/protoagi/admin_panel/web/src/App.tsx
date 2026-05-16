import { Navigate, Route, Routes } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { OverviewPage } from "./pages/OverviewPage";
import { MemoryPage } from "./pages/MemoryPage";
import { GoalsPage } from "./pages/GoalsPage";
import { ConflictsPage } from "./pages/ConflictsPage";
import { ChatsPage } from "./pages/ChatsPage";
import { StickersPage } from "./pages/StickersPage";

export default function App() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <main className="flex-1 overflow-auto bg-zinc-950">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/goals" element={<GoalsPage />} />
          <Route path="/conflicts" element={<ConflictsPage />} />
          <Route path="/stickers" element={<StickersPage />} />
          <Route path="/chats" element={<ChatsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
