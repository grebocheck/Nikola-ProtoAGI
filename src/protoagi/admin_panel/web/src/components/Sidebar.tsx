import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { api, type HealthSummary } from "../lib/api";

const NAV_ITEMS = [
  { to: "/", label: "Огляд", icon: "◎" },
  { to: "/memory", label: "Память", icon: "▤" },
  { to: "/goals", label: "Цілі", icon: "◇" },
  { to: "/conflicts", label: "Суперечності", icon: "⚠" },
  { to: "/chats", label: "Чати", icon: "✎" },
] as const;

const PERSONAS = ["solomiya", "mykola"] as const;

export function Sidebar() {
  const [persona, setPersona] = usePersistedPersona();
  const [health, setHealth] = useState<HealthSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    api
      .health(persona)
      .then((value) => {
        if (!cancelled) setHealth(value);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [persona]);

  return (
    <aside className="w-64 shrink-0 border-r border-zinc-800 bg-zinc-900/60 flex flex-col">
      <div className="px-5 py-4 border-b border-zinc-800">
        <div className="text-sm uppercase tracking-wider text-zinc-500">ProtoAGI</div>
        <div className="text-lg font-medium">Admin</div>
      </div>
      <div className="px-5 py-3 border-b border-zinc-800">
        <label className="text-xs text-zinc-500 block mb-1">Persona</label>
        <select
          value={persona}
          onChange={(e) => setPersona(e.target.value)}
          className="w-full bg-zinc-800 text-zinc-100 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-zinc-600"
        >
          {PERSONAS.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>
      <nav className="flex-1 px-3 py-3 space-y-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              [
                "flex items-center gap-3 px-3 py-2 rounded text-sm transition-colors",
                isActive
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200",
              ].join(" ")
            }
          >
            <span className="text-zinc-500">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="px-5 py-4 border-t border-zinc-800 text-xs text-zinc-500 space-y-1">
        {error ? (
          <div className="text-red-400">{error}</div>
        ) : health ? (
          <>
            <StatLine label="active memories" value={health.memories_active} />
            <StatLine label="open goals" value={health.open_goals} />
            <StatLine label="unresolved" value={health.unresolved_conflicts} />
            <StatLine label="user models" value={health.user_states_tracked} />
          </>
        ) : (
          <div>Loading…</div>
        )}
      </div>
    </aside>
  );
}

function StatLine({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex justify-between">
      <span>{label}</span>
      <span className="text-zinc-300 tabular-nums">{value}</span>
    </div>
  );
}

const PERSONA_STORAGE_KEY = "protoagi.admin.persona";

function usePersistedPersona(): [string, (value: string) => void] {
  const [value, setValue] = useState(() => {
    try {
      return localStorage.getItem(PERSONA_STORAGE_KEY) ?? "solomiya";
    } catch {
      return "solomiya";
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(PERSONA_STORAGE_KEY, value);
    } catch {
      /* ignore */
    }
  }, [value]);
  return [value, setValue];
}

/**
 * Read the active persona from localStorage. Pages call this so the
 * sidebar's <select> reflects the same value the data fetches use.
 * Returns ``"solomiya"`` when unset (matches sidebar default).
 */
export function readPersona(): string {
  try {
    return localStorage.getItem(PERSONA_STORAGE_KEY) ?? "solomiya";
  } catch {
    return "solomiya";
  }
}

export function usePersona(): string {
  const [value, setValue] = useState(readPersona);
  useEffect(() => {
    const onChange = () => setValue(readPersona());
    window.addEventListener("storage", onChange);
    // The sidebar updates localStorage in the same tab, so we also
    // listen for a custom event that we dispatch ourselves.
    const interval = setInterval(onChange, 1000);
    return () => {
      window.removeEventListener("storage", onChange);
      clearInterval(interval);
    };
  }, []);
  return value;
}
