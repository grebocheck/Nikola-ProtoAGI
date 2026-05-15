import { useCallback, useEffect, useMemo, useState } from "react";
import {
  EmptyState,
  ErrorBox,
  IconButton,
  Loading,
  PageBody,
  PageHeader,
  Pill,
} from "../components/Page";
import { usePersona } from "../components/Sidebar";
import { api, type MemoryItem } from "../lib/api";

const KIND_OPTIONS = ["", "fact", "semantic", "episodic", "procedural", "persona_self"];
const SCOPE_OPTIONS = ["", "global", "user", "chat", "persona"];

export function MemoryPage() {
  const persona = usePersona();
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState("");
  const [scope, setScope] = useState("");
  const [onlyPinned, setOnlyPinned] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");

  const fetchItems = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .memories({
        persona,
        search: search || undefined,
        kind: kind || undefined,
        scope: scope || undefined,
        pinned: onlyPinned ? true : undefined,
        limit: 200,
      })
      .then((value) => {
        if (!cancelled) setItems(value);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [persona, search, kind, scope, onlyPinned]);

  useEffect(() => {
    return fetchItems();
  }, [fetchItems]);

  const togglePin = (item: MemoryItem) => {
    api
      .pinMemory(item.id)
      .then(() => fetchItems())
      .catch((err) => setError(String(err)));
  };

  const deleteOne = (item: MemoryItem) => {
    if (!confirm(`Видалити пам'ятку #${item.id}? Це незворотньо.`)) return;
    api
      .deleteMemory(item.id)
      .then(() => fetchItems())
      .catch((err) => setError(String(err)));
  };

  const startEdit = (item: MemoryItem) => {
    setEditingId(item.id);
    setEditDraft(item.text);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditDraft("");
  };

  const commitEdit = () => {
    if (editingId === null) return;
    api
      .editMemory(editingId, { text: editDraft })
      .then(() => {
        cancelEdit();
        fetchItems();
      })
      .catch((err) => setError(String(err)));
  };

  return (
    <PageBody>
      <PageHeader
        title="Память"
        subtitle={`${items.length} рядків`}
        actions={
          <button
            type="button"
            onClick={fetchItems}
            className="text-xs px-3 py-1.5 rounded border border-zinc-700 text-zinc-300 hover:bg-zinc-800"
          >
            Оновити
          </button>
        }
      />
      <Filters
        search={search}
        kind={kind}
        scope={scope}
        onlyPinned={onlyPinned}
        onSearch={setSearch}
        onKind={setKind}
        onScope={setScope}
        onPinned={setOnlyPinned}
      />
      {error ? <ErrorBox message={error} /> : null}
      {loading && items.length === 0 ? <Loading /> : null}
      {!loading && items.length === 0 && !error ? (
        <EmptyState message="Нічого не знайдено за цими фільтрами." />
      ) : null}
      <ul className="space-y-2 mt-4">
        {items.map((item) => (
          <li
            key={item.id}
            className="border border-zinc-800 rounded-lg p-4 bg-zinc-900/40"
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <div className="flex flex-wrap gap-2 items-center">
                <span className="text-xs font-mono text-zinc-500">#{item.id}</span>
                <Pill>{item.kind}</Pill>
                <Pill tone="info">{item.scope}</Pill>
                {item.persona_key ? <Pill>{item.persona_key}</Pill> : null}
                {item.pinned ? <Pill tone="warn">pinned</Pill> : null}
                {item.expires_at ? <Pill tone="info">expires</Pill> : null}
                {item.superseded_by ? (
                  <Pill tone="warn">superseded #{item.superseded_by}</Pill>
                ) : null}
              </div>
              <div className="flex gap-2 shrink-0">
                <IconButton onClick={() => togglePin(item)} title="Пін/анпін">
                  {item.pinned ? "Unpin" : "Pin"}
                </IconButton>
                {editingId === item.id ? null : (
                  <IconButton onClick={() => startEdit(item)} title="Редагувати">
                    Edit
                  </IconButton>
                )}
                <IconButton
                  onClick={() => deleteOne(item)}
                  title="Видалити назавжди"
                  variant="danger"
                >
                  Delete
                </IconButton>
              </div>
            </div>
            {editingId === item.id ? (
              <div className="space-y-2">
                <textarea
                  value={editDraft}
                  onChange={(e) => setEditDraft(e.target.value)}
                  className="w-full text-sm bg-zinc-900 border border-zinc-700 rounded p-2 text-zinc-100 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                  rows={3}
                />
                <div className="flex gap-2">
                  <IconButton onClick={commitEdit} title="Save" variant="primary">
                    Save
                  </IconButton>
                  <IconButton onClick={cancelEdit} title="Cancel">
                    Cancel
                  </IconButton>
                </div>
              </div>
            ) : (
              <div className="text-sm text-zinc-200 whitespace-pre-wrap">
                {item.text}
              </div>
            )}
            <Footer item={item} />
          </li>
        ))}
      </ul>
    </PageBody>
  );
}

function Filters({
  search,
  kind,
  scope,
  onlyPinned,
  onSearch,
  onKind,
  onScope,
  onPinned,
}: {
  search: string;
  kind: string;
  scope: string;
  onlyPinned: boolean;
  onSearch: (value: string) => void;
  onKind: (value: string) => void;
  onScope: (value: string) => void;
  onPinned: (value: boolean) => void;
}) {
  return (
    <div className="flex flex-wrap gap-3 mb-4">
      <input
        value={search}
        onChange={(e) => onSearch(e.target.value)}
        placeholder="Пошук (FTS)…"
        className="flex-1 min-w-[200px] bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-zinc-500"
      />
      <select
        value={kind}
        onChange={(e) => onKind(e.target.value)}
        className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-sm"
      >
        {KIND_OPTIONS.map((value) => (
          <option key={value} value={value}>
            {value || "будь-який kind"}
          </option>
        ))}
      </select>
      <select
        value={scope}
        onChange={(e) => onScope(e.target.value)}
        className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-sm"
      >
        {SCOPE_OPTIONS.map((value) => (
          <option key={value} value={value}>
            {value || "будь-який scope"}
          </option>
        ))}
      </select>
      <label className="flex items-center gap-2 text-sm text-zinc-300">
        <input
          type="checkbox"
          checked={onlyPinned}
          onChange={(e) => onPinned(e.target.checked)}
        />
        тільки pinned
      </label>
    </div>
  );
}

function Footer({ item }: { item: MemoryItem }) {
  const tags = useMemo(() => item.tags.filter(Boolean), [item.tags]);
  return (
    <div className="mt-2 text-xs text-zinc-500 flex flex-wrap gap-x-4 gap-y-1">
      <span>importance: {item.importance.toFixed(2)}</span>
      <span>conf: {item.confidence.toFixed(2)}</span>
      <span>created: {item.created_at}</span>
      {item.origin_message_id ? (
        <span>origin: {item.origin_message_id}</span>
      ) : null}
      {tags.length ? <span>tags: {tags.join(", ")}</span> : null}
    </div>
  );
}
