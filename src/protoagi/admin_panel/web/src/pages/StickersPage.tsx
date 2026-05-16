import { useCallback, useEffect, useMemo, useState } from "react";
import {
  EmptyState,
  ErrorBox,
  Loading,
  PageBody,
  PageHeader,
  Pill,
} from "../components/Page";
import { api, type StickerDescription } from "../lib/api";

type FilterDescribed = "all" | "yes" | "no";

export function StickersPage() {
  const [items, setItems] = useState<StickerDescription[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pack, setPack] = useState<string>("");
  const [described, setDescribed] = useState<FilterDescribed>("all");
  const [search, setSearch] = useState("");

  const fetchItems = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .stickers({ pack: pack || undefined, described, limit: 1000 })
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
  }, [pack, described]);

  useEffect(() => fetchItems(), [fetchItems]);

  // Derive the pack dropdown options from whatever showed up in the
  // raw fetch so we don't hardcode the list (still adds packs the bot
  // discovered automatically).
  const packs = useMemo(() => {
    const set = new Set<string>();
    for (const item of items) set.add(item.set_name);
    return Array.from(set).sort();
  }, [items]);

  const searchLower = search.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!searchLower) return items;
    return items.filter((item) => {
      return (
        item.description.toLowerCase().includes(searchLower) ||
        item.emoji.includes(search) ||
        item.set_name.toLowerCase().includes(searchLower) ||
        item.sticker_id.toLowerCase().includes(searchLower)
      );
    });
  }, [items, search, searchLower]);

  const counts = useMemo(() => {
    const total = items.length;
    const describedN = items.filter((i) => i.description).length;
    const failedN = items.filter((i) => i.attempt_count >= 3 && !i.description)
      .length;
    return { total, describedN, failedN };
  }, [items]);

  return (
    <PageBody>
      <PageHeader
        title="Стікери"
        subtitle={`${counts.describedN}/${counts.total} описано · ${counts.failedN} стійко падали`}
        actions={
          <>
            <button
              type="button"
              onClick={() => {
                if (!confirm(
                  "Скинути лічильник спроб для всіх не-описаних стікерів? " +
                  "Описувач спробує їх знову при наступному циклі."
                )) return;
                api
                  .resetStickers({ pack: pack || undefined, only_failed: true })
                  .then((res) => {
                    setError(null);
                    fetchItems();
                    alert(`Скинуто ${res.reset} рядків. Перезапусти бот або зачекай наступного циклу описувача.`);
                  })
                  .catch((err) => setError(String(err)));
              }}
              className="text-xs px-3 py-1.5 rounded border border-amber-700/60 text-amber-300 hover:bg-amber-950/30"
            >
              Скинути спроби
            </button>
            <button
              type="button"
              onClick={fetchItems}
              className="text-xs px-3 py-1.5 rounded border border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            >
              Оновити
            </button>
          </>
        }
      />
      <Filters
        pack={pack}
        described={described}
        search={search}
        packs={packs}
        onPack={setPack}
        onDescribed={setDescribed}
        onSearch={setSearch}
      />
      {error ? <ErrorBox message={error} /> : null}
      {loading && filtered.length === 0 ? <Loading /> : null}
      {!loading && filtered.length === 0 && !error ? (
        <EmptyState message="Тут ще пусто — стікер описувач не зробив пас, або фільтр відсікнув усе." />
      ) : null}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 mt-4">
        {filtered.map((item) => (
          <Card key={item.sticker_id} item={item} />
        ))}
      </div>
    </PageBody>
  );
}

function Filters({
  pack,
  described,
  search,
  packs,
  onPack,
  onDescribed,
  onSearch,
}: {
  pack: string;
  described: FilterDescribed;
  search: string;
  packs: string[];
  onPack: (v: string) => void;
  onDescribed: (v: FilterDescribed) => void;
  onSearch: (v: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-3 mb-4">
      <input
        value={search}
        onChange={(e) => onSearch(e.target.value)}
        placeholder="Пошук по опису, emoji, пакету..."
        className="flex-1 min-w-[200px] bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-zinc-500"
      />
      <select
        value={pack}
        onChange={(e) => onPack(e.target.value)}
        className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-sm"
      >
        <option value="">всі пакети</option>
        {packs.map((value) => (
          <option key={value} value={value}>
            {value}
          </option>
        ))}
      </select>
      <select
        value={described}
        onChange={(e) => onDescribed(e.target.value as FilterDescribed)}
        className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-sm"
      >
        <option value="all">всі</option>
        <option value="yes">тільки описані</option>
        <option value="no">тільки в черзі</option>
      </select>
    </div>
  );
}

function Card({ item }: { item: StickerDescription }) {
  const [imageBroken, setImageBroken] = useState(false);
  const ageDays = useMemo(() => {
    try {
      const created = new Date(item.created_at);
      return Math.floor((Date.now() - created.getTime()) / 86400000);
    } catch {
      return null;
    }
  }, [item.created_at]);

  return (
    <div className="border border-zinc-800 rounded-lg bg-zinc-900/40 overflow-hidden flex flex-col">
      <div className="aspect-square bg-zinc-950 flex items-center justify-center p-2">
        {imageBroken ? (
          <div className="text-xs text-zinc-500 text-center">
            thumbnail не закешований
          </div>
        ) : (
          <img
            src={api.stickerThumbnailUrl(item.sticker_id)}
            alt={item.description || item.sticker_id}
            className="max-w-full max-h-full object-contain"
            onError={() => setImageBroken(true)}
            loading="lazy"
          />
        )}
      </div>
      <div className="px-3 py-2 flex-1 flex flex-col gap-1 text-sm">
        <div className="flex items-center gap-2 flex-wrap text-xs">
          <Pill>{item.set_name}</Pill>
          {item.emoji ? <span>{item.emoji}</span> : null}
          {!item.description ? (
            <Pill tone="warn">{`x${item.attempt_count}`}</Pill>
          ) : null}
        </div>
        <div className="text-zinc-200 leading-snug min-h-[3em]">
          {item.description || (
            <span className="text-zinc-500 italic">
              {item.failure_reason || "очікує опису"}
            </span>
          )}
        </div>
        <div className="text-xs text-zinc-500 flex flex-wrap gap-x-3 gap-y-0.5">
          {item.has_embedding ? <span>vec</span> : null}
          {ageDays !== null ? <span>{ageDays}d</span> : null}
        </div>
      </div>
    </div>
  );
}
