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
import { api, type StickerDescription } from "../lib/api";

type FilterDescribed = "all" | "yes" | "no";

const DESCRIBED_LABELS: Record<FilterDescribed, string> = {
  all: "Всі стікери",
  yes: "Тільки з описом",
  no: "Тільки без опису",
};

export function StickersPage() {
  const [items, setItems] = useState<StickerDescription[]>([]);
  const [packs, setPacks] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pack, setPack] = useState<string>("");
  const [described, setDescribed] = useState<FilterDescribed>("all");
  const [search, setSearch] = useState("");
  const [busyIds, setBusyIds] = useState<Set<string>>(new Set());
  // Multi-select. Operators tick the cards they want to redescribe,
  // then hit one of the bulk-action buttons in the floating bar that
  // appears at the top of the grid.
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const toggleSelection = useCallback((stickerId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(stickerId)) next.delete(stickerId);
      else next.add(stickerId);
      return next;
    });
  }, []);
  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  const fetchPacks = useCallback(() => {
    api
      .stickerPacks()
      .then((value) => setPacks(value))
      .catch(() => {
        /* dropdown stays empty; not fatal */
      });
  }, []);

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
  useEffect(() => fetchPacks(), [fetchPacks]);

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
    const failedN = items.filter(
      (i) => i.attempt_count >= 3 && !i.description
    ).length;
    return { total, describedN, failedN };
  }, [items]);

  const handleReset = (opts: { clear: boolean }) => {
    const message = opts.clear
      ? "Стерти ВСІ описи у вибраному наборі і поставити в чергу на повторний опис? Це незворотньо."
      : "Скинути лічильник спроб для стікерів без опису?";
    if (!confirm(message)) return;
    api
      .resetStickers({
        pack: pack || undefined,
        only_failed: !opts.clear,
        clear_descriptions: opts.clear,
      })
      .then((res) => {
        setError(null);
        fetchItems();
        alert(
          `Готово. ${res.reset} рядків ${opts.clear ? "очищено й поставлено в чергу" : "скинуто"}.`
        );
      })
      .catch((err) => setError(String(err)));
  };

  const handleRedescribeOne = (item: StickerDescription) => {
    setBusyIds((prev) => new Set(prev).add(item.sticker_id));
    api
      .redescribeSticker(item.sticker_id)
      .then(() => fetchItems())
      .catch((err) => setError(String(err)))
      .finally(() => {
        setBusyIds((prev) => {
          const next = new Set(prev);
          next.delete(item.sticker_id);
          return next;
        });
      });
  };

  const handleRedescribeSelected = () => {
    if (selectedIds.size === 0) return;
    if (
      !confirm(
        `Стерти описи у ${selectedIds.size} вибраних стікерів і поставити в чергу на повторний опис?`
      )
    )
      return;
    const ids = Array.from(selectedIds);
    api
      .resetStickers({
        sticker_ids: ids,
        only_failed: false,
        clear_descriptions: true,
      })
      .then((res) => {
        setError(null);
        clearSelection();
        fetchItems();
        alert(`Готово. ${res.reset} рядків очищено й поставлено в чергу.`);
      })
      .catch((err) => setError(String(err)));
  };

  const handleSelectAllVisible = () => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const item of filtered) next.add(item.sticker_id);
      return next;
    });
  };

  return (
    <PageBody>
      <PageHeader
        title="Стікери"
        subtitle={`${counts.describedN}/${counts.total} описано · ${counts.failedN} стійко падали`}
        actions={
          <>
            <button
              type="button"
              onClick={() => handleReset({ clear: false })}
              className="text-xs px-3 py-1.5 rounded border border-amber-700/60 text-amber-300 hover:bg-amber-950/30"
              title="Зняти лічильник спроб тільки з тих стікерів, що не описалися"
            >
              Повторити невдалі
            </button>
            <button
              type="button"
              onClick={() => handleReset({ clear: true })}
              className="text-xs px-3 py-1.5 rounded border border-red-700/60 text-red-300 hover:bg-red-950/30"
              title="Стерти всі описи у вибраному наборі і поставити в чергу"
            >
              Передописати все
            </button>
            <button
              type="button"
              onClick={() => {
                fetchItems();
                fetchPacks();
              }}
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
      {pack ? (
        <div className="text-xs text-zinc-500 mb-3">
          Кнопки "Повторити невдалі" і "Передописати все" діють тільки на пакет:
          <span className="text-zinc-300 ml-1 font-mono">{pack}</span>
        </div>
      ) : null}
      {selectedIds.size > 0 ? (
        <div className="sticky top-0 z-10 -mx-2 px-2 py-2 mb-3 bg-zinc-900/95 border border-sky-700/40 rounded flex flex-wrap items-center gap-2 backdrop-blur">
          <span className="text-sm text-sky-300">
            Вибрано: <span className="font-semibold">{selectedIds.size}</span>
          </span>
          <button
            type="button"
            onClick={handleRedescribeSelected}
            className="text-xs px-3 py-1 rounded border border-sky-700/60 text-sky-200 hover:bg-sky-950/40"
          >
            Передописати вибрані
          </button>
          <button
            type="button"
            onClick={handleSelectAllVisible}
            className="text-xs px-3 py-1 rounded border border-zinc-700 text-zinc-300 hover:bg-zinc-800"
          >
            Виділити всі видимі ({filtered.length})
          </button>
          <button
            type="button"
            onClick={clearSelection}
            className="text-xs px-3 py-1 rounded border border-zinc-700 text-zinc-400 hover:bg-zinc-800"
          >
            Скасувати вибір
          </button>
        </div>
      ) : (
        <div className="text-xs text-zinc-500 mb-3">
          Підказка: натисни чекбокс у лівому верхньому куті картки щоб вибрати кілька стікерів для bulk-операції.
        </div>
      )}
      {error ? <ErrorBox message={error} /> : null}
      {loading && filtered.length === 0 ? <Loading /> : null}
      {!loading && filtered.length === 0 && !error ? (
        <EmptyState message="Тут ще пусто — стікер описувач не зробив пас, або фільтр відсікнув усе." />
      ) : null}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 mt-4">
        {filtered.map((item) => (
          <Card
            key={item.sticker_id}
            item={item}
            busy={busyIds.has(item.sticker_id)}
            selected={selectedIds.has(item.sticker_id)}
            onRedescribe={() => handleRedescribeOne(item)}
            onToggleSelect={() => toggleSelection(item.sticker_id)}
          />
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
        placeholder="Пошук по опису / emoji / пакету / id"
        className="flex-1 min-w-[200px] bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-zinc-500"
      />
      <select
        value={pack}
        onChange={(e) => onPack(e.target.value)}
        className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-sm min-w-[200px]"
        title="Pack filter — впливає й на список, і на bulk-actions"
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
        {(["all", "yes", "no"] as FilterDescribed[]).map((value) => (
          <option key={value} value={value}>
            {DESCRIBED_LABELS[value]}
          </option>
        ))}
      </select>
    </div>
  );
}

function Card({
  item,
  busy,
  selected,
  onRedescribe,
  onToggleSelect,
}: {
  item: StickerDescription;
  busy: boolean;
  selected: boolean;
  onRedescribe: () => void;
  onToggleSelect: () => void;
}) {
  const [imageBroken, setImageBroken] = useState(false);
  const ageDays = useMemo(() => {
    try {
      const created = new Date(item.created_at);
      return Math.floor((Date.now() - created.getTime()) / 86400000);
    } catch {
      return null;
    }
  }, [item.created_at]);

  const borderClass = selected
    ? "border-sky-500/70 ring-1 ring-sky-500/40"
    : "border-zinc-800";

  return (
    <div
      className={`border ${borderClass} rounded-lg bg-zinc-900/40 overflow-hidden flex flex-col transition-colors`}
    >
      <div className="aspect-square bg-zinc-950 flex items-center justify-center p-2 relative group">
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
        {/* Always-visible checkbox so multi-select discoverable. The
            redescribe button still appears only on hover. */}
        <label
          className="absolute top-1 left-1 flex items-center justify-center w-6 h-6 rounded bg-zinc-950/70 backdrop-blur cursor-pointer hover:bg-zinc-900"
          onClick={(e) => e.stopPropagation()}
        >
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="w-4 h-4 accent-sky-500 cursor-pointer"
          />
        </label>
        <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <IconButton
            onClick={onRedescribe}
            title="Стерти опис і описати знову"
            variant="primary"
          >
            {busy ? "..." : "↻"}
          </IconButton>
        </div>
      </div>
      <div className="px-3 py-2 flex-1 flex flex-col gap-1 text-sm">
        <div className="flex items-center gap-2 flex-wrap text-xs">
          <Pill>{item.set_name}</Pill>
          {item.emoji ? <span>{item.emoji}</span> : null}
          {!item.description ? (
            <Pill tone="warn">{`спроб: ${item.attempt_count}`}</Pill>
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
