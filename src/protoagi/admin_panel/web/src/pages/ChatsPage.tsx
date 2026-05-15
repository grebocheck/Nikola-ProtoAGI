import { useCallback, useEffect, useState } from "react";
import {
  EmptyState,
  ErrorBox,
  Loading,
  PageBody,
  PageHeader,
  Pill,
} from "../components/Page";
import {
  api,
  type ReasoningEntry,
  type TelegramChat,
} from "../lib/api";

export function ChatsPage() {
  const [chats, setChats] = useState<TelegramChat[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [entries, setEntries] = useState<ReasoningEntry[]>([]);
  const [entriesLoading, setEntriesLoading] = useState(false);
  const [entriesError, setEntriesError] = useState<string | null>(null);

  const fetchChats = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .chats()
      .then((value) => {
        if (!cancelled) setChats(value);
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
  }, []);

  useEffect(() => fetchChats(), [fetchChats]);

  useEffect(() => {
    if (!selectedChatId) {
      setEntries([]);
      return;
    }
    let cancelled = false;
    setEntriesLoading(true);
    setEntriesError(null);
    api
      .reasoningEntries(selectedChatId, 30)
      .then((value) => {
        if (!cancelled) setEntries(value.entries);
      })
      .catch((err) => {
        if (!cancelled) setEntriesError(String(err));
      })
      .finally(() => {
        if (!cancelled) setEntriesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedChatId]);

  return (
    <PageBody>
      <PageHeader
        title="Чати"
        subtitle="Telegram чати + reasoning log. Натисни на чат щоб побачити останні decision-сліди."
      />
      {error ? <ErrorBox message={error} /> : null}
      {loading && chats.length === 0 ? <Loading /> : null}
      <div className="grid md:grid-cols-[320px_1fr] gap-4 mt-2">
        <ul className="space-y-1">
          {chats.length === 0 && !loading ? (
            <EmptyState message="Чатів ще немає." />
          ) : null}
          {chats.map((chat) => {
            const active = chat.chat_id === selectedChatId;
            return (
              <li key={chat.chat_id}>
                <button
                  type="button"
                  onClick={() => setSelectedChatId(chat.chat_id)}
                  className={[
                    "w-full text-left border rounded-lg px-3 py-2 transition-colors",
                    active
                      ? "border-zinc-600 bg-zinc-900"
                      : "border-zinc-800 bg-zinc-900/40 hover:bg-zinc-900",
                  ].join(" ")}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-medium text-zinc-100 text-sm truncate">
                      {chat.display_name || chat.chat_id}
                    </div>
                    <Pill tone={chat.proactive_enabled ? "success" : "default"}>
                      {chat.chat_type}
                    </Pill>
                  </div>
                  <div className="text-xs text-zinc-500 mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
                    <span>reply: {chat.reply_mode}</span>
                    {chat.last_seen_at ? (
                      <span>last: {chat.last_seen_at}</span>
                    ) : null}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
        <div className="space-y-3">
          {selectedChatId ? (
            <>
              <div className="text-sm text-zinc-400">
                Reasoning log · chat <span className="font-mono">{selectedChatId}</span>
              </div>
              {entriesError ? <ErrorBox message={entriesError} /> : null}
              {entriesLoading && entries.length === 0 ? <Loading /> : null}
              {!entriesLoading && entries.length === 0 && !entriesError ? (
                <EmptyState message="У цьому чаті немає reasoning записів." />
              ) : null}
              <ul className="space-y-3">
                {entries.map((entry, idx) => (
                  <li
                    key={`${entry.created_at}-${idx}`}
                    className="border border-zinc-800 rounded-lg p-3 bg-zinc-900/40 text-sm space-y-2"
                  >
                    <div className="flex flex-wrap gap-2 items-center text-xs text-zinc-500">
                      <Pill tone="info">{entry.decision_kind}</Pill>
                      <span>{entry.created_at}</span>
                      {entry.message_id ? (
                        <span>msg #{entry.message_id}</span>
                      ) : null}
                    </div>
                    {entry.incoming_text ? (
                      <div>
                        <div className="text-xs text-zinc-500 mb-1">user:</div>
                        <div className="whitespace-pre-wrap text-zinc-200">
                          {entry.incoming_text}
                        </div>
                      </div>
                    ) : null}
                    {entry.reply_excerpt ? (
                      <div>
                        <div className="text-xs text-zinc-500 mb-1">reply:</div>
                        <div className="whitespace-pre-wrap text-zinc-200">
                          {entry.reply_excerpt}
                        </div>
                      </div>
                    ) : null}
                    {entry.reasoning_text ? (
                      <details className="text-xs text-zinc-400">
                        <summary className="cursor-pointer text-zinc-500">
                          chain of thought
                        </summary>
                        <pre className="mt-2 whitespace-pre-wrap font-mono text-xs leading-relaxed">
                          {entry.reasoning_text}
                        </pre>
                      </details>
                    ) : null}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <EmptyState message="Обери чат зліва щоб побачити reasoning." />
          )}
        </div>
      </div>
    </PageBody>
  );
}
