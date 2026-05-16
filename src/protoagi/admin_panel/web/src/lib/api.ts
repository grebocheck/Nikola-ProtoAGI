// Typed wrappers around the Python admin server's JSON endpoints.
//
// Each function returns the parsed body, throws on non-2xx (so React
// Query / useEffect handlers can `catch` uniformly). Pages should only
// ever talk to the server through this module — that way URL shape
// changes stay in one place.

export interface HealthSummary {
  persona_key: string | null;
  memories_active: number;
  memories_superseded: number;
  memories_total: number;
  open_goals: number;
  user_states_tracked: number;
  unresolved_conflicts: number;
}

export interface MemoryItem {
  id: number;
  kind: string;
  text: string;
  scope: string;
  tags: string[];
  importance: number;
  confidence: number;
  user_id: string | null;
  chat_id: string | null;
  persona_key: string | null;
  media_id: string | null;
  created_at: string;
  updated_at: string | null;
  access_count: number;
  pinned: boolean;
  origin_message_id: string | null;
  expires_at: string | null;
  superseded_by: number | null;
  supersedes_id: number | null;
  source: string | null;
}

export interface Goal {
  id: number;
  persona_key: string;
  text: string;
  status: "open" | "completed" | "abandoned";
  priority: number;
  chat_id: string | null;
  user_id: string | null;
  origin_message_id: number | null;
  due_at: string | null;
  last_touched_at: string;
  created_at: string;
  updated_at: string | null;
  closed_at: string | null;
  metadata: Record<string, unknown>;
}

export interface ConflictSide {
  id: number;
  text: string;
  kind: string;
  scope: string;
  tags: string[];
  created_at: string;
  origin_message_id: string | null;
  importance: number;
  superseded_by: number | null;
}

export type ConflictStatus =
  | "unresolved"
  | "superseded"
  | "kept_both"
  | "dismissed";

export interface Conflict {
  id: number;
  memory_a_id: number;
  memory_b_id: number;
  similarity: number;
  persona_key: string | null;
  detected_at: string;
  resolution_status: ConflictStatus;
  resolution_winner_id: number | null;
  resolved_at: string | null;
  metadata: Record<string, unknown>;
  memory_a?: ConflictSide;
  memory_b?: ConflictSide;
}

export interface StickerDescription {
  sticker_id: string;
  set_name: string;
  emoji: string;
  description: string;
  embedding_model: string | null;
  has_embedding: boolean;
  failure_reason: string | null;
  attempt_count: number;
  last_used_at: string | null;
  created_at: string;
  updated_at: string | null;
}

export interface UserState {
  user_id: string;
  persona_key: string;
  mood: string;
  themes: string[];
  open_questions: string[];
  preferences: Record<string, unknown>;
  summary: string;
  confidence: number;
  last_updated_at: string;
  messages_at_last_update: number;
  metadata: Record<string, unknown>;
}

export interface TelegramChat {
  chat_id: string;
  display_name: string | null;
  chat_type: string;
  reply_mode: string;
  proactive_enabled: boolean;
  last_seen_at: string | null;
  last_user_message_at: string | null;
  last_bot_message_at: string | null;
}

export interface ReasoningOverviewItem {
  chat_id: string;
  display_name: string | null;
  chat_type: string | null;
  last_decision_kind: string | null;
  updated_at: string | null;
  entries: number;
}

export interface ReasoningEntry {
  decision_kind: string;
  incoming_text: string;
  reasoning_text: string;
  reply_excerpt: string;
  created_at: string;
  message_id: number | null;
}

async function http<T>(
  path: string,
  init?: RequestInit & { params?: Record<string, string | number | boolean | undefined> }
): Promise<T> {
  const params = init?.params;
  let url = path;
  if (params) {
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null || value === "") continue;
      search.set(key, String(value));
    }
    const q = search.toString();
    if (q) url += (url.includes("?") ? "&" : "?") + q;
  }
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { error?: string };
      if (body.error) detail = body.error;
    } catch {
      /* ignore */
    }
    throw new Error(`${response.status} ${detail}`);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: (persona?: string) =>
    http<HealthSummary>("/api/health", { params: { persona } }),

  memories: (params?: {
    limit?: number;
    kind?: string;
    scope?: string;
    persona?: string;
    search?: string;
    pinned?: boolean;
  }) =>
    http<MemoryItem[]>("/api/memories", {
      params: {
        limit: params?.limit ?? 100,
        kind: params?.kind,
        scope: params?.scope,
        persona: params?.persona,
        search: params?.search,
        pinned: params?.pinned,
      },
    }),

  pinMemory: (id: number, pinned?: boolean) =>
    http<{ id: number; pinned: boolean }>(`/api/memories/${id}/pin`, {
      method: "POST",
      body: JSON.stringify(pinned === undefined ? {} : { pinned }),
    }),

  deleteMemory: (id: number) =>
    http<{ deleted: number }>(`/api/memories/${id}/delete`, {
      method: "POST",
      body: "{}",
    }),

  editMemory: (
    id: number,
    body: { text?: string; importance?: number; tags?: string[] }
  ) =>
    http<MemoryItem>(`/api/memories/${id}/edit`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  goals: (params?: { status?: string; persona?: string; limit?: number }) =>
    http<Goal[]>("/api/goals", { params }),

  updateGoal: (
    id: number,
    body: { status?: string; text?: string; priority?: number; due_at?: string | null }
  ) =>
    http<Goal>(`/api/goals/${id}/update`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  conflicts: (params?: { status?: string; persona?: string; limit?: number }) =>
    http<Conflict[]>("/api/conflicts", { params }),

  resolveConflict: (
    id: number,
    body: { status: ConflictStatus; winner_id?: number }
  ) =>
    http<{
      id: number;
      status: ConflictStatus;
      winner_id: number | null;
      resolved_at: string | null;
    }>(`/api/conflicts/${id}/resolve`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  userStates: (persona?: string) =>
    http<UserState[]>("/api/user_state", { params: { persona } }),

  stickers: (params?: {
    pack?: string;
    described?: "all" | "yes" | "no";
    limit?: number;
  }) =>
    http<StickerDescription[]>("/api/stickers", {
      params: {
        pack: params?.pack,
        described: params?.described,
        limit: params?.limit ?? 1000,
      },
    }),

  stickerThumbnailUrl: (stickerId: string) =>
    `/api/sticker_thumbnail/${encodeURIComponent(stickerId)}`,

  resetStickers: (body?: {
    pack?: string;
    sticker_ids?: string[];
    only_failed?: boolean;
    clear_descriptions?: boolean;
  }) =>
    http<{
      reset: number;
      pack: string | null;
      only_failed: boolean;
      clear_descriptions: boolean;
      sticker_ids_count: number;
    }>("/api/stickers/reset", {
      method: "POST",
      body: JSON.stringify(body ?? { only_failed: true }),
    }),

  redescribeSticker: (stickerId: string) =>
    http<{ sticker_id: string; queued: boolean }>(
      `/api/stickers/${encodeURIComponent(stickerId)}/redescribe`,
      { method: "POST", body: "{}" }
    ),

  stickerPacks: () => http<string[]>("/api/stickers/packs"),

  chats: () => http<TelegramChat[]>("/api/chats"),

  reasoningOverview: () =>
    http<ReasoningOverviewItem[]>("/api/reasoning"),

  reasoningEntries: (chatId: string, limit = 20) =>
    http<ReasoningEntry[]>(
      `/api/reasoning/${encodeURIComponent(chatId)}`,
      { params: { limit } }
    ),
};
