import { useCallback, useEffect, useState } from "react";
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
import {
  api,
  type Conflict,
  type ConflictSide,
  type ConflictStatus,
} from "../lib/api";

const STATUSES: ConflictStatus[] = [
  "unresolved",
  "superseded",
  "kept_both",
  "dismissed",
];

export function ConflictsPage() {
  const persona = usePersona();
  const [status, setStatus] = useState<ConflictStatus>("unresolved");
  const [conflicts, setConflicts] = useState<Conflict[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchConflicts = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .conflicts({ status, persona, limit: 200 })
      .then((value) => {
        if (!cancelled) setConflicts(value);
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
  }, [status, persona]);

  useEffect(() => {
    return fetchConflicts();
  }, [fetchConflicts]);

  const resolveWith = (
    conflict: Conflict,
    body: { status: ConflictStatus; winner_id?: number }
  ) => {
    api
      .resolveConflict(conflict.id, body)
      .then(() => fetchConflicts())
      .catch((err) => setError(String(err)));
  };

  return (
    <PageBody>
      <PageHeader
        title="Суперечності"
        subtitle={`Пари памʼяток які системa запідозрила як несумісні. ${conflicts.length} записів.`}
        actions={
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as ConflictStatus)}
            className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-sm"
          >
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        }
      />
      {error ? <ErrorBox message={error} /> : null}
      {loading && conflicts.length === 0 ? <Loading /> : null}
      {!loading && conflicts.length === 0 && !error ? (
        <EmptyState message="Немає суперечностей у цій категорії." />
      ) : null}
      <ul className="space-y-4 mt-4">
        {conflicts.map((conflict) => (
          <li
            key={conflict.id}
            className="border border-zinc-800 rounded-lg p-4 bg-zinc-900/40 space-y-3"
          >
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-mono text-zinc-500">
                #{conflict.id}
              </span>
              <Pill tone={tone(conflict.resolution_status)}>
                {conflict.resolution_status}
              </Pill>
              <Pill>cosine {conflict.similarity.toFixed(3)}</Pill>
              {conflict.persona_key ? <Pill>{conflict.persona_key}</Pill> : null}
              {conflict.resolution_winner_id ? (
                <Pill tone="success">
                  winner #{conflict.resolution_winner_id}
                </Pill>
              ) : null}
            </div>
            <div className="grid md:grid-cols-2 gap-3">
              <SideCard side={conflict.memory_a} fallback={`#${conflict.memory_a_id}`} />
              <SideCard side={conflict.memory_b} fallback={`#${conflict.memory_b_id}`} />
            </div>
            {conflict.resolution_status === "unresolved" ? (
              <div className="flex gap-2 flex-wrap">
                {conflict.memory_a ? (
                  <IconButton
                    onClick={() =>
                      resolveWith(conflict, {
                        status: "superseded",
                        winner_id: conflict.memory_a_id,
                      })
                    }
                    title="A точніше — superseded B"
                    variant="primary"
                  >
                    A переможниця
                  </IconButton>
                ) : null}
                {conflict.memory_b ? (
                  <IconButton
                    onClick={() =>
                      resolveWith(conflict, {
                        status: "superseded",
                        winner_id: conflict.memory_b_id,
                      })
                    }
                    title="B точніше — superseded A"
                    variant="primary"
                  >
                    B переможниця
                  </IconButton>
                ) : null}
                <IconButton
                  onClick={() => resolveWith(conflict, { status: "kept_both" })}
                  title="Обидві лишаються"
                >
                  Лишити обидві
                </IconButton>
                <IconButton
                  onClick={() => resolveWith(conflict, { status: "dismissed" })}
                  title="Це не суперечність, забути пару"
                  variant="danger"
                >
                  Не суперечність
                </IconButton>
              </div>
            ) : null}
            <FooterMeta conflict={conflict} />
          </li>
        ))}
      </ul>
    </PageBody>
  );
}

function SideCard({
  side,
  fallback,
}: {
  side: ConflictSide | undefined;
  fallback: string;
}) {
  if (!side) {
    return (
      <div className="border border-zinc-800 rounded p-3 text-xs text-zinc-500">
        {fallback} — деталей нема (можливо видалена).
      </div>
    );
  }
  return (
    <div className="border border-zinc-800 rounded p-3">
      <div className="text-xs text-zinc-500 mb-1 flex flex-wrap gap-2 items-center">
        <span className="font-mono">#{side.id}</span>
        <Pill>{side.kind}</Pill>
        <Pill tone="info">{side.scope}</Pill>
        {side.superseded_by ? (
          <Pill tone="warn">superseded</Pill>
        ) : null}
      </div>
      <div className="text-sm text-zinc-200 whitespace-pre-wrap">{side.text}</div>
      <div className="mt-2 text-xs text-zinc-500">
        {side.created_at}
        {side.origin_message_id ? ` · ${side.origin_message_id}` : ""}
      </div>
    </div>
  );
}

function FooterMeta({ conflict }: { conflict: Conflict }) {
  const lastReasoning = (conflict.metadata?.["last_reasoning"] ??
    conflict.metadata?.["reasoning"]) as string | undefined;
  const lastConfidence = (conflict.metadata?.["last_confidence"] ??
    conflict.metadata?.["confidence"]) as number | undefined;
  return (
    <div className="text-xs text-zinc-500 space-y-1">
      <div>
        detected: {conflict.detected_at}
        {conflict.resolved_at ? ` · resolved: ${conflict.resolved_at}` : ""}
      </div>
      {lastReasoning ? (
        <div className="italic">model note: {lastReasoning}</div>
      ) : null}
      {typeof lastConfidence === "number" ? (
        <div>last model confidence: {lastConfidence.toFixed(2)}</div>
      ) : null}
    </div>
  );
}

function tone(status: ConflictStatus): "default" | "warn" | "success" | "info" {
  switch (status) {
    case "unresolved":
      return "warn";
    case "superseded":
      return "success";
    case "kept_both":
      return "info";
    case "dismissed":
      return "default";
  }
}
