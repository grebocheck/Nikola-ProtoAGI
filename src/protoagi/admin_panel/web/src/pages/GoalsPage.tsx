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
import { api, type Goal } from "../lib/api";

const STATUSES = ["open", "completed", "abandoned", "all"] as const;

export function GoalsPage() {
  const persona = usePersona();
  const [status, setStatus] = useState<(typeof STATUSES)[number]>("open");
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchGoals = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .goals({ status, persona, limit: 200 })
      .then((value) => {
        if (!cancelled) setGoals(value);
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
    return fetchGoals();
  }, [fetchGoals]);

  const close = (goal: Goal, statusValue: "completed" | "abandoned") => {
    api
      .updateGoal(goal.id, { status: statusValue })
      .then(() => fetchGoals())
      .catch((err) => setError(String(err)));
  };

  const reopen = (goal: Goal) => {
    api
      .updateGoal(goal.id, { status: "open" })
      .then(() => fetchGoals())
      .catch((err) => setError(String(err)));
  };

  return (
    <PageBody>
      <PageHeader
        title="Цілі"
        subtitle={`Незакриті обіцянки персони ${persona}. ${goals.length} записів.`}
        actions={
          <select
            value={status}
            onChange={(e) =>
              setStatus(e.target.value as (typeof STATUSES)[number])
            }
            className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-sm"
          >
            {STATUSES.map((value) => (
              <option key={value} value={value}>
                {labelFor(value)}
              </option>
            ))}
          </select>
        }
      />
      {error ? <ErrorBox message={error} /> : null}
      {loading && goals.length === 0 ? <Loading /> : null}
      {!loading && goals.length === 0 && !error ? (
        <EmptyState message="Немає цілей у цій категорії." />
      ) : null}
      <ul className="space-y-2 mt-4">
        {goals.map((goal) => (
          <li
            key={goal.id}
            className="border border-zinc-800 rounded-lg p-4 bg-zinc-900/40"
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-mono text-zinc-500">
                  #{goal.id}
                </span>
                <Pill tone={pillTone(goal.status)}>{goal.status}</Pill>
                <Pill>priority {goal.priority.toFixed(2)}</Pill>
                {goal.due_at ? <Pill tone="info">due {goal.due_at}</Pill> : null}
                {goal.chat_id ? <Pill>chat {goal.chat_id}</Pill> : null}
              </div>
              <div className="flex gap-2 shrink-0">
                {goal.status === "open" ? (
                  <>
                    <IconButton
                      onClick={() => close(goal, "completed")}
                      title="Позначити як виконане"
                      variant="primary"
                    >
                      Complete
                    </IconButton>
                    <IconButton
                      onClick={() => close(goal, "abandoned")}
                      title="Покинути"
                      variant="danger"
                    >
                      Abandon
                    </IconButton>
                  </>
                ) : (
                  <IconButton onClick={() => reopen(goal)} title="Відкрити знову">
                    Reopen
                  </IconButton>
                )}
              </div>
            </div>
            <div className="text-sm text-zinc-200 whitespace-pre-wrap">
              {goal.text}
            </div>
            <div className="mt-2 text-xs text-zinc-500 flex flex-wrap gap-x-4 gap-y-1">
              <span>created: {goal.created_at}</span>
              <span>touched: {goal.last_touched_at}</span>
              {goal.closed_at ? <span>closed: {goal.closed_at}</span> : null}
            </div>
          </li>
        ))}
      </ul>
    </PageBody>
  );
}

function labelFor(status: string): string {
  switch (status) {
    case "open":
      return "відкриті";
    case "completed":
      return "виконані";
    case "abandoned":
      return "покинуті";
    case "all":
      return "всі";
    default:
      return status;
  }
}

function pillTone(status: string): "default" | "success" | "warn" | "info" {
  switch (status) {
    case "open":
      return "info";
    case "completed":
      return "success";
    case "abandoned":
      return "warn";
    default:
      return "default";
  }
}
