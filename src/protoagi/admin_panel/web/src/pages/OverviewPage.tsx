import { useEffect, useState } from "react";
import { Card, ErrorBox, Loading, PageBody, PageHeader } from "../components/Page";
import { usePersona } from "../components/Sidebar";
import { api, type HealthSummary } from "../lib/api";

export function OverviewPage() {
  const persona = usePersona();
  const [health, setHealth] = useState<HealthSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setHealth(null);
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
    <PageBody>
      <PageHeader
        title="Огляд"
        subtitle={`Стан памʼяті персони ${persona}`}
      />
      {error ? <ErrorBox message={error} /> : null}
      {!health && !error ? <Loading /> : null}
      {health ? (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <StatCard
            label="Активна память"
            value={health.memories_active}
            hint={`+${health.memories_superseded} superseded`}
          />
          <StatCard
            label="Відкриті цілі"
            value={health.open_goals}
            hint="Незакриті обіцянки персони"
          />
          <StatCard
            label="Невирішені суперечності"
            value={health.unresolved_conflicts}
            hint="Чекають перегляду"
            tone={health.unresolved_conflicts > 0 ? "warn" : "default"}
          />
          <StatCard
            label="Моделі співрозмовників"
            value={health.user_states_tracked}
            hint="user_state записів"
          />
          <StatCard
            label="Всього у памʼяті"
            value={health.memories_total}
            hint="з усіма superseded"
          />
        </div>
      ) : null}
    </PageBody>
  );
}

function StatCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: number;
  hint?: string;
  tone?: "default" | "warn";
}) {
  return (
    <Card>
      <div className="text-xs uppercase tracking-wide text-zinc-500">{label}</div>
      <div
        className={[
          "text-3xl font-semibold tabular-nums mt-2",
          tone === "warn" ? "text-amber-300" : "text-zinc-100",
        ].join(" ")}
      >
        {value}
      </div>
      {hint ? <div className="text-xs text-zinc-500 mt-2">{hint}</div> : null}
    </Card>
  );
}
