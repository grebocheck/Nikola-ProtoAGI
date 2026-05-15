import type { ReactNode } from "react";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 mb-6">
      <div>
        <h1 className="text-2xl font-semibold text-zinc-100">{title}</h1>
        {subtitle ? (
          <p className="text-sm text-zinc-500 mt-1">{subtitle}</p>
        ) : null}
      </div>
      {actions ? <div className="flex gap-2 items-center">{actions}</div> : null}
    </div>
  );
}

export function PageBody({ children }: { children: ReactNode }) {
  return <div className="px-8 py-6 max-w-7xl mx-auto">{children}</div>;
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="border border-dashed border-zinc-800 rounded-lg px-6 py-12 text-center text-zinc-500">
      {message}
    </div>
  );
}

export function Loading({ message = "Завантаження…" }: { message?: string }) {
  return <div className="text-sm text-zinc-500">{message}</div>;
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="border border-red-900/60 bg-red-950/30 text-red-300 text-sm rounded px-4 py-3">
      {message}
    </div>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`bg-zinc-900/50 border border-zinc-800 rounded-lg p-4 ${className}`}
    >
      {children}
    </div>
  );
}

export function Pill({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "warn" | "success" | "info";
}) {
  const tones: Record<string, string> = {
    default: "bg-zinc-800 text-zinc-300",
    warn: "bg-amber-950/40 text-amber-300 border border-amber-900/50",
    success: "bg-emerald-950/40 text-emerald-300 border border-emerald-900/50",
    info: "bg-sky-950/40 text-sky-300 border border-sky-900/50",
  };
  return (
    <span
      className={`inline-flex items-center text-xs px-2 py-0.5 rounded-full ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function IconButton({
  onClick,
  title,
  children,
  variant = "default",
}: {
  onClick: () => void;
  title: string;
  children: ReactNode;
  variant?: "default" | "danger" | "primary";
}) {
  const styles: Record<string, string> = {
    default:
      "border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100",
    danger:
      "border-red-900/60 text-red-300 hover:bg-red-950/40 hover:text-red-200",
    primary:
      "border-sky-700/60 text-sky-300 hover:bg-sky-950/40 hover:text-sky-200",
  };
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`text-xs border rounded px-2 py-1 transition-colors ${styles[variant]}`}
    >
      {children}
    </button>
  );
}
