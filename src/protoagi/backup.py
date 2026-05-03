from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class BackupError(RuntimeError):
    pass


def default_backup_path(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return db_path.parent / "backups" / f"{stamp}.sqlite3"


def backup_database(db_path: Path, to_path: Path | None = None) -> Path:
    source_path = db_path.resolve()
    if not source_path.exists():
        raise BackupError(f"database does not exist: {source_path}")
    target_path = (to_path or default_backup_path(source_path)).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path == target_path:
        raise BackupError("backup target must be different from the source database")

    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(target_path)
        try:
            source.backup(target)
            target.commit()
        finally:
            target.close()
    finally:
        source.close()
    validate_database(target_path)
    return target_path


def restore_database(db_path: Path, from_path: Path) -> Path:
    source_path = from_path.resolve()
    target_path = db_path.resolve()
    if not source_path.exists():
        raise BackupError(f"restore source does not exist: {source_path}")
    if source_path == target_path:
        raise BackupError("restore source must be different from the target database")
    validate_database(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target_path.with_name(f".{target_path.name}.restore-{os.getpid()}.tmp")
    try:
        source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        try:
            temp = sqlite3.connect(temp_path)
            try:
                source.backup(temp)
                temp.commit()
            finally:
                temp.close()
        finally:
            source.close()
        validate_database(temp_path)
        os.replace(temp_path, target_path)
        _remove_sqlite_sidecars(target_path)
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
    return target_path


def validate_database(path: Path) -> None:
    db_path = path.resolve()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise BackupError(f"cannot open database: {db_path}") from exc
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]).lower() != "ok":
            detail = "unknown" if integrity is None else str(integrity[0])
            raise BackupError(f"integrity_check failed: {detail}")
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('memory_items', 'kv')
            """
        ).fetchall()
        names = {str(row[0]) for row in rows}
        missing = {"memory_items", "kv"} - names
        if missing:
            raise BackupError(f"not a ProtoAGI memory database; missing {sorted(missing)}")
    except sqlite3.Error as exc:
        raise BackupError(f"database validation failed: {db_path}") from exc
    finally:
        conn.close()


def _remove_sqlite_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        try:
            Path(str(db_path) + suffix).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise BackupError(f"could not remove SQLite sidecar {db_path}{suffix}: {exc}") from exc


__all__ = [
    "BackupError",
    "backup_database",
    "default_backup_path",
    "restore_database",
    "validate_database",
]
