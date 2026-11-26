"""Persistent local jobs, replayable events, and one model worker."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4


TERMINAL = frozenset({"completed", "failed", "interrupted"})
log = logging.getLogger(__name__)


class QueueFullError(ValueError):
    pass


class WorkerAlreadyRunningError(RuntimeError):
    """A live worker already owns this job database."""


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class JobStore:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    request_json TEXT NOT NULL, result_json TEXT, error TEXT, resource_id TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_job_queue ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    event TEXT NOT NULL, data_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_job_events ON events(job_id, event_id);
            """)
            # The process lock guarantees no previous worker is still running.
            for row in db.execute("SELECT job_id FROM jobs WHERE status='running'").fetchall():
                message = "Server stopped during this job. Inspect saved progress before submitting again."
                db.execute("UPDATE jobs SET status='interrupted', error=?, updated_at=? WHERE job_id=?",
                           (message, now(), row["job_id"]))
                self._event(db, row["job_id"], "interrupted", {"message": message})

    @staticmethod
    def _event(db, job_id, event, data):
        db.execute("INSERT INTO events(job_id,event,data_json,created_at) VALUES (?,?,?,?)",
                   (job_id, event, encode(data), now()))

    def enqueue(self, kind, payload, capacity):
        job_id, timestamp = str(uuid4()), now()
        with self.connection() as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= capacity:
                raise QueueFullError("The local job queue is full. Wait for a job to finish.")
            db.execute("INSERT INTO jobs VALUES (?,?,'queued',?,NULL,NULL,?,?,?)",
                       (job_id, kind, encode(payload), payload.get("run_id"), timestamp, timestamp))
            self._event(db, job_id, "queued", {"kind": kind})
        return self.get(job_id)

    def claim(self):
        with self.connection() as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,rowid LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated_at=? WHERE job_id=?", (now(), row["job_id"]))
            self._event(db, row["job_id"], "running", {})
            return dict(row)

    def emit(self, job_id, event, data):
        with self.connection() as db, db:
            self._event(db, job_id, event, data)
            db.execute("UPDATE jobs SET updated_at=?,resource_id=COALESCE(?,resource_id) WHERE job_id=?",
                       (now(), data.get("history_run_id"), job_id))

    def finish(self, job_id, *, result=None, error=None):
        status = "failed" if error is not None or (isinstance(result, dict) and result.get("status") == "failed") else "completed"
        with self.connection() as db, db:
            db.execute("UPDATE jobs SET status=?,result_json=?,error=?,updated_at=? WHERE job_id=?",
                       (status, encode(result) if result is not None else None, error, now(), job_id))
            self._event(db, job_id, status, {"result": result, "error": error})

    def get(self, job_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            draft = db.execute(
                "SELECT json_extract(data_json, '$.code') FROM events "
                "WHERE job_id=? AND event='stage' AND json_extract(data_json, '$.stage')='draft_ready' "
                "ORDER BY event_id DESC LIMIT 1", (job_id,),
            ).fetchone() if row is not None else None
        if row is None:
            return None
        value = dict(row)
        payload = json.loads(value.pop("request_json"))
        # Restore the submitted UI settings without exposing internal paths,
        # queued context, catalog hashes, or executable test fixtures.
        public_fields = {"problem", "task_id", "catalog", "language", "execution_mode", "entry_point",
                         "model", "judge_model", "llm_judge", "prompt_id", "config", "seed", "max_attempts",
                         "reasoning_mode", "test_model", "test_generation_method"}
        value["request"] = {key: item for key, item in payload.get("request", {}).items() if key in public_fields}
        value["result"] = json.loads(value.pop("result_json") or "null")
        value["draft_code"] = draft[0] if draft is not None else None
        return value

    def events(self, job_id, after=0, limit=100):
        with self.connection() as db:
            rows = db.execute("SELECT event_id,event,data_json,created_at FROM events "
                              "WHERE job_id=? AND event_id>? ORDER BY event_id LIMIT ?",
                              (job_id, after, limit)).fetchall()
        return [{"id": r["event_id"], "event": r["event"], "data": json.loads(r["data_json"]),
                 "created_at": r["created_at"]} for r in rows]

    def counts(self):
        with self.connection() as db:
            return {r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}


class JobManager:
    def __init__(self, store, executor, capacity=8):
        self.store, self.executor, self.capacity = store, executor, capacity
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._lock = None

    def start(self):
        import fcntl
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.store.path.with_suffix(self.store.path.suffix + ".lock")
        handle = lock_path.open("a+")
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.seek(0)
                owner = handle.read(64).strip()
                process = f" (PID {owner})" if owner.isascii() and owner.isdecimal() and int(owner) > 0 else ""
                raise WorkerAlreadyRunningError(
                    f"Another SkillForge API worker{process} already owns the job database "
                    f"{self.store.path}. Stop the existing server before starting another. "
                    f"Find the owner with: lsof {lock_path}."
                ) from exc
            # Write only after acquiring the lock; a failed second startup must
            # not change the owner's metadata or initialize its live database.
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            self.store.initialize()
        except BaseException:
            handle.close()
            raise
        self._lock = handle
        self._stop.clear()
        self._thread = threading.Thread(target=self._work, name="skillforge-model-worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()  # Finish the active job; queued jobs stay durable.
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def submit(self, kind, payload):
        if not self.alive:
            raise QueueFullError("The job worker is not available.")
        job = self.store.enqueue(kind, payload, self.capacity)
        self._wake.set()
        return job

    def _work(self):
        while not self._stop.is_set():
            self._wake.clear()
            job = self.store.claim()
            if job is None:
                self._wake.wait(1)
                continue
            try:
                result = self.executor(job["kind"], json.loads(job["request_json"]),
                                       lambda event, data: self.store.emit(job["job_id"], event, data))
                self.store.finish(job["job_id"], result=result)
            except Exception as exc:
                log.exception("Job %s failed", job["job_id"])
                message = str(exc)[:1000] if isinstance(exc, (ValueError, RuntimeError)) else "Job failed; see server log."
                self.store.finish(job["job_id"], error=message)
