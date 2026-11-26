"""Read-only views over experiment history. Checkpoints stay private."""

from contextlib import contextmanager
from dataclasses import fields
from difflib import unified_diff
import json
import sqlite3

from src.benchmark.models import BenchmarkSummary, TaskBenchmarkResult


def diff(before, after, before_name, after_name):
    return "".join(unified_diff([line + "\n" for line in before.splitlines()],
                                [line + "\n" for line in after.splitlines()],
                                fromfile=before_name, tofile=after_name))


def decoded(row):
    return {key[:-5] if key.endswith("_json") else key:
            json.loads(value) if key.endswith("_json") and value is not None else value
            for key, value in dict(row).items()}


class HistoryQueries:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def connection(self, table):
        if not self.path.exists():
            yield None
            return
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            yield db if exists else None
        finally:
            db.close()

    def runs(self, limit, offset, split=None, prompt_id=None):
        clauses, args = [], []
        for name, value in (("benchmark_split", split), ("prompt_version", prompt_id)):
            if value is not None:
                clauses.append(name + "=?")
                args.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection("runs") as db:
            if db is None:
                return {"items": [], "total": 0}
            total = db.execute("SELECT COUNT(*) FROM runs" + where, args).fetchone()[0]
            rows = db.execute("SELECT * FROM runs" + where + " ORDER BY started_at DESC,run_id LIMIT ? OFFSET ?",
                              [*args, limit, offset]).fetchall()
        return {"items": [dict(row) for row in rows], "total": total}

    def run(self, run_id):
        with self.connection("runs") as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone() if db else None
            if row is None:
                raise KeyError("unknown run")
            result = dict(row)
            result["attempts"] = [decoded(r) for r in db.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY attempt_number", (run_id,))]
            result["failure_signals"] = [decoded(r) for r in db.execute(
                "SELECT * FROM failure_signals WHERE run_id=? ORDER BY attempt_number,signal_index", (run_id,))]
        previous = None
        for attempt in result["attempts"]:
            attempt["code_diff"] = (diff(previous["candidate_code"], attempt["candidate_code"],
                                         str(previous["attempt_number"]), str(attempt["attempt_number"]))
                                    if previous else None)
            previous = attempt
        return result

    @staticmethod
    def benchmark_header(row):
        manifest = json.loads(row["manifest_json"])
        return {"run_id": row["run_id"], "started_at": row["started_at"], "completed_at": row["completed_at"],
                "expected_tasks": len(manifest["task_ids"]),
                **{key: manifest.get(key) for key in ("prompt_id", "generator_model", "judge_model", "config_name", "seed")}}

    def benchmarks(self, limit, offset):
        with self.connection("benchmark_runs") as db:
            if db is None:
                return {"items": [], "total": 0}
            total = db.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
            rows = db.execute("SELECT * FROM benchmark_runs ORDER BY started_at DESC,run_id LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()
        return {"items": [self.benchmark_header(row) for row in rows], "total": total}

    def benchmark(self, run_id):
        with self.connection("benchmark_runs") as db:
            row = db.execute("SELECT * FROM benchmark_runs WHERE run_id=?", (run_id,)).fetchone() if db else None
            if row is None:
                raise KeyError("unknown benchmark")
            header = self.benchmark_header(row)
            # Never select checkpoints: they may contain hidden test inputs and outputs.
            tasks = db.execute("SELECT result_json,interrupted_model_calls,interrupted_wall_clock_seconds "
                               "FROM benchmark_tasks WHERE benchmark_run_id=? ORDER BY task_id", (run_id,)).fetchall()
        names = {f.name for f in fields(TaskBenchmarkResult)}
        summary = BenchmarkSummary(prompt_version=header["prompt_id"], run_id=run_id)
        for task in tasks:
            if task["result_json"] is not None:
                result = json.loads(task["result_json"])
                summary.task_results.append(TaskBenchmarkResult(**{k: v for k, v in result.items() if k in names}))
            summary.interrupted_model_calls += task["interrupted_model_calls"]
            summary.interrupted_wall_clock_seconds += task["interrupted_wall_clock_seconds"]
        return {**header, "summary": summary.to_dict()}

    def evolutions(self, limit, offset):
        with self.connection("evolution_rounds") as db:
            if db is None:
                return {"items": [], "total": 0}
            total = db.execute("SELECT COUNT(*) FROM evolution_rounds").fetchone()[0]
            rows = db.execute("SELECT round_id,status,trigger_id,candidate_prompt_id,validation_id,"
                              "meta_model_calls,meta_wall_clock_seconds,created_at,updated_at,result_json "
                              "FROM evolution_rounds ORDER BY created_at DESC,round_id LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()
        return {"items": [decoded(row) for row in rows], "total": total}

    def model_requests(self, owner_id, limit=50, offset=0):
        with self.connection('model_requests') as db:
            if db is None:
                return {'items': [], 'total': 0}
            total = db.execute('SELECT COUNT(*) FROM model_requests WHERE owner_id=?', (owner_id,)).fetchone()[0]
            rows = db.execute('SELECT * FROM model_requests WHERE owner_id=? ORDER BY call_id LIMIT ? OFFSET ?',
                              (owner_id, limit, offset)).fetchall()
            return {'items': [decoded(row) for row in rows], 'total': total}

    def evolution(self, round_id):
        with self.connection("evolution_rounds") as db:
            row = db.execute("SELECT round_id,status,trigger_id,candidate_prompt_id,validation_id,"
                             "meta_model_calls,meta_wall_clock_seconds,created_at,updated_at,result_json "
                             "FROM evolution_rounds WHERE round_id=?", (round_id,)).fetchone() if db else None
            if row is None:
                raise KeyError("unknown evolution round")
            result = decoded(row)
            validation = db.execute("SELECT result_json FROM prompt_validations WHERE validation_id=?",
                                    (row["validation_id"],)).fetchone()
            result["validation"] = json.loads(validation[0]) if validation and validation[0] else None
        with self.connection('proposal_rejections') as db:
            result['proposal_rejections'] = ([dict(r) for r in db.execute(
                'SELECT response_text,error,created_at FROM proposal_rejections WHERE decision_id=? ORDER BY rejection_id',
                (result['trigger_id'],))] if db else [])
        return result
