"""Local HTTP interface. Importing this module does not open databases or models."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import logging
import sqlite3
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from api.config import Settings
from api.jobs import JobManager, JobStore, QueueFullError, TERMINAL, encode
from api.models import EvolutionRequest, PromptID, SolveRequest
from api.queries import HistoryQueries, diff
from api.service import Service
from src.evolution.prompt_registry import PromptRegistry


Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


def create_app(settings=None, *, client_factory=None, meta_factory=None, test_client_factory=None):
    settings = settings or Settings()
    factories = {k: v for k, v in {"client_factory": client_factory, "meta_factory": meta_factory,
                                  "test_client_factory": test_client_factory}.items() if v is not None}
    service = Service(settings, **factories)
    jobs = JobStore(settings.jobs_db)
    manager = JobManager(jobs, service.execute, settings.queue_capacity)
    history = HistoryQueries(settings.history_db)

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        try:
            yield
        finally:
            await asyncio.to_thread(manager.stop)

    app = FastAPI(title="SkillForge API", version="0.1.0", lifespan=lifespan)
    app.state.jobs, app.state.manager, app.state.service = jobs, manager, service
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Last-Event-ID"])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def check_origin(request, call_next):
        same_origin = str(request.base_url).rstrip("/")
        if request.method == "POST" and request.headers.get("origin") not in (None, same_origin, *settings.allowed_origins):
            return JSONResponse(status_code=403, content={"detail": "Origin is not allowed"})
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc.args[0])})

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(QueueFullError)
    async def full(request, exc):
        return JSONResponse(status_code=503, headers={"Retry-After": "5"}, content={"detail": str(exc)})

    async def unavailable(request, exc):
        logging.getLogger(__name__).error("API data access failed: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Configured data is unavailable; see server log."})

    app.add_exception_handler(OSError, unavailable)
    app.add_exception_handler(sqlite3.Error, unavailable)

    @app.get("/health")
    def health():
        return {"status": "ok" if manager.alive else "unavailable", "worker_alive": manager.alive,
                "jobs": jobs.counts()}

    @app.get("/catalogs")
    def catalogs():
        return {"items": list(settings.catalogs)}

    @app.get("/tasks")
    def tasks(catalog: str = "problems", limit: Limit = 50, offset: Offset = 0,
              split: Literal["train", "validation", "hidden", "test"] | None = None):
        _, dataset = service.catalog(catalog)
        selected = [task for task in dataset.tasks if split is None or task.split == split]
        names = ("task_id", "title", "problem_statement", "category", "difficulty", "split",
                 "language", "execution_mode", "entry_point")
        return {"total": len(selected), "items": [{name: getattr(task, name) for name in names}
                                                   for task in selected[offset:offset + limit]]}

    def submit(kind, body):
        job = manager.submit(kind, service.prepare(kind, body))
        return JSONResponse(status_code=202, headers={"Location": f"/jobs/{job['job_id']}"}, content=job)

    @app.post("/solves", status_code=202)
    def solve(body: SolveRequest):
        return submit("solve", body)

    @app.post("/evolutions", status_code=202)
    def evolve(body: EvolutionRequest):
        return submit("evolution", body)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise KeyError("unknown job")
        return job

    @app.get("/jobs/{job_id}/events")
    async def events(request: Request, job_id: str, after: Offset = 0,
                     last_event_id: Annotated[int | None, Header(ge=0)] = None):
        if await asyncio.to_thread(jobs.get, job_id) is None:
            raise HTTPException(404, "unknown job")

        async def stream():
            cursor = max(after, last_event_id or 0)
            idle = 0
            while not await request.is_disconnected():
                batch = await asyncio.to_thread(jobs.events, job_id, cursor)
                if not batch:
                    job = await asyncio.to_thread(jobs.get, job_id)
                    if job["status"] in TERMINAL or not manager.alive:
                        # Status and terminal event commit together. Drain after checking status.
                        batch = await asyncio.to_thread(jobs.events, job_id, cursor)
                        if not batch:
                            return
                if batch:
                    for event in batch:
                        cursor = event["id"]
                        yield f"id: {cursor}\nevent: {event['event']}\ndata: {encode(event['data'])}\n\n"
                    idle = 0
                    continue
                idle += 1
                if idle >= 50:
                    yield ": heartbeat\n\n"
                    idle = 0
                await asyncio.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/runs")
    def runs(limit: Limit = 50, offset: Offset = 0,
             split: Literal["adhoc", "train", "validation", "hidden", "test"] | None = None, prompt_id: PromptID | None = None):
        return history.runs(limit, offset, split, prompt_id)

    @app.get("/runs/{run_id}")
    def run(run_id: str):
        return history.run(run_id)

    @app.get("/prompts")
    def prompts():
        registry = PromptRegistry(settings.registry)
        return {"active_prompt_id": registry.active_prompt.prompt_id,
                "items": [asdict(version) for version in registry.versions]}

    @app.get("/prompts/diff")
    def prompt_diff(baseline: PromptID, candidate: PromptID):
        registry = PromptRegistry(settings.registry)
        known = {p.prompt_id for p in registry.versions}
        if baseline not in known or candidate not in known:
            raise KeyError("unknown prompt")
        return {"baseline": baseline, "candidate": candidate,
                "diff": diff(registry.get(baseline).prompt_text, registry.get(candidate).prompt_text, baseline, candidate)}

    @app.get("/benchmarks")
    def benchmarks(limit: Limit = 50, offset: Offset = 0):
        return history.benchmarks(limit, offset)

    @app.get("/benchmarks/{run_id}")
    def benchmark(run_id: str):
        return history.benchmark(run_id)

    @app.get("/evolutions")
    def evolutions(limit: Limit = 50, offset: Offset = 0):
        return history.evolutions(limit, offset)

    @app.get("/evolutions/readiness")
    def readiness(catalog: str = "scala_http"):
        return service.readiness(catalog)

    @app.get("/model-requests/{owner_id}")
    def model_requests(owner_id: str, limit: Limit = 50, offset: Offset = 0):
        return history.model_requests(owner_id, limit, offset)

    @app.get("/evolutions/{round_id}")
    def evolution(round_id: str):
        return history.evolution(round_id)

    return app
