from dataclasses import dataclass, field
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    history_db: Path = PROJECT / "data" / "experiments.sqlite3"
    registry: Path = PROJECT / "data" / "prompts.json"
    jobs_db: Path = PROJECT / "data" / "api_jobs.sqlite3"
    cache_dir: Path = PROJECT / "data" / "task_spec_cache"
    catalogs: dict[str, Path] = field(default_factory=lambda: {
        "problems": PROJECT / "data" / "problems.json",
        "mbpp": PROJECT / "benchmark" / "mbpp_plus_240_python.json",
        "scala_algorithms": PROJECT / "data" / "scala_algorithms.json",
        "scala_functions": PROJECT / "data" / "scala_functions.json",
        "scala_http": PROJECT / "data" / "scala_http.json",
    })
    allowed_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    queue_capacity: int = 8

    def __post_init__(self):
        for name in ("history_db", "registry", "jobs_db", "cache_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        object.__setattr__(self, "catalogs", {key: Path(value).resolve() for key, value in self.catalogs.items()})
        if self.jobs_db == self.history_db:
            raise ValueError("API jobs and experiment history require separate databases")
        if type(self.queue_capacity) is not int or self.queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
