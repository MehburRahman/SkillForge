"""Run the local API with a single durable model-job worker."""

import argparse

import uvicorn

from api.config import Settings
from api.main import create_app


def main(argv=None):
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--history-db", default=defaults.history_db)
    parser.add_argument("--prompt-registry", default=defaults.registry)
    parser.add_argument("--jobs-db", default=defaults.jobs_db)
    parser.add_argument("--cache-dir", default=defaults.cache_dir)
    parser.add_argument("--queue-capacity", type=int, default=defaults.queue_capacity)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.queue_capacity < 1:
        parser.error("queue capacity must be positive")
    settings = Settings(history_db=args.history_db, registry=args.prompt_registry, jobs_db=args.jobs_db,
                        cache_dir=args.cache_dir, queue_capacity=args.queue_capacity)
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
