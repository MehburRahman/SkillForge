"""Run or resume one complete prompt-evolution round."""

from src.evolution.scope import validate_evolution_scope

import argparse
import json
from pathlib import Path

from src.agents.llm_client import DEFAULT_MODEL, DEFAULT_CONTEXT_WINDOW, LLMClient, build_clients
from src.benchmark.loader import load_benchmark
from src.benchmark.runner import select_benchmark_tasks
from src.evolution.promotion import check_rollbacks
from src.evolution.prompt_registry import DEFAULT_REGISTRY_PATH, PromptRegistry
from src.evolution.workflow import run_evolution_round
from src.experiments import validate_seed
from src.history import DEFAULT_HISTORY_PATH, ExperimentStore


def build_parser():
    parser = argparse.ArgumentParser(description="Propose, validate and apply one prompt evolution round.")
    parser.add_argument("benchmark", nargs="?", default=str(
        Path(__file__).resolve().parent / "benchmark" / "mbpp_plus_240_python.json"))
    parser.add_argument("--history-db", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--prompt-registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--meta-model", help="Proposal model; defaults to the generator.")
    parser.add_argument("--judge-model")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=3, help="Same fixed budget for both validation arms.")
    parser.add_argument("--proposal-attempts", type=int, default=2)
    parser.add_argument("--max-observations", type=int, default=50)
    parser.add_argument("--run-id", help="Persistent evolution round ID; printed before model calls.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-proposal", action="store_true", help="Retry a recorded proposal-validation failure once using its frozen evidence; requires --resume.")
    parser.add_argument("--check-rollback-only", action="store_true",
                        help="Check saved training outcomes and recover pending actions without model calls.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_seed(args.seed)
        if args.resume and not args.run_id:
            raise ValueError("--resume requires --run-id")
        if args.retry_proposal and not args.resume:
            raise ValueError('--retry-proposal requires --resume')
        if not 1024 <= args.context_window <= 32768:
            raise ValueError('--context-window must be between 1024 and 32768')
        if min(args.max_attempts, args.proposal_attempts, args.max_observations) < 1:
            raise ValueError("attempt and observation limits must be positive")
        if args.check_rollback_only and (args.resume or args.run_id):
            raise ValueError("--check-rollback-only cannot be combined with --run-id or --resume")
        registry = PromptRegistry(args.prompt_registry)
        if args.check_rollback_only:
            with ExperimentStore(args.history_db) as store:
                result = {"checks": check_rollbacks(store, registry),
                          "active_prompt_id": registry.active_prompt.prompt_id,
                          "monitors": store.rollback_monitors()}
        else:
            dataset = load_benchmark(args.benchmark)
            tasks = select_benchmark_tasks(dataset, splits=["validation"])
            validate_evolution_scope(tasks)
            client, judge = build_clients(args.model, args.judge_model, args.llm_judge, context_window=args.context_window)
            meta = LLMClient(args.meta_model, context_window=args.context_window) if args.meta_model else client
            with ExperimentStore(args.history_db) as store:
                result = run_evolution_round(
                    client, dataset, registry, store, meta_client=meta, judge_client=judge,
                    seed=args.seed, max_attempts=args.max_attempts, round_id=args.run_id, resume=args.resume,
                    max_observations=args.max_observations, proposal_attempts=args.proposal_attempts,
                    on_progress=lambda message: print(message, flush=True),
                    retry_proposal=args.retry_proposal,
                )
        print("\nEVOLUTION SUMMARY")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") == "failed":
            parser.exit(1)
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted. Resume using the printed --run-id and --resume.\n")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"\nERROR: {exc}\nUse the same --run-id and --resume to inspect or recover the round; "
                       "failed proposals are not regenerated.\n")


if __name__ == "__main__":
    main()
