"""Compare direct and planned solving on frozen tests using the same local model."""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.code_prompts import prompt_template_identity
from src.agents.llm_client import LLMClient, ollama
from src.agents.test_generator import load_test_suite
from src.evolution.prompt_registry import load_default_registry, PromptRegistry
from src.experiments import preflight_execution, validate_experiment
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop
from src.verifier.outcome_verifier import verify


def _model_identity(model):
    """Pin the actual local weights, since a model tag can be replaced."""
    if ollama is None:
        raise RuntimeError('Install the Ollama Python client before comparing models.')
    name = model if ':' in model.rsplit('/', 1)[-1] else model + ':latest'
    response = ollama.list()
    models = response.get('models', []) if isinstance(response, dict) else response.models
    for item in models:
        actual_name = item.get('model') or item.get('name') if isinstance(item, dict) else item.model
        digest = item.get('digest') if isinstance(item, dict) else item.digest
        if actual_name == name and digest:
            return {'name': actual_name, 'digest': digest}
    raise RuntimeError(f'Local model {name!r} is not available with a recorded digest.')


def _usage(history_path, owner_id):
    if owner_id is None:
        return {}
    with sqlite3.connect(f'file:{history_path.resolve()}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT usage_json, duration_seconds FROM model_requests WHERE owner_id=?',
                          (owner_id,)).fetchall()
    usages = [json.loads(row[0]) for row in rows]
    return dict(recorded_model_calls=len(rows),
                backend_requests=sum(u.get('backend_request_count', 1) for u in usages),
                infrastructure_retries=sum(len(u.get('infrastructure_retries', [])) for u in usages),
                usage_recorded_calls=sum(u.get('prompt_eval_count') is not None and u.get('eval_count') is not None
                                         for u in usages),
                prompt_tokens=sum(u.get('prompt_eval_count') or 0 for u in usages),
                output_tokens=sum(u.get('eval_count') or 0 for u in usages),
                model_seconds=sum(row[1] for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem-file', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--hidden-tests', type=Path)
    parser.add_argument('--language', choices=['python', 'javascript', 'scala'], default='python')
    parser.add_argument('--model', default='qwen2.5-coder:7b')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--context-window', type=int, default=8192)
    parser.add_argument('--strategies', nargs='+', choices=['full', 'best_of_n', 'single_shot', 'scalar_repair'],
                        default=['full'], help='Compare execution-selected sampling and repair under recorded budgets.')
    parser.add_argument('--first-mode', choices=['direct', 'plan_critique'], default='direct',
                        help='Reverse the order in a separate run to assess model/cache warmup effects.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('output directory must be empty; choose a new comparison directory')
    validate_experiment('full', args.seed, args.language)
    if args.max_attempts < 1:
        parser.error('max-attempts must be at least 1')
    LLMClient(args.model, context_window=args.context_window)
    statement = args.problem_file.read_text()
    if not statement.strip():
        parser.error('problem statement must not be empty')
    suite = load_test_suite(args.tests)
    hidden = load_test_suite(args.hidden_tests) if args.hidden_tests else None
    if hidden is not None and {c.input for c in suite.cases} & {c.input for c in hidden.cases}:
        parser.error('visible and hidden test inputs must not overlap')
    environment = preflight_execution(args.language)
    identity = _model_identity(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = load_default_registry().active_prompt
    registry_file = args.output_dir / 'prompts.json'
    registry_file.write_text(json.dumps({'schema_version': 1, 'prompts': [asdict(prompt)]}))
    registry = PromptRegistry(registry_file)
    modes = [args.first_mode, 'plan_critique' if args.first_mode == 'direct' else 'direct']
    manifest = dict(model=args.model, model_identity=identity, toolchain=environment,
                    language=args.language, seed=args.seed, config='full', mode_order=modes,
                    max_attempts=args.max_attempts, context_window=args.context_window,
                    prompt_templates=prompt_template_identity(),
                    problem_sha256=hashlib.sha256(statement.encode()).hexdigest(),
                    visible_sha256=hashlib.sha256(args.tests.read_bytes()).hexdigest(),
                    hidden_sha256=hashlib.sha256(args.hidden_tests.read_bytes()).hexdigest() if hidden else None,
                    candidate_output_limit=4096, judge_enabled=False)
    manifest['strategies'] = args.strategies
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summaries = []
    history_path = args.output_dir / 'history.sqlite3'
    with ExperimentStore(history_path) as store:
        for mode, strategy in itertools.product(modes, args.strategies):
            print(json.dumps({'mode': mode, 'stage': 'starting'}), flush=True)
            started = time.perf_counter()
            summary = dict(mode=mode, strategy=strategy)
            owner_id = None

            def on_stage(name, data):
                nonlocal owner_id
                if name == 'request_logging':
                    owner_id = data.get('request_log_id')
                print(json.dumps({'mode': mode, 'stage': name, **data}), flush=True)

            try:
                if _model_identity(args.model) != identity:
                    raise RuntimeError('Model weights changed during comparison; start a new run.')
                client = LLMClient(args.model, context_window=args.context_window)
                client.generation_options['num_predict'] = 4096
                result = run_candidate_refinement_loop(client, statement, trusted_test_suite=suite,
                        language=args.language, config_name=strategy, seed=args.seed,
                        max_attempts=args.max_attempts, reasoning_mode=mode, history_store=store,
                        prompt_registry=registry,
                        on_stage=on_stage,
                        on_attempt=lambda attempt: print(json.dumps({'mode': mode, 'attempt': attempt.attempt_number,
                                                                    'passed': attempt.verification.passed,
                                                                    'total': attempt.verification.total}), flush=True))
                label = mode if args.strategies == ['full'] else mode + '-' + strategy
                (args.output_dir / (label + '.source')).write_text(result.best_code)
                summary['stage_metrics'] = getattr(result, 'stage_metrics', {})
                summary.update(run_id=result.history_run_id,
                           first_passed=result.attempts[0].verification.passed,
                           best_passed=result.best_attempt.verification.passed, visible_total=len(suite.cases),
                           attempts=len(result.attempts), model_calls=result.model_calls,
                           seconds=result.wall_clock_seconds, status=result.best_attempt.evaluation_report.status.value)
                if hidden is not None:
                    hidden_started = time.perf_counter()
                    try:
                        verification = verify(result.best_code, hidden, language=args.language, max_failures=0)
                        summary.update(hidden_passed=verification.passed, hidden_total=verification.total)
                    except Exception as exc:
                        summary['hidden_error'] = f'{type(exc).__name__}: {exc}'
                    finally:
                        summary['hidden_seconds'] = time.perf_counter() - hidden_started
            except Exception as exc:
                summary.update(status='run_error', error=f'{type(exc).__name__}: {exc}',
                               seconds=time.perf_counter() - started)
            summary.update(request_log_id=owner_id, **_usage(history_path, owner_id))
            summaries.append(summary)
            (args.output_dir / 'results.json').write_text(json.dumps(summaries, indent=2))
            print(json.dumps(summary), flush=True)
    return int(any(s.get('error') or s.get('hidden_error') for s in summaries))


if __name__ == '__main__':
    raise SystemExit(main())
