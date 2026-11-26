"""Review browser evaluation archives; hidden verification never calls a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmark.loader import load_benchmark
from src.evaluation.runtime_grader import grade_runtime_robustness
from src.verifier.outcome_verifier import verify
from src.verifier.runners.scala import preflight


def failure_types(attempt, mode):
    runtime = attempt['runtime_grade']
    functional = attempt['functional_grade']
    types = set()
    if runtime.get('compilation_error_count', 0):
        types.add('compilation_error')
    if runtime.get('timeout_count', 0):
        types.add('timeout')
    for evidence in runtime.get('failure_evidence', []):
        text = evidence.get('stderr', '')
        if mode == 'stdio' and any(kind in text for kind in ('NumberFormatException', 'NoSuchElementException', 'EOFException')):
            types.add('input_parsing')
        elif evidence.get('status') == 'error':
            types.add('runtime_exception')
    if functional.get('score', 0) < 1 and not types:
        types.add('http_response' if mode == 'http' else 'incorrect_answer')
    return sorted(types)


def summarize(records, requests=()):
    rows, counts = [], Counter()
    for record in records:
        job, run = record['job'], record.get('run')
        if job['status'] != 'completed' or not run or not run.get('attempts'):
            rows.append({'task_id': record['task']['task_id'], 'infrastructure_error': job.get('error') or job['status']})
            continue
        attempts = run['attempts']
        best = next(a for a in attempts if a['attempt_number'] == run['best_attempt_number'])
        first_pass = attempts[0]['functional_grade']['score'] == 1 and bool(attempts[0]['hard_gate_passed'])
        final_pass = best['functional_grade']['score'] == 1 and bool(best['hard_gate_passed'])
        failures = sorted({kind for a in attempts for kind in failure_types(a, run['execution_mode'])})
        counts.update(failures)
        fingerprints = [hashlib.sha256(a['candidate_code'].strip().encode()).hexdigest() for a in attempts]
        hidden = record.get('hidden')
        rows.append({
            'task_id': run['task_id'], 'run_id': run['run_id'], 'split': run['benchmark_split'],
            'mode': run['execution_mode'], 'first_pass': first_pass, 'final_pass': final_pass,
            'first_score': attempts[0]['functional_grade']['score'], 'final_score': best['functional_grade']['score'],
            'repaired': not first_pass and final_pass, 'attempts': len(attempts),
            'unchanged_candidates': len(fingerprints) - len(set(fingerprints)), 'failure_types': failures,
            'stop_reason': run['stop_reason'], 'model_calls': job['result']['model_calls'],
            'wall_clock_seconds': job['result']['wall_clock_seconds'],
            'compilation_seconds': sum(a['runtime_grade'].get('compilation_seconds', 0) for a in attempts),
            'execution_seconds': sum(a['runtime_grade'].get('total_duration_seconds', 0) for a in attempts),
            'hidden': hidden,
        })
    completed = [r for r in rows if 'infrastructure_error' not in r]
    repairs = [r for r in completed if not r['first_pass']]
    checked = [r for r in completed if r.get('hidden') and 'infrastructure_error' not in r['hidden']]
    repair_requests = [r for r in requests if 'you repair incorrect' in (r.get('system') or '').lower()]
    tokens = [r.get('usage', {}).get('prompt_eval_count') for r in repair_requests]
    return {
        'task_count': len(rows), 'completed_tasks': len(completed),
        'infrastructure_failures': len(rows) - len(completed),
        'first_passes': sum(r['first_pass'] for r in completed),
        'final_passes': sum(r['final_pass'] for r in completed),
        'repaired_tasks': sum(r['repaired'] for r in repairs),
        'repair_eligible_tasks': len(repairs),
        'hidden_evaluated_tasks': len(checked), 'hidden_passes': sum(r['hidden']['passed_all'] for r in checked),
        'hidden_infrastructure_failures': sum(bool(r.get('hidden', {}).get('infrastructure_error')) for r in completed if r.get('hidden')),
        'model_calls': sum(r['model_calls'] for r in completed),
        'unchanged_candidates': sum(r['unchanged_candidates'] for r in completed),
        'failure_counts_by_task': dict(counts),
        'repair_request_count': len(repair_requests),
        'max_repair_prompt_characters': max((r['prompt_characters'] for r in repair_requests), default=0),
        'max_repair_prompt_tokens': max((t for t in tokens if t is not None), default=None),
        'tasks': rows,
    }


def _validate_record(record, manifest, tasks):
    if set(record['task']) - {'task_id', 'title', 'split'}:
        raise ValueError('Task archive metadata must exclude catalog test fixtures.')
    if record.get('manifest') != manifest:
        raise ValueError('Run archive belongs to different evaluation settings.')
    task_id = record['task']['task_id']
    if task_id not in tasks:
        raise ValueError('Run task is absent from the pinned catalog.')
    run = record.get('run')
    if record['job']['status'] != 'completed':
        return
    if not run or not run.get('attempts'):
        raise ValueError('Completed job has no saved candidates.')
    expected = {'task_id': task_id, 'language': manifest['language'],
                'execution_mode': manifest['execution_mode'], 'seed': manifest['seed'],
                'config_name': manifest['config'], 'model_name': manifest['model'],
                'benchmark_split': tasks[task_id].split, 'test_source': 'trusted'}
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved run does not match the evaluation contract.')
    if run.get('attempt_budget') != manifest['max_attempts'] or len(run['attempts']) > manifest['max_attempts']:
        raise ValueError('Saved run exceeds or changes the pinned candidate budget.')
    if any(a['runtime_grade'].get('execution_environment') != manifest['toolchain'] for a in run['attempts']):
        raise ValueError('Saved candidate ran with a different toolchain.')
    best = next((a for a in run['attempts'] if a['attempt_number'] == run['best_attempt_number']), None)
    if best is None or best['candidate_code'] != record['job']['result']['code']:
        raise ValueError('Selected result does not match the frozen saved candidate.')


def _hidden_identity(code, manifest):
    return hashlib.sha256(json.dumps([code, manifest['catalog_sha256'], manifest['toolchain']], sort_keys=True).encode()).hexdigest()


def _read_requests(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    requests = []
    for index, line in enumerate(lines):
        try:
            requests.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed model worker can leave one incomplete trailing write.
            if index != len(lines) - 1:
                raise
            print('Ignoring incomplete final model-request log entry.', flush=True)
    return requests


def review_workspace(workspace, *, evaluate_hidden=False):
    workspace = Path(workspace)
    manifest = json.loads((workspace / 'manifest.json').read_text())
    dataset_path = PROJECT / 'data' / (manifest['catalog'] + '.json')
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != manifest['catalog_sha256']:
        raise ValueError('Catalog changed since evaluation; do not mix different test suites.')
    tasks = {t.task_id: t for t in load_benchmark(dataset_path).tasks}
    records = [json.loads(p.read_text()) for p in sorted((workspace / 'runs').glob('*.json')) if not p.name.endswith('.pending.json')]
    for record in records:
        _validate_record(record, manifest, tasks)
    if len({r['task']['task_id'] for r in records}) != len(records):
        raise ValueError('Duplicate task archives would bias evaluation counts.')
    if evaluate_hidden:
        environment = preflight(manifest['execution_mode'])
        if environment != manifest['toolchain']:
            raise ValueError('Toolchain changed since evaluation; hidden scoring requires the pinned environment.')
        for record in records:
            if record['job']['status'] != 'completed' or not record.get('run'):
                continue
            task = tasks[record['task']['task_id']]
            code = record['job']['result']['code']
            identity = _hidden_identity(code, manifest)
            hidden_path = workspace / 'runs' / (task.task_id + '.hidden')
            saved = json.loads(hidden_path.read_text()) if hidden_path.exists() else None
            if saved and saved.get('identity') == identity:
                record['hidden'] = saved
                continue
            try:
                result = verify(code, task.hidden_tests, language=task.language, execution_mode=task.execution_mode,
                                entry_point=task.entry_point, max_failures=0)
                runtime = grade_runtime_robustness(result)
                hidden = {'identity': identity, 'score': result.pass_rate, 'passed': result.passed, 'total': result.total,
                          'passed_all': result.passed == result.total and not runtime.critical_failure,
                          'compilation_seconds': runtime.compilation_seconds, 'execution_seconds': runtime.total_duration_seconds,
                          'execution_environment': runtime.execution_environment}
            except RuntimeError:
                hidden = {'identity': identity, 'infrastructure_error': 'Hidden execution infrastructure failed; inspect toolchain and trusted harness.'}
            hidden_path.write_text(json.dumps(hidden, indent=2))
            record['hidden'] = hidden
            print(f'{task.task_id}: hidden={hidden.get("score", "infrastructure error")}', flush=True)
    else:
        for record in records:
            path = workspace / 'runs' / (record['task']['task_id'] + '.hidden')
            if path.exists() and record['job']['status'] == 'completed':
                saved = json.loads(path.read_text())
                if saved.get('identity') == _hidden_identity(record['job']['result']['code'], manifest):
                    record['hidden'] = saved
    request_path = workspace / 'model_requests.jsonl'
    requests = _read_requests(request_path)
    summary = {'manifest': manifest, 'expected_tasks': len(tasks),
               'missing_tasks': sorted(set(tasks) - {r['task']['task_id'] for r in records}),
               **summarize(records, requests)}
    (workspace / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Scala browser evaluation', '',
             f'Mode: {manifest["execution_mode"]}; model: {manifest["model"]}; seed: {manifest["seed"]}; budget: {manifest["max_attempts"]}.', '',
             f'Completed: {summary["completed_tasks"]}/{summary["expected_tasks"]}. First passes: {summary["first_passes"]}; final passes: {summary["final_passes"]}; repaired: {summary["repaired_tasks"]}/{summary["repair_eligible_tasks"]}.',
             f'Hidden passes: {summary["hidden_passes"]}/{summary["hidden_evaluated_tasks"]}. Model calls: {summary["model_calls"]}. Unchanged candidates: {summary["unchanged_candidates"]}.', '',
             '| Task | First | Final | Hidden | Attempts | Failures |', '| --- | --- | --- | --- | --- | --- |']
    for row in summary['tasks']:
        if 'infrastructure_error' in row:
            lines.append(f'| {row["task_id"]} | — | infrastructure | — | — | job failed |')
        else:
            hidden = row.get('hidden') or {}
            lines.append(f'| {row["task_id"]} | {row["first_score"]:.2f} | {row["final_score"]:.2f} | {hidden.get("score", "pending")} | {row["attempts"]} | {", ".join(row["failure_types"]) or "none"} |')
    lines += ['', 'Visible tests drive generation and repair. Hidden evaluation runs only on the selected final code and never calls a model.',
              'These development catalogs provide limited correctness evidence. This is same-task repair evaluation, not a cross-task prompt promotion.']
    (workspace / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--evaluate-hidden', action='store_true')
    args = parser.parse_args()
    result = review_workspace(args.workspace, evaluate_hidden=args.evaluate_hidden)
    print(json.dumps({k: v for k, v in result.items() if k not in ('manifest', 'tasks')}, indent=2))
