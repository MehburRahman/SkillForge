"""Keep an evolution workspace on one language and execution contract."""
from src.experiments import validate_experiment
from src.benchmark.execution_mode import ExecutionMode


def validate_workspace_scope(language, mode, store=None):
    language, mode = str(language), str(mode)
    validate_experiment(language=language)
    ExecutionMode(mode)
    if mode == 'http' and language != 'scala':
        raise ValueError('HTTP evolution currently supports Scala only')
    if store is not None:
        if any(reasoning != 'direct' for reasoning in store.training_reasoning_modes()):
            raise ValueError('prompt evolution currently validates direct reasoning only; '
                             'keep plan-and-review training in a separate experiment workspace '
                             'until validation supports the same reasoning mode')
        existing = store.training_execution_scopes()
        if any(pair != (language, mode) for pair in existing):
            raise ValueError('training history belongs to a different language/execution mode; '
                             'use separate --history-db and --prompt-registry paths for this workspace')
    return language, mode


def validate_evolution_scope(tasks, store=None):
    if not tasks:
        raise ValueError('evolution requires validation tasks')
    scopes = {(str(t.language), str(t.execution_mode)) for t in tasks}
    if len(scopes) != 1:
        raise ValueError('use a separate evolution workspace per language and execution mode')
    return validate_workspace_scope(*next(iter(scopes)), store=store)
