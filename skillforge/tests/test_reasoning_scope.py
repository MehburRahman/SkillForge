import pytest

from src.evolution.scope import validate_workspace_scope
from src.history import ExperimentStore


def test_evolution_rejects_planned_training_instead_of_validating_a_different_pipeline(tmp_path):
    with ExperimentStore(tmp_path / 'history.db') as store:
        store.start_run('Sum integers.', 'P0', 'coder', 'trusted', False, benchmark_split='train',
                        language='python', reasoning_mode='plan_critique')
        with pytest.raises(ValueError, match='validates direct reasoning only'):
            validate_workspace_scope('python', 'stdio', store)


def test_planned_ad_hoc_runs_do_not_block_direct_training_workspace(tmp_path):
    with ExperimentStore(tmp_path / 'history.db') as store:
        store.start_run('Sum integers.', 'P0', 'coder', 'trusted', False, benchmark_split='adhoc',
                        language='python', reasoning_mode='plan_critique')
        assert validate_workspace_scope('python', 'stdio', store) == ('python', 'stdio')
