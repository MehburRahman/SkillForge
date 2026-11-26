"""Opt-in real Ollama API for browser smoke checks, with separate saved history."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
import hashlib
import json
import time
import httpx
from api.config import Settings, PROJECT
from api.main import create_app
from src.verifier.runners.scala import preflight
from tools.setup_scala import initialize_workspace
from src.agents.llm_client import LLMClient, build_clients
from src.agents.code_prompts import prompt_template_identity


CATALOGS = {'stdio': 'scala_algorithms', 'function': 'scala_functions', 'http': 'scala_http'}


class RecordedClient(LLMClient):
    """Archive visible model requests so failures and repair sizes can be audited."""
    def __init__(self, model, workspace):
        context = int(os.environ.get('SKILLFORGE_EVALUATION_CONTEXT', '4096'))
        super().__init__(model, context_window=context)
        self.workspace = workspace

    def _generate(self, prompt, **kwargs):
        started = time.monotonic()
        record = {'model': self.model, 'prompt': prompt, **kwargs,
                  'model_options': self.generation_options,
                  'prompt_characters': len(prompt) + len(kwargs.get('system') or '')}
        try:
            answer = super()._generate(prompt, **kwargs)
            record['response'] = answer
            record['usage'] = self.last_usage
            return answer
        except Exception as exc:
            record['error'] = str(exc)
            raise
        finally:
            record['seconds'] = time.monotonic() - started
            with (self.workspace / 'model_requests.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    if os.environ.get('SKILLFORGE_REAL_MODEL') != '1':
        raise SystemExit('Set SKILLFORGE_REAL_MODEL=1 to run real model calls.')
    mode = os.environ.get('SKILLFORGE_EVALUATION_MODE')
    if mode is not None and mode not in CATALOGS:
        raise SystemExit('Evaluation mode must be stdio, function, or http.')
    environment = preflight(mode or 'stdio')
    label = ('evaluation-' + mode + '-' + os.environ.get('SKILLFORGE_EVALUATION_ARM', 'baseline')) if mode else 'browser-smoke'
    if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in label):
        raise SystemExit('Workspace label must use lowercase letters, numbers, and hyphens.')
    workspace = initialize_workspace(label, PROJECT / 'data' / 'workspaces')
    catalogs = {CATALOGS[mode]: PROJECT / 'data' / (CATALOGS[mode] + '.json')} if mode else Settings().catalogs
    if mode:
        model = os.environ.get('SKILLFORGE_REAL_MODEL_NAME', 'qwen2.5-coder:7b')
        with httpx.Client(base_url='http://127.0.0.1:11434', timeout=10) as ollama:
            tags = ollama.get('/api/tags')
            tags.raise_for_status()
            match = next((item for item in tags.json()['models'] if item['name'] == model), None)
            if match is None:
                raise SystemExit(f'Install model {model} before evaluation; no models are downloaded by this runner.')
            version = ollama.get('/api/version')
            version.raise_for_status()
        manifest = {'protocol': 1, 'language': 'scala', 'execution_mode': mode,
                    'model': model, 'model_digest': match['digest'], 'ollama_version': version.json(),
                    'prompt_templates': prompt_template_identity(), 'toolchain': environment,
                    'context_window': int(os.environ.get('SKILLFORGE_EVALUATION_CONTEXT', '4096')),
                    'feedback_policy_sha256': hashlib.sha256((PROJECT / 'src/feedback/collector.py').read_bytes()).hexdigest(),
                    'model_transport_sha256': hashlib.sha256((PROJECT / 'src/agents/llm_client.py').read_bytes()).hexdigest(),
                    'catalog': CATALOGS[mode], 'catalog_sha256': hashlib.sha256(next(iter(catalogs.values())).read_bytes()).hexdigest(),
                    'seed': 42, 'config': 'full', 'max_attempts': 3}
        path = workspace / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise SystemExit('Evaluation settings changed; choose a new SKILLFORGE_EVALUATION_ARM.')
        path.write_text(json.dumps(manifest, indent=2))
    settings = Settings(history_db=workspace / 'experiments.sqlite3', registry=workspace / 'prompts.json',
                        jobs_db=workspace / 'api_jobs.sqlite3', cache_dir=workspace / 'task_spec_cache',
                        catalogs=catalogs, allowed_origins=('http://127.0.0.1:5174', 'http://localhost:5174'))
    def clients(**kwargs):
        generator, judge = build_clients(**kwargs)
        return RecordedClient(generator.model, workspace), RecordedClient(judge.model, workspace) if judge else None
    uvicorn.run(create_app(settings, client_factory=clients), host='127.0.0.1', port=18766)
