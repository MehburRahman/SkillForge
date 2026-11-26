"""Download pinned Scala dependencies once; subsequent runs compile offline."""
from pathlib import Path
import sys
import json
import tempfile
import argparse
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier.runners.scala import compile_sources, SCALA_VERSION, toolchain, READY_FILE, readiness_identity
from src.verifier.runners.base import Status


def initialize_workspace(mode, root):
    """Never overwrite an evolved registry when setup is repeated."""
    workspace = root / ('scala-' + mode)
    workspace.mkdir(parents=True, exist_ok=True)
    registry = workspace / 'prompts.json'
    if not registry.exists():
        with registry.open('x') as handle:
            json.dump({'schema_version': 1, 'prompts': [{
                'prompt_id': 'P0', 'parent_prompt_id': None, 'status': 'active',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'prompt_text': 'Choose a correct and efficient solution for the submitted requirements. '
                    'Output ONLY complete Scala solution code, following the authoritative execution contract. '
                    'No explanation or Markdown fences.',
                'change_reason': f'Initial Scala {mode} workspace strategy',
                'training_metrics': {}, 'validation_metrics': {},
            }]}, handle, indent=2)
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', choices=['stdio', 'function', 'http'],
                        help='Initialize independent experiment and prompt state for this contract.')
    args = parser.parse_args()
    toolchain()
    with tempfile.TemporaryDirectory(prefix='skillforge-scala-setup-') as directory:
        root = Path(directory)
        source = root / 'Main.scala'
        source.write_text('object Main { def main(args: Array[String]): Unit = println("ready") }')
        result = compile_sources([source], root / 'probe.jar', offline=False)
        if result.status != Status.PASS:
            raise RuntimeError(result.stderr)
        READY_FILE.parent.mkdir(parents=True, exist_ok=True)
        READY_FILE.write_text(json.dumps(readiness_identity()))
        print(f'Scala {SCALA_VERSION} ready for offline compilation.')
    if args.workspace:
        workspace = initialize_workspace(args.workspace, READY_FILE.parent / 'workspaces')
        print(f'Workspace: {workspace}\nUse its prompts.json and experiments.sqlite3 for CLI/API experiments.')


if __name__ == '__main__':
    main()
