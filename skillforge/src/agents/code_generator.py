from dataclasses import dataclass

from src.agents.code_prompts import execution_contract, generation_system, task_context
from src.evolution.prompt_registry import load_default_registry
from src.utils.parsing import strip_markdown_fence


@dataclass
class CodeCandidate:
    raw_code: str
    problem_type: str
    prompt_version: str = "unversioned"
    model_name: str = "unknown"
    generation_temperature: float | None = None
    repair_validation_error: str | None = None
    algorithm_plan: str | None = None


def generate_code(
    client,
    task_spec,
    temperature=0.2,
    prompt_registry=None,
    prompt_id=None,
    *,
    language="python",
    execution_mode="stdio",
    entry_point=None,
    algorithm_plan=None,
):
    """Generate code with the active or explicitly selected prompt version."""
    contract = execution_contract(language, execution_mode, entry_point)
    registry = prompt_registry or load_default_registry()
    prompt_version = (
        registry.get(prompt_id) if prompt_id is not None else registry.active_prompt
    )

    prompt = f"{task_context(task_spec)}\n\nExecution contract:\n{contract}"
    if algorithm_plan is not None:
        prompt += f"""

Reviewed algorithm plan (model-generated and fallible):
{algorithm_plan.text}

Implement the plan only where it satisfies the original problem. Check each state
update and invariant while coding. Respect the exact grammar and maximum bounds.
Return the complete source code for the execution contract above."""

    raw = client.generate(
        prompt,
        system=generation_system(prompt_version.prompt_text, contract),
        temperature=temperature,
        **({'max_output_tokens': 4096} if algorithm_plan is not None else {}),
    )
    code = strip_markdown_fence(raw)

    return CodeCandidate(
        raw_code=code,
        problem_type=task_spec.problem_type,
        prompt_version=prompt_version.prompt_id,
        model_name=getattr(client, "model", "unknown"),
        generation_temperature=temperature,
        algorithm_plan=algorithm_plan.text if algorithm_plan is not None else None,
    )
