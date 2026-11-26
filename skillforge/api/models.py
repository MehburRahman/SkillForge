from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.agents.llm_client import DEFAULT_MODEL, DEFAULT_CONTEXT_WINDOW


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
PromptID = Annotated[str, StringConstraints(pattern=r"^P\d+$")]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Budget = Annotated[int, Field(strict=True, ge=1, le=20)]
ContextWindow = Annotated[int, Field(strict=True, ge=1024, le=32768)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StdioTest(InputModel):
    input: str = Field(max_length=10000)
    expected_output: Any
    category: Literal["normal", "edge", "stress"] = "normal"


class FunctionTest(InputModel):
    assertion: str = Field(min_length=1, max_length=10000)
    category: Literal["normal", "edge", "stress"] = "normal"


class HTTPRequest(InputModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    path: str = Field(min_length=1, max_length=4096)
    body: str = Field(default="", max_length=1048576)
    headers: dict[str, str] = Field(default_factory=dict)
    expected_status: Annotated[int, Field(strict=True, ge=100, le=599)]
    expected_body: str | None = Field(max_length=1048576)
    expected_headers: dict[str, str] = Field(default_factory=dict)
    expected_body_one_of: list[str] | None = Field(default=None,min_length=1,max_length=16)
    parallel_group: Annotated[int, Field(strict=True, ge=0, le=16)] = 0


class HTTPTest(InputModel):
    requests: list[HTTPRequest] = Field(min_length=1, max_length=128)
    category: Literal["normal", "edge", "stress"] = "normal"
    clean_environment: bool = False
    port: Annotated[int, Field(strict=True, ge=1, le=65535)] | None = None
    requirements: list[Annotated[str, StringConstraints(min_length=1, max_length=500)]] = Field(default_factory=list, max_length=16)


class SolveRequest(InputModel):
    problem: str | None = Field(default=None, min_length=1, max_length=20000)
    task_id: Text | None = None
    catalog: Text = "problems"
    language: Literal["python", "javascript", "scala"] | None = None
    execution_mode: Literal["stdio", "function", "http"] | None = None
    entry_point: Text | None = None
    tests: list[StdioTest | FunctionTest | HTTPTest] | None = Field(default=None, min_length=1, max_length=100)
    reference_tests: list[StdioTest] | None = Field(default=None, min_length=1, max_length=100)
    model: Text = DEFAULT_MODEL
    context_window: ContextWindow = DEFAULT_CONTEXT_WINDOW
    reasoning_mode: Literal["direct", "plan_critique"] = "direct"
    test_model: Text | None = None
    test_generation_method: Literal['program', 'json'] = 'program'
    differential_cases: Annotated[int, Field(strict=True, ge=0, le=1000)] = 0
    judge_model: Text | None = None
    # 'passing' judges only attempts whose tests all pass, saving a model swap and
    # a judge call per failed attempt; 'all' matches benchmark/CLI behaviour.
    judge_scope: Literal['all', 'passing'] = 'passing'
    llm_judge: bool = False
    prompt_id: PromptID | None = None
    config: Literal["single_shot", "best_of_n", "scalar_repair", "full"] | None = None
    seed: Seed | None = None
    max_attempts: Budget | None = None

    @model_validator(mode="after")
    def one_problem_source(self):
        if self.differential_cases and (self.task_id is not None or self.tests is not None
                or self.execution_mode not in (None, 'stdio') or self.test_generation_method != 'program'):
            raise ValueError('differential testing requires generated stdio program tests')
        if (self.problem is None) == (self.task_id is None):
            raise ValueError("provide exactly one of problem or task_id")
        if self.problem is not None:
            if not self.problem.strip():
                raise ValueError("problem cannot be blank")
            if "catalog" in self.model_fields_set:
                raise ValueError("catalog is only used with task_id")
        elif self.tests is not None:
            raise ValueError("catalog tasks use their stored visible tests")
        if self.reference_tests is not None and (
            self.task_id is not None or self.tests is not None or self.execution_mode not in (None, 'stdio')
        ):
            raise ValueError('reference_tests require an ad hoc stdio task without supplied solution tests')
        return self


class EvolutionRequest(InputModel):
    catalog: Text = "mbpp"
    model: Text = DEFAULT_MODEL
    context_window: ContextWindow = DEFAULT_CONTEXT_WINDOW
    meta_model: Text | None = None
    judge_model: Text | None = None
    llm_judge: bool = False
    seed: Seed = 42
    max_attempts: Budget = 3
    proposal_attempts: Annotated[int, Field(strict=True, ge=1, le=5)] = 2
    max_observations: Annotated[int, Field(strict=True, ge=1, le=100)] = 50
    run_id: Text | None = None
    resume: bool = False
    retry_proposal: bool = False

    @model_validator(mode="after")
    def resume_has_id(self):
        if self.resume and self.run_id is None:
            raise ValueError("resume requires run_id")
        if self.retry_proposal and not self.resume:
            raise ValueError("retry_proposal requires resume and the original run_id")
        return self
