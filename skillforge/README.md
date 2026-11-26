# SkillForge

SkillForge generates Python, JavaScript, and Scala solutions, evaluates them, and
uses feedback to repair them. Ordinary Python problems live in **`data/problems.json`**. Both the
single-task runner and benchmark runner use this catalog.

For a detailed explanation of the code, read the
[step-by-step project guide](docs/project-guide/README.md). It follows one task
through the system, explains every application module and the tests, and includes
a [source navigation index](docs/project-guide/source-index/README.md).

The [full pipeline diagram](docs/pipeline.md) shows task solving, runtime execution,
repair strategies, benchmarking, prompt evolution, and persistence.

Run the commands below from `skillforge/`, with the repository's virtual
environment activated. Solving requires Ollama to be running with the selected
model available (default: `qwen2.5-coder:7b`). Listing tasks and running the
repository tests do not require a running model.

## Select a problem

List the available IDs without starting a model or writing experiment history:

```bash
python run_reinforcement.py --list-tasks
```

Solve one catalog problem:

```bash
python run_reinforcement.py --task-id arithmetic_sum_001
python run_reinforcement.py --task-id arithmetic_square_001
```

The runner loads the statement, visible tests, task ID, and split together.
Omitting both a statement and `--task-id` selects `arithmetic_sum_001` from the
catalog. This default now uses supplied tests instead of generating tests.
Single-task solving uses only visible tests; hidden tests are reserved for the
benchmark runner's final evaluation.

Experiment settings still apply, for example:

```bash
python run_reinforcement.py --task-id arithmetic_square_001 --config full --seed 42 --no-history
```

## Plan and review before coding

Use the same generator model to propose an algorithm, review that plan, and then
write code:

```bash
python run_reinforcement.py --task-id arithmetic_square_001 --reasoning-mode plan_critique --config full --seed 42
python run_benchmark.py --reasoning-mode plan_critique --config full --seed 42
```

In the browser, select **Reasoning mode → Plan and review**. Restart a backend
started before this setting was added. API requests use
`"reasoning_mode": "plan_critique"`; omitting it keeps the default `direct` mode.

Planning adds two model requests before code generation. When a candidate fails
a behavioral test, full-feedback repair first revises the algorithm using a
compact visible counterexample. Feedback distinguishes a successfully executed
program from a passed output comparison. Compiler repairs remain focused on
compiler diagnostics. Scalar repair receives the initial reviewed plan without
counterexamples; independent candidates share that initial plan.

These extra requests are counted and saved in the model audit log. The candidate
budget stays the same, so planned solving costs more model calls and time. Plans
remain fallible guidance: this workflow does not change model weights or
guarantee that a 7B model can solve a hard problem.

Run a controlled direct/planned comparison on a frozen statement and supplied
tests, optionally with withheld tests used only for final scoring:

```bash
python tools/compare_reasoning.py --problem-file path/to/problem.txt --tests path/to/visible.json --hidden-tests path/to/hidden.json --language scala --model qwen2.5-coder:7b --seed 42 --max-attempts 3 --context-window 8192 --output-dir path/to/new-comparison
```

Both arms use the same model, seed, context, prompt registry, and candidate
budget, with no model judge. The output records first/best visible passes,
withheld passes, model calls, tokens, time, source, history, and a settings
manifest with the model weights' digest and Scala toolchain. Withheld execution
time is recorded separately. Use `--first-mode plan_critique` in another output
directory to assess execution order effects. Both arms use the supplied tests;
this comparison measures coding and repair, while generated-test quality needs
separate evaluation.
Compare correctness alongside the extra cost; a single task and seed do not
establish general improvement.

Keep planned training history separate from direct training history. Prompt
evolution currently validates direct mode and rejects planned training evidence
in that workspace. Planned ad hoc solves can still use the normal history.

## Run the benchmark

Evaluate all catalog problems, or filter by their stored split or category:

```bash
python run_benchmark.py
python run_benchmark.py --split train --config full --seed 42
python run_benchmark.py --category arithmetic
```

The benchmark selects and repairs code using visible tests, then evaluates the
selected code on hidden tests. Hidden failures do not trigger further repair.

Both runners can also read another catalog using the same JSON format:

```bash
python run_reinforcement.py --catalog path/to/problems.json --list-tasks
python run_reinforcement.py --catalog path/to/problems.json --task-id your_task_id
python run_benchmark.py path/to/problems.json
```

## Add a problem

Add an object to the `tasks` array in `data/problems.json`. Each object needs:

| Field | Contents |
| --- | --- |
| `task_id` | A unique, stable ID used by the CLI and history |
| `title` | Short human-readable name |
| `problem_statement` | Complete problem, including input/output requirements |
| `category` | A supported category such as `arithmetic`, `array`, or `sorting` |
| `difficulty` | `easy`, `medium`, or `hard` |
| `split` | `train`, `validation`, or `hidden` |
| `visible_tests` | Nonempty list of tests available during repair |
| `hidden_tests` | Nonempty list of tests for final benchmark evaluation |

Each test has exact standard-input text in `input`, an `expected_output`, and a
`category` of `normal`, `edge`, or `stress`. Test inputs must be unique within
each suite and must not overlap between visible and hidden suites. Check the
expected answers yourself; the loader validates structure, not mathematical
correctness. The task's `split` and its visible/hidden suites are separate ideas:
every task has both suites, including training tasks.

## Ad hoc problems

You can still supply a temporary problem directly. Without `--tests`, SkillForge
asks the model to generate its test suite:

Keep the candidate generator separate from test generation by passing
`--model qwen2.5-coder:7b --test-model <installed-model>` to `run_reinforcement.py`.
The optional `test_model` API field and **Test generation model** UI field select
the model used for test construction, blind review, executable references and their
repairs. Omitting it uses the candidate model. The judge remains a separate setting.
Choose a test model based on measured task performance; a larger model does not
automatically provide correct tests. Test-model requests are audited under their own
role, included in the total call count, and recorded with the run's model identity.
Test-model requests (including reuse of the candidate model) default to a 4,096-token output budget, leaving the
candidate generator's output settings unchanged. This bounds generated output;
it does not guarantee a particular inference duration on local hardware.

Task analysis uses a 2,048-token output budget and limits the candidate-budget
explanation to 240 characters in both the schema and host validation. Failed
responses that reach the output limit are regenerated from the statement without
copying a repeating explanation into the next request. The host never adds
missing JSON delimiters or invents analysis fields. Stdio examples use exact
input/output text; function examples can retain native JSON values.
Native Ollama requests also use a
300-second read timeout and a connection timeout of at most 10 seconds, so a model
that does not return cannot hold the worker indefinitely. Configure the read
timeout before starting the backend with `SKILLFORGE_LLM_TIMEOUT_SECONDS`, or use
`LLMClient(request_timeout=...)` programmatically. Timeouts retain their diagnostic
in the job and model audit; the worker can then process the next queued job.
These settings apply to new requests after restarting the backend. They cannot
interrupt an existing request made by the previous process.

Official or otherwise trusted tests can be supplied using `--tests <JSON_FILE>`,
the API's `tests` field, or the UI's test field. They bypass all test-model calls;
catalog problems already use their stored visible tests. SkillForge does not
invent official tests or add task-specific answers when none have been supplied.

The CLI and API default to `--test-generation-method program` /
`"test_generation_method": "program"`. The model writes a bounded Python
`build_inputs()` function that returns **inputs and categories only** as a list
of dictionaries. The host validates the returned data and serializes it; builders
must not print JSON or read stdin. A separate request plans and reviews
a simple small-input reference algorithm from the original statement, then
generates `solve_input(input_text: str) -> str`. Builders and references are
requested as plain Python source to avoid nested JSON/source escaping. The
builder receives the reviewed model plan to help preserve the input grammar.
Its total output is bounded to 16 KiB with small workload targets, leaving room
for blind review in a local model's context. These targets never override the
problem's legal input bounds. Maximum-size performance needs separate verification.
The blind reviewer first validates inputs and tries to derive answers without
seeing reference code or answers. Executing the separate reference supplies
answers, which must agree with every confident blind derivation. If a small-case
answer remains uncertain, the reference must pass independently verified cases
before its outputs can grade the candidate. Supply these through the CLI's
`--reference-tests`, the API's `reference_tests`, or the UI's **Verified reference
cases** field. Grounded statement examples can also provide reference checks.
Additional reference-case inputs and answers remain outside model prompts,
and a disagreement stops validation without repairing toward their answers.
Each repaired reference is checked again. Passing those cases is evidence for
the reference, not a proof that it handles every legal input. Valid stress inputs
may use executable derivation when textual tracing is impractical.
This separates input construction from solving, and avoids generating large
literal test bodies in model responses. Every input still has to pass validation.
The plans and reference are model-generated and fallible; they do not establish
maximum-size candidate performance or independently trusted correctness.
`--test-generation-method json` retains direct input/answer generation and its
bounded program fallback. Programmatic `generate_tests` keeps its legacy JSON default.

Before candidate evaluation, a separate model request checks each generated input
against the original problem and derives its output without seeing the proposed
answer. If the reviewer cannot derive the answers, SkillForge asks the model for
a separate Python reference function using only the original question. It validates
inputs and computes outputs through actual execution, without seeing candidate
code or proposed answers. The model defines `solve_input(input_text: str) -> str` and returns the
complete task output as a string, raising the host-provided `InvalidInput` exception
for invalid input. The application handles stdin, validity metadata and JSON
serialization, and checks Python syntax and the function interface before execution.
The input contract also rejects global stdin reads (including common aliases) and
requires the function to consume or validate the supplied input string. It does
not infer a task algorithm or change the required output.
Runtime and return-format repairs use validated line-range edits bound to the
reference source's SHA-256, with non-overlapping ranges. Replacement blocks are
relative to the original block's indentation and retain their nested indentation.
Repaired source is checked before it replaces the accepted reference. Syntax,
or function-signature failures receive a fresh function. Syntactically valid
input-contract violations may receive bounded line edits, but never execute until
all checks pass. At most three reference generation/repair calls are allowed per
solve (in addition to the two program-mode planning calls), with the final call reserved for a fresh
function after an invalid edit or an unsuccessful runtime repair;
a working reference is reused across regenerated suites. Invalid inputs
or output disagreements regenerate tests, never select a more agreeable reference.
Reference crashes, timeouts, or invalid result formats remain validation failures.
An exhausted, broken reference stops validation immediately rather than
regenerating suites that would reuse the same broken program.
Invalid inputs, unresolved answers, disagreement, and inappropriate case
categories cause bounded test regeneration. After three rejected suites, the solve
stops before evaluating candidates. With generated tests, the candidate is drafted
before test generation starts. Failed or uncertain verification preserves that
code as an **unverified draft**, with no evaluation, reward or training-history
entry. The API job result and replayable events save the code and validation
failure. The browser shows the draft during verification and when the job is
reopened, including interrupted jobs. The CLI prints the draft without a reward.
Review requests, responses, and regeneration
calls and reference programs are saved in the model audit log. Reference execution
results appear in progress events. This is model-based review; agreement does
not make generated tests independently trusted. Extracted TaskSpec examples must
quote their original source. After bounded analysis repairs, unsupported optional
operations and examples are discarded together with grounding warnings; malformed
required fields still stop analysis. Discarded example values stay out of coding
and judging prompts, while the raw model responses remain in the audit log.
When the statement omits input/output encoding, the analysis model chooses an
explicit format. That same format is passed to test construction, blind review,
references, algorithm planning, candidate generation and all candidate repairs, while the original
statement remains authoritative. No task-specific format or algorithm is inserted
by the application. The API result includes the chosen input/output formats.
Native stdio analysis separates task requirements from a focused delivery request
with a 1,024-token output budget. Each stdin/stdout contract specifies serialization,
layout, whitespace and empty-value handling. The model distinguishes explicitly
stated delivery from its own choices; stated rules require a verbatim source quote,
and selected rules cannot claim a quote. Incomplete contracts and invented quotes
receive bounded fresh retries. Function and HTTP analysis retain their execution
contracts, and older custom clients keep the combined analysis interface.

After two rejected suites (such as duplicate inputs, malformed JSON, or wrong answers), the
next existing generation attempt asks for a Python test-building program. It runs
with a five-second deadline and a 256 KiB output limit to materialize unique inputs,
derive record counts, and compute proposed answers. Its output must pass the same
suite validation and blind review; executing a builder does not approve its answers
or increase the three-attempt budget. Builder source is saved in the model audit
log, and execution results appear in progress events.
Structured review responses are limited to exactly one entry per test and concise
explanations, preventing an unbounded stream of repeated review objects.
Native reviewers state the derivation before reporting whether the input obeys
stated input rules and whether an answer was derived. A definite empty or negative
answer can be derived just as an affirmative answer can. They classify the actual
workload as normal, edge or stress; mismatches still reject the suite. These fields
are normalized into the existing internal review records without changing answers,
accepting uncertain derivations or relaxing category checks.

```bash
python run_reinforcement.py "Read an integer and print its cube."
python run_reinforcement.py "Read an integer and print its cube." --tests path/to/cube_tests.json
```

For an ad hoc problem, `--history-task-id cube_demo` supplies a history label and
`--benchmark-split train` supplies an optional split (default: `adhoc`).
`--task-id` now selects a catalog entry; it is no longer a free-form history label.
Catalog selection cannot be mixed with a custom statement, `--tests`, or metadata
overrides, so the statement and tests cannot accidentally refer to different tasks.

## Judge diagnostic

`examples/judge_sanity_cases.json` remains separate. It contains known correct and
deliberately broken programs to check the judge, rather than tasks for the
generator to solve.

```bash
python run_judge_sanity.py --verify-only
python run_judge_sanity.py
```

## Migration from the example files

The old `benchmark/sample_benchmark.json`, `examples/addition_tests.json`, and
`examples/square_tests.json` have been consolidated into `data/problems.json`.
Use the catalog task IDs instead of the old file paths. Existing benchmark
hidden cases remain hidden; additional addition cases were merged, and square
cases were split into visible and hidden suites. The catalog now contains four
tasks.

Because the dataset contents changed, existing benchmark sweeps cannot be resumed
against the new catalog. Start a new run ID, or provide the exact original dataset
to resume an old sweep. Existing history records are retained.

## Repository tests

```bash
python -m pytest tests -q
```

## Repair diagnostics and prompt evolution

Compilation failures now use a focused repair request containing the original
requirements, previous source, and distinct compiler diagnostics. Behavioral
tests have not run at that point, so SkillForge skips the semantic judge and
does not infer an algorithm failure from the zero pass count. After compilation,
the normal functional and runtime feedback applies. The selected versioned
strategy is supplied to both initial generation and repair.

Every candidate still consumes its configured generation budget. Repeated source
is marked with its original attempt number in live progress and saved history.
An exact repeat of a compilation failure reuses that run's compiler evidence;
it records zero additional compilation time. Runtime results are executed again,
since identical code can still have nondeterministic behavior. Adaptive runs keep
their existing bounded repeat-stop policy. Fixed-budget runs receive repetition
feedback without extra uncounted model calls.

When a full-feedback compiler repair repeats an earlier candidate, or consecutive
evaluated candidates fail compilation, the next allowed repair request proposes
source edits in JSON. The application validates
exact, unique, non-overlapping anchors against the previous source before applying
any edits. Missing anchors, unchanged replacements, malformed responses, and empty
programs are rejected and their reasons are recorded in progress and feedback.
The resulting candidate still has to compile and pass the same frozen tests.
Native compiler, runtime and wrong-output repairs use source-hash-bound line
ranges with relative indentation, including the first compiler repair. Repeated
repairs from structured custom clients use validated anchors;
older generate-only runtime clients retain their full-source interface.
Scala feedback recognizes line readers that split a single consumed line on
newlines and then crash while indexing missing records. It directs the model to
repair the reader while preserving spaces and empty records. Recognized parsing
failures skip the optional algorithm-redesign call. This mechanism uses actual
failure evidence for any supported language; it contains no problem-specific
source replacements and stays within the existing candidate and repeat budgets.

The solve API, browser, and command-line runners use an explicit 8192-token model
context by default. Change it with `context_window` in API requests or
`--context-window` on the CLI (1024–32768). Runs with history enabled save model
requests, responses, options, seeds, usage, finish reasons, and errors immediately,
including analysis and failed calls. Open **Runs → Model requests** to inspect
them, or request `/model-requests/{request_log_id}`. The run contains its
`request_log_id`; the early `request_logging` job event provides it when a model
fails before run creation. Meta-agent calls use the evolution round ID. Hidden
test execution does not enter these logs or model prompts. Stored prompts and
code remain local in the configured history database.

The production judge rejects copied placeholder responses and `NONE` categories
that contradict deterministic failures. Exhausted judge retries are recorded as
unavailable, not as a numerical zero. The separate judge-sanity diagnostic records
raw contradictions to measure missed defects; its saved configuration identifies
that diagnostic policy.

**Evolution** in the browser shows the active prompt, unused completed training
runs, recurring failure categories, and the blocking reason. Checking readiness
is read-only. Ad hoc solves do not count as training; solving does not automatically
start a prompt proposal. Keep each language/execution mode in a separate workspace
and use independently checked training and validation tasks. The existing trigger
still requires 20 new training runs and three categories recurring in three runs
each. The browser's **Propose and validate** action uses seed 42 and the same
three-candidate budget for both validation arms; the existing promotion criteria
remain in force.

The meta-agent requests schema-constrained proposal JSON and preserves the coding
contract inside the proposed instructions. Rejected proposal responses and exact
validation errors are saved with the evolution round. A newly recorded
`proposal_validation` failure can be retried **once**, explicitly, using its
original frozen training evidence and unchanged experiment settings:

```bash
python run_evolution.py data/scala_http.json \
  --prompt-registry data/workspaces/scala-http/prompts.json \
  --history-db data/workspaces/scala-http/experiments.sqlite3 \
  --run-id YOUR_ROUND_ID --resume --retry-proposal
```

Use the same model, context, seed, catalog, and budgets as the original round.
API callers use `resume: true`, `retry_proposal: true`, and the original `run_id`.
The retry is audited and its calls are added to the round's cost; it still must
pass paired validation before promotion. Ordinary resume never regenerates a
failed proposal. Historical failures without a recorded failure classification
cannot use this recovery path. Existing history and prompt versions are preserved;
changed prompt/evaluation policies require new experiment runs.

## Scala execution and experiments

Scala support includes programs (`object Main` reading stdin), native function
tests (`object Solution` with the task's method signature), and HTTP services
(`object Main`, JDK `HttpServer`). Tasks use Scala/JDK standard libraries only.
The submitted problem and execution contract are authoritative; model-generated
analysis and examples can be wrong and must be checked against that specification.
The model selects its algorithm from the submitted requirements. Solutions and
application routes are never built into the execution tools.

The pinned toolchain is **Scala CLI 1.18.0, Scala 3.3.8, and JDK 21.0.13**.
Install those Scala CLI/JDK releases, put `scala-cli` on PATH, and set
`SKILLFORGE_JAVA_HOME` to your JDK installation. Setup checks these versions
and downloads the Scala compiler and packaging dependencies. Model runs compile
offline and fail preflight before model calls if the toolchain/cache is missing.

Initialize a separate experiment workspace for each execution contract:

```bash
python tools/setup_scala.py --workspace stdio
python tools/setup_scala.py --workspace function
python tools/setup_scala.py --workspace http
```

Setup preserves existing prompt registries. Local workspace state, Scala caches,
temporary compilation products, and readiness records are excluded from Git.
Compilation uses a temporary directory, runs once per candidate test batch, and
has a separate 180-second timeout. Each algorithm case runs in a fresh JVM with
a 256 MiB heap. Compilation and execution timings are recorded separately.
Function assertions compile together in a trusted harness; malformed harness
source is an infrastructure error. Incorrect interfaces and candidate compiler
errors become repair evidence. Compiler timeouts, runtime crashes, incorrect
answers, and runtime timeouts remain distinguishable in the saved evidence.

Three development catalogs are included:

| Catalog | Problems | Contract |
| --- | --- | --- |
| `data/scala_algorithms.json` | Weighted paths, coin counts, intervals, rectangles, DAG paths, range updates | `stdio` |
| `data/scala_functions.json` | Four-string subsequences, edit distance, balanced partitions | `function` |
| `data/scala_http.json` | Text store, counter, statistics, edit distance, queue, scheduling | `http` |

Their expected answers are checked independently in `tests/test_scala_catalogs.py`.
These small catalogs verify the tools; a larger set of independently verified,
unseen tasks is needed to establish general improvement.

```bash
python run_reinforcement.py --catalog data/scala_algorithms.json --list-tasks
python run_reinforcement.py --catalog data/scala_algorithms.json --task-id scala_shortest_paths_train --prompt-registry data/workspaces/scala-stdio/prompts.json --history-db data/workspaces/scala-stdio/experiments.sqlite3 --config full --seed 42
python run_benchmark.py data/scala_algorithms.json --split train --prompt-registry data/workspaces/scala-stdio/prompts.json --history-db data/workspaces/scala-stdio/experiments.sqlite3 --config full --seed 42
python run_evolution.py data/scala_algorithms.json --prompt-registry data/workspaces/scala-stdio/prompts.json --history-db data/workspaces/scala-stdio/experiments.sqlite3 --seed 42
```

Use the function/HTTP catalog and corresponding workspace paths for those modes.
The evolution trigger requires at least 20 new completed training runs and at
least three failure categories recurring in three runs each. It may correctly
decline to propose a prompt. Paired validation uses identical tasks, toolchains,
seeds, task analyses, and attempt budgets. Hidden cases and diagnostics do not
enter repair or evolution prompts. Promotion, rejection, recovery, and rollback
use the existing workflow. Resume rejects changed toolchains, execution contracts,
or test suites. Unsupported Scala constraint checks are reported as unverified.

HTTP execution starts a fresh server per scenario and sends its requests in
order. Tests specify methods, paths, UTF-8 bodies, statuses, response bodies, and
optional headers. The server binds localhost port zero and publishes the chosen
port through `SKILLFORGE_PORT_FILE` after startup. Startup and each complete
request have deadlines; cleanup kills the server process group.

Build the isolated HTTP runner during setup, with Docker running:

```bash
docker build -t skillforge-scala-http:1 -f tools/scala-http.Dockerfile .
```

The base image pins Temurin JRE 21.0.13+9 by verified registry digest; the built
image's exact image ID and actual runtime version are
recorded, and experiments execute that image ID. You can pin a base image digest
with `--build-arg JRE_IMAGE=eclipse-temurin@sha256:YOUR_VERIFIED_DIGEST`.
Containers have no external network, published ports, writable root filesystem,
or extra capabilities. For local development, explicitly set
`SKILLFORGE_SCALA_HTTP_LOCAL=1`; that runs generated servers on the host without
container isolation. Python/JavaScript and Scala algorithm execution also run
local processes and are intended for a trusted local development environment.

Run the browser against the same dedicated workspace as the CLI:

```bash
python run_api.py --prompt-registry data/workspaces/scala-stdio/prompts.json --history-db data/workspaces/scala-stdio/experiments.sqlite3 --jobs-db data/workspaces/scala-stdio/api_jobs.sqlite3 --cache-dir data/workspaces/scala-stdio/task_spec_cache
cd frontend
npm run dev
```

If API startup reports that another worker owns the job database, stop the
existing API server with Ctrl+C in its terminal. To find a background server,
run `lsof -nP data/api_jobs.sqlite3.lock` from `skillforge/` and send that
process `kill -TERM PID`. Keep the database and lock file in place; stopping
the owning process releases the lock. New servers record their PID in the
lock file and include it in the startup error.

The browser offers Scala and all three catalogs, displays compilation progress
and compiler evidence, and saves toolchain/timing information with each attempt.
Select the matching workspace before collecting evolution evidence. Ad hoc
function and HTTP problems require trusted assertions/request scenarios. Ad hoc
programs can use supplied tests or model-generated tests; passing generated tests
provides weaker evidence of correctness.

Local HTTP integration and frontend verification:

```bash
python -m pytest tests/test_scala_execution.py tests/test_http_execution.py tests/test_scala_catalogs.py -q
SKILLFORGE_TEST_DOCKER=1 python -m pytest tests/test_http_execution.py -q
cd frontend
npm test
npm run build
npm run test:browser
```

The Scala tests require the prepared toolchain, and HTTP/browser checks need
permission to bind local ports. The regular tests use deterministic model
fixtures. A real-model browser smoke test is available with
`SKILLFORGE_REAL_MODEL=1 npm run test:browser -- --config playwright.real.config.ts`;
start Ollama with the configured model installed first. Its results are saved in
`data/workspaces/scala-browser-smoke/`. Same-task repair and validated cross-task
prompt evolution are separate measurements; neither guarantees eventual success.

### Reproducible real-model catalog evaluation

The catalog browser runner submits every selected task through the production
UI/API and saves its history, screenshot, and visible model requests. It uses
`qwen2.5-coder:7b`, seed 42, the `full` strategy, at most three candidates, and
an explicit 4096-token model context. Ollama and the prepared Scala toolchain
must already be available. HTTP evaluation uses the prepared Docker image.

Run each contract sequentially from `frontend/`:

```bash
SKILLFORGE_REAL_MODEL=1 SKILLFORGE_EVALUATION_MODE=stdio SKILLFORGE_EVALUATION_ARM=requirements-fixed npm run test:browser -- --config playwright.real.config.ts catalog.spec.ts
SKILLFORGE_REAL_MODEL=1 SKILLFORGE_EVALUATION_MODE=function SKILLFORGE_EVALUATION_ARM=requirements-fixed npm run test:browser -- --config playwright.real.config.ts catalog.spec.ts
SKILLFORGE_REAL_MODEL=1 SKILLFORGE_EVALUATION_MODE=http SKILLFORGE_EVALUATION_ARM=requirements-fixed npm run test:browser -- --config playwright.real.config.ts catalog.spec.ts
```

Use a new arm label for a different implementation or configuration.
`SKILLFORGE_REAL_MODEL_NAME` and `SKILLFORGE_EVALUATION_CONTEXT` override the
model/context; `SKILLFORGE_EVALUATION_TASK_IDS` selects comma-separated IDs.
Each arm records the model digest, Ollama version, prompt/feedback fingerprints,
catalog checksum, toolchain, and run budget in an immutable manifest. Resuming
rejects incompatible settings, skips completed tasks, and reconnects to pending
jobs without submitting duplicate model work. Failed or interrupted jobs are
preserved; retry them under a new label.

From the project directory, score hidden tests on the frozen selected code:

```bash
python tools/review_scala_evaluation.py data/workspaces/scala-evaluation-stdio-requirements-fixed --evaluate-hidden
python tools/review_scala_evaluation.py data/workspaces/scala-evaluation-function-requirements-fixed --evaluate-hidden
python tools/review_scala_evaluation.py data/workspaces/scala-evaluation-http-requirements-fixed --evaluate-hidden
```

The review tool makes no model calls and saves hidden aggregate scores without
case inputs, outputs, or diagnostics. It rejects mismatched catalogs, toolchains,
and run provenance. Each workspace receives `summary.json` and `report.md` with
first/final/hidden passes, repair success, repeated candidates, failure types,
model usage, and separate compilation/execution timings. A completed browser
test means the workflow and saved history worked; the solution's correctness is
reported separately. Results from the development catalogs measure same-task
repair and do not establish cross-task prompt promotion.

The completed baseline, feedback pilot, and final evaluation are documented in
[Scala evaluation results](docs/scala-evaluation.md). The final implementation
passed visible tests on 7/15 tasks and all hidden tests on 4/15 tasks with
`qwen2.5-coder:7b`; this evaluation did not establish improved model performance.
