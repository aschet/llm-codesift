# llm-codesift

A screening harness for locally hosted coding models served by Ollama.

It decides which models deserve a deeper benchmark and excludes those that fail outright. It is
not a leaderboard: task counts are small, and when several models reach the maximum the harness
says so rather than inventing an order.

Every result is produced by executing something. Generated code is run against assertions, and
agent tasks are graded on the files left on disk. No model judges another model.

## What It Measures

| Stage | Question It Answers |
|---|---|
| `triage` | Can this model be ruled out cheaply, before anything expensive runs? |
| `screen` | Can it write code, repair code, obey output constraints, emit valid tool calls, and predict what a program prints? |
| `probe` | How long to the first token at a realistic context depth, and does it silently lose context? |
| `cache` | Is the prefill cost paid once per session or on every turn? |
| `agent` | Driving a real harness, does it finish a task and write a module from a spec? |
| `report` | All of the above, as one HTML page with a computed recommendation. |

Context truncation is checked, not assumed: Ollama discards prompt overflow without reporting it
and answers anyway, so the harness compares tokens processed against tokens submitted and
retrieves a fact planted at the start of the prompt. A model that truncates loses the beginning
of its context first, which in an agent harness is its own instructions.

Only one job touches the GPU at a time. Two models sharing a card thrash and corrupt each
other's timings, so every stage takes a lock before loading.

## Requirements

- Python 3.9 or newer, no third-party dependencies
- A reachable Ollama server
- opencode, for the `agent` stage only

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install .
```

## Quick Start

```bash
codesift run --models qwen3:14b granite4:8b gemma3:12b -o report.html
```

With no `--models`, every model on the server is evaluated, which is usually more than intended.
See [Finding Candidates](#finding-candidates) for choosing a shortlist.

## Usage

Every stage shares these options:

| Option | Meaning |
|---|---|
| `--host URL` | Ollama server; defaults to `$OLLAMA_HOST`, then `http://localhost:11434` |
| `--models A B C` | Models to evaluate; defaults to every model on the server |
| `--models-file PATH` | One model per line; blank lines and `#` comments ignored |
| `--results-dir DIR` | Where records are written; defaults to `results` |
| `--ctx N` | Context window for every request; defaults to `65536` |
| `--timeout SECONDS` | Per-request timeout; defaults to `2400` |

```bash
codesift triage  --models-file models.txt --apply     # reject cheaply, first
codesift screen  --models-file models.txt --taskset basic --runs 3
codesift screen  --models-file models.txt --taskset hard  --runs 3
codesift probe   --models-file models.txt --depth 48000
codesift cache   --models-file models.txt
codesift agent   --models-file models.txt
codesift report  -o report.html
```

`codesift run` does all of it in order. Every stage records per task and skips work already
stored, so an interrupted run is restarted, not repeated; `--redo` measures again regardless.

Running against a remote server works unchanged (`--host http://host:11434`). Local GPU details
are then omitted from the report, and the GPU lock is local to the machine running the harness.

## Finding Candidates

Screening costs GPU hours, so choose the shortlist rather than pulling at random.

```bash
codesift discover --coding --max-size-gb 24 --write-models candidates.txt
```

A candidate must be pullable rather than cloud-only, emit tool calls, advertise at least `--ctx`
tokens, fit the size range, and be recent. Tool calling has no flag to relax it: a harness
reaches the filesystem through tool calls or not at all.

| Option | Meaning |
|---|---|
| `--since WHEN` | Date, year, or months back; defaults to `18m` |
| `--max-size-gb` / `--min-size-gb` | Download size bounds; default `32` and `4` |
| `--min-context TOKENS` | Smallest advertised context; defaults to `--ctx` |
| `--coding` | Only models whose listing names programming work |
| `--match` / `--exclude REGEX` | Filter on name and description |
| `--sort date\|name`, `--limit N` | Ordering and length; default `date` and `20` |
| `--include-installed` | Also list what the server already holds |
| `--json`, `--write-models PATH`, `--refresh` | Machine output, model list, cache bypass |

Every column is copied from the library; nothing ranks models by likely ability, since a
description is written to sell. Architecture matters more than size: a dense 24B and a 30B
mixture of experts download to the same few gigabytes and behave nothing alike once the weights
exceed VRAM. An architecture is marked `moe` only when its name establishes it, so the absence
of the mark is not a claim of dense — `ollama show` settles it after a pull. Installed models
are hidden by manifest digest rather than name, so `gemma4:31b` is still offered when you hold
`gemma4:26b`.

## Triage

`codesift triage` asks the cheapest decisive question first and stops at the first answer that
ends the matter. The order comes from what the measurements cost and what they rejected:

| Gate | Cost | What it has caught |
|---|---|---|
| speed | ~10s | four models generating 6 to 14 tokens a second |
| tools | ~20s | one model that could not emit a parseable tool call |
| quality | ~70s | the bulk of the field, below 70% on the hard set |
| context | ~90s | one model that truncated at 64k and lost the needle |

No gate invents a threshold; each applies a rule the report already applies, so a model rejected
here is one the full run would have rejected anyway. The gates record into the screen's own
ledger as run 1, so a task graded during triage is not run again. `--apply` adds rejections to
`results/discarded.txt`, and `codesift run` triages first with `--apply`.

## Regrading And Discarding

A grading fix is worthless if benefiting from it means re-running every model. The reply is
stored with each result, so a changed grader can be applied to stored replies with no GPU:

```bash
codesift regrade --apply     # rewrite the records, originals kept as .bak
```

Only code tasks are re-graded, and a reply cut short by an older excerpt limit cannot be graded
at all — grading a truncated reply would fail code that continued past the cut. Those keep their
original verdict and are counted as unverifiable, so the count says how many need a re-run.

```bash
codesift prune --apply               # drop ruled-out models and their records
codesift prune --models a:1 --apply  # by name, whatever the verdict says
codesift prune --forget --apply      # restore everything discarded
```

Every ledger is copied to `.bak` first, and an unparseable line is kept rather than dropped.
Discarded names go to `results/discarded.txt`, which sweeps resolve through, so a discarded model
returns only when named explicitly. `--keep-records` writes the list without deleting anything.

## The Agent Stage

`codesift agent` drives opencode headlessly. opencode does not discover Ollama models, so they
must be declared in its configuration:

```bash
codesift sync-opencode --write    # applies, keeping a backup
```

Four tasks: a bug fix, a small feature, a refactor, and `ag_module`. The first three edit code
that already exists; the fourth asks whether a model can write ordinary things from a written
contract at all. A seeded `SPEC.md` describes one standard-library module keeping a list of
tasks, and the model writes it and its own tests. It is graded per function over sixteen checks
and verifies in under a second, because nothing outside the interpreter is involved. Only the
import can cascade, so a model that gets eight functions right scores eight, and a module
missing one function still imports and costs only the checks that needed it.

`tests/tasklist_reference/` holds a solution that must score all sixteen, and the suite damages
it in ways that have to cost specific points and no others — a grader nobody has run against a
working implementation measures its own bugs.

Results feed into the verdict only downward. A failure counts once it has survived a retry: the
stage is variable, and a task that failed once is reported without lowering anything. Failing
the one check everything rests on — the module importing — is not a shortfall but a refusal, and
is judged on a single attempt, since that is a property of the code rather than of the session.
Treating a completed task as a credit would rank a model above another for having been measured
at all, so nothing here raises a verdict.

By default the stage runs only models the screen called suitable or limited; `--select all`
ignores the verdicts. It prints its worst-case duration before starting, runs the harness in its
own process group and takes that group down on a timeout, and unloads the model when done. It
refuses to start if opencode is missing, cannot resolve the models, or denies a tool outright,
rather than producing failures that look like model problems.

The application task that used to live here is now [appsift](https://github.com/aschet/appsift),
its own tool and repository. Thirteen of its eighteen checks could not be attempted unless the
application started, so a model that mis-named one file scored what a model that wrote nothing
scored — too blunt to grade with. `ag_module` asks the same question without a web server.

## Interpreting the Report

Models are labelled suitable, limited or unsuitable.

Code tasks are scored per assertion, so a model that satisfies four of five checks scores 0.8
rather than nothing. Most failing answers are near misses — across one sweep the median
satisfied three quarters of what it was given — and scoring those the same as code that does not
run reports a difference that is not there. Everything else is answered or not, and scores one
or nothing. `passed` still means every check met, so the thresholds below are unchanged.

Unsuitable marks a failure that ends a model's usefulness: a tool call that could not be parsed
or was never emitted, context truncation, failed retrieval at depth, generation below 20 tokens
a second, or more than 300 seconds per task. Those models are excluded before any scoring,
because such failures cannot be traded against speed.

Limited marks a graded shortfall: under 85% on the hard set, over 120 seconds per task, or a
well-formed call to the wrong tool. That last distinction matters — a model emitting no parseable
call cannot be driven at all, while one calling the wrong tool gets another turn.

A card also reports replies cut off at the output budget. That budget covers reasoning as well as
the answer, so when it binds the grader sees an unfinished reply and records a failure
indistinguishable from a wrong one. Such a score is a floor; raise it with `--num-predict`.

Task time is prefill plus the observed volume of output at the measured generation rate, for one
task from a cold context. Both thresholds are absolute seconds, not multiples of the fastest
model in the run — an earlier version used the latter, so adding one quick model relabelled two
models scoring 100%.

A model too large for VRAM is caught by the generation floor rather than by either time
threshold. What governs generation is how much of a model moves per token, not how much is
resident: measured across ten models on one 12GB card, every model sat between 33% and 64%
resident, yet generation split 6 to 14 tokens a second for the dense ones against 44 to 60 for
the mixtures, with nothing in between. The floor is not fitted to that gap — below roughly 20
tokens a second a reply of this length takes over a minute to appear.

The recommendation is 60% quality and 40% speed, where speed is the fastest task time divided by
the model's own. Below a quality floor set 20 points under the best quality measured, speed earns
nothing and the composite is quality alone: a model that fails most of its tasks finishes them
quickly, and unconditional latency credit ranks it above models that produce working code. Its
speed is still shown, so the trade being refused stays visible.

Scoring is relative to the models given to it, so pass the intended shortlist with
`--models-file`; without one the report covers every model that has records.

## Adding Tasks

Task definitions live in `src/codesift/tasks/`. Before adding one, confirm that a correct
reference solution passes its assertions and, for edit and agent tasks, that the supplied
starting code genuinely fails — otherwise the task scores everything correct and measures
nothing. The agent tasks are the most informative and the fewest, so they are the first place
worth expanding.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

Standard library only, no server needed. Most of it guards the tasks: every reference solution
must satisfy its assertions, every seeded defect must genuinely fail, and every trace answer is
compared against what the snippet actually prints. A task that quietly accepts anything measures
nothing, and that failure is invisible during a run.

`tests/test_pipeline.py` is different in kind: it runs the whole pipeline for real, patching only
the model and the agent harness, against a temporary directory and a relative results path.
Seeding, grading, ledgers, resumption, verdicts and rendering are all genuine. Every other test
of `run` mocks each stage, which proves the stages are called and nothing about what happens
inside them — and the two worst faults this project has had were invisible to that.

One file uses pytest rather than unittest: `tests/tasklist_reference/tests/test_tasklist.py`,
the reference solution's own suite. It stands in for the tests a model writes for itself, and the
grader runs those by shelling out to pytest, so writing it that way exercises the real path.

## Known Limitations

- Task counts are small. Large differences and outright failures resolve reliably; small gaps
  between adjacent models do not.
- All tasks are Python, so other languages are not measured.
- Quality is measured with short prompts while latency is measured at depth. Whether a model
  reasons well at 40k tokens, as opposed to retrieving from there, is not covered.
- Sampling is deterministic at temperature 0 with a fixed seed, which aids reproducibility but
  examines one point of the output distribution.
- Results describe the quantised builds installed locally and will not match published figures,
  which are typically full precision.

## Licence

MIT.
