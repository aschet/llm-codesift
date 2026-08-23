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

`run` performs every stage in order. With no `--models`, every model on the server is evaluated,
which is usually more than intended.

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

Every stage records per task and skips work already stored, so an interrupted run is restarted,
not repeated; `--redo` measures again regardless. Only one job touches the GPU at a time.

Running against a remote server works unchanged (`--host http://host:11434`). Local GPU details
are then omitted from the report.

## Finding Candidates

Screening costs GPU hours, so choose the shortlist rather than pulling at random.

```bash
codesift discover --coding --max-size-gb 24 --write-models candidates.txt
```

A candidate must be pullable rather than cloud-only, emit tool calls, advertise at least `--ctx`
tokens, fit the size range, and be recent.

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

Architecture matters more than size: a dense 24B and a 30B mixture of experts download to the
same few gigabytes and behave nothing alike once the weights exceed VRAM. An architecture is
marked `moe` only when its name establishes it, so the absence of the mark is not a claim of
dense — `ollama show` settles it after a pull.

## Regrading And Discarding

The reply is stored with each result, so a changed grader can be applied without touching the GPU.
Every ledger is copied to `.bak` first.

```bash
codesift regrade --apply             # re-grade stored replies
codesift prune --apply               # drop ruled-out models and their records
codesift prune --models a:1 --apply  # by name, whatever the verdict says
codesift prune --forget --apply      # restore everything discarded
```

Discarded names go to `results/discarded.txt`, which sweeps resolve through, so a discarded model
returns only when named explicitly.

## The Agent Stage

`codesift agent` drives opencode headlessly. opencode does not discover Ollama models, so they
must be declared in its configuration:

```bash
codesift sync-opencode --write    # applies, keeping a backup
```

Four tasks: a bug fix, a small feature, a refactor, and `ag_module`, which asks the model to
write one standard-library module and its own tests from a seeded `SPEC.md`. By default the stage
runs only models the screen called suitable or limited; `--select all` ignores the verdicts.

## Interpreting the Report

Models are labelled suitable, limited or unsuitable. Models that triage ruled out are named
separately with the gate they failed, since pruning a discarded model removes its measurements
and silence cannot distinguish a model that failed from one that was never run.

Unsuitable marks a failure that ends a model's usefulness: a tool call that could not be parsed
or was never emitted, context truncation, failed retrieval at depth, generation below 20 tokens
a second, or more than 300 seconds per task. Those models are excluded before any scoring,
because such failures cannot be traded against speed.

Limited marks a graded shortfall: under 85% on the hard set, over 120 seconds per task, or a
well-formed call to the wrong tool. A model emitting no parseable call cannot be driven at all,
while one calling the wrong tool gets another turn.

The recommendation is 70% quality and 30% speed, where speed is the fastest task time divided by
the model's own. Below a quality floor set 20 points under the best quality measured, speed earns
nothing and the composite is quality alone, so a model that fails its tasks quickly cannot
outrank one that produces working code. Its speed is still shown.

Task time is prefill plus the observed volume of output at the measured generation rate, for one
task from a cold context. Code tasks are scored per assertion, so satisfying four of five checks
scores 0.8; everything else is answered or not. A card also reports replies cut off at the output
budget, which covers reasoning as well as the answer — such a score is a floor, raised with
`--num-predict`.

Scoring is relative to the models given to it, so pass the intended shortlist with
`--models-file`; without one the report covers every model that has records.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

Standard library only, no server needed.

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
