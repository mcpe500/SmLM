# model-compressor autoresearch program

This document defines how an autonomous coding/research agent should operate in this repository.

The mission is to turn an existing small language model (e.g. GPT-2, SmolLM2-135M, or another CPU-manageable transformer) into a **smaller, faster, lower-memory deployable model**, and to produce a **CPU inference engine** that runs the compressed artifact.

This repository is **CPU-first**, assumes **limited RAM**, and may use a **24/7 worker** for long-running jobs. The agent must optimize for practical wins, not clever research tricks that are hard to deploy.

## Two Implementation Tracks

### Track A: Pure C++ (Resource-Constrained Environments)

For Termux/Android or environments without PyTorch:

- **No Python ML dependencies** - pure C++17 inference engine
- **Binary model format (.slm)** - simple, cross-platform, little-endian
- **Minimal RAM usage** - 10-500MB depending on model size
- **Direct compilation** - `make` builds everything
- **Test model generator** - no PyTorch needed for testing

Use this track when:
- Running on Termux/Android
- RAM < 4GB
- No PyTorch available
- Need minimal dependencies

### Track B: Python + PyTorch (Full Pipeline)

For servers/workstations with PyTorch support:

- **Full training pipeline** - distillation, pruning, quantization
- **HuggingFace integration** - load/export any transformer model
- **ONNX export** - industry-standard runtime format
- **Comprehensive benchmarks** - quality, latency, memory

Use this track when:
- Training student models
- Export from HuggingFace
- Full experimentation pipeline
- Research and development

---

## Core objective

Given a teacher model, the system should be able to:

1. Create a compressed student model using practical methods.
2. Reduce memory usage.
3. Improve CPU inference speed.
4. Export the compressed result into a runtime-friendly format.
5. Run the compressed artifact through a CPU engine.
6. Benchmark the original vs compressed result.

The preferred end state is:

- a dense or semi-practical compressed student model,
- optionally quantized,
- exported into a CPU-friendly runtime format,
- served by a C++ inference engine,
- with benchmark evidence showing whether the change helped.

---

## Research philosophy

This project values **deployable compression**, not just parameter deletion on paper.

Prefer, in this order:

1. **Dense student shrinkage + distillation**
2. **Structured shrinking** (fewer layers, narrower width, smaller FFN, fewer heads where sensible)
3. **Post-training quantization or runtime quantization flow**
4. **Graph/runtime optimization**
5. **Structured pruning only when it leads to practical CPU wins**

Avoid relying primarily on:

- unstructured sparsity,
- GPU-only kernels,
- methods that require large VRAM,
- methods that make the checkpoint look smaller but do not improve real CPU latency,
- complex research additions that break export or deployment.

Rule of thumb:

- If a method makes the model smaller **and** easier to run on CPU, it is good.
- If a method makes a paper metric look impressive but complicates export and runtime, it is probably bad.

---

## Hardware assumptions

The agent must assume:

- CPU only
- RAM is limited
- long jobs can run continuously on a worker
- wall-clock time matters less than robustness and reproducibility

Therefore:

- prioritize resumable jobs,
- keep batch sizes modest,
- prefer gradient accumulation over large batches,
- write checkpoints often,
- avoid reading huge datasets into memory,
- avoid keeping multiple large model copies live unless necessary.

---

## Success criteria

A change is successful only if it helps the actual mission.

### Primary success criteria

1. **Quality retention**: the compressed student must retain acceptable quality relative to the teacher on the chosen evaluation.
2. **Inference speed**: the compressed artifact must improve CPU latency and/or throughput.
3. **Memory reduction**: peak RAM or model footprint must go down meaningfully.
4. **Deployability**: the result must export and run in the CPU engine.

### Secondary success criteria

- smaller on-disk artifact,
- cleaner code,
- fewer runtime dependencies,
- simpler pipeline,
- better job resumability,
- easier benchmarking.

### Failure cases

A run is a failure if any of these happen:

- training crashes and cannot be recovered quickly,
- export fails,
- engine cannot load the result,
- latency gets worse with no compensating quality win,
- quality collapses beyond the allowed threshold,
- RAM usage becomes unacceptable for the target machine.

---

## Optimization priorities

Always optimize in this order unless explicitly overridden by the human:

1. correctness
2. reproducibility
3. deployability
4. memory efficiency
5. inference speed
6. compression ratio
7. elegance

This order matters.

A very small model that is unstable or unusable is worse than a slightly larger model that actually runs well.

---

## Repo orientation

At the beginning of a new run, inspect the repository and identify the equivalent of these areas:

- `README.md` — overall repository context and setup
- `compressor/` — compression, distillation, pruning, export logic
- `engine_cpp/` — C++ CPU inference engine
- `worker/` — long-running job orchestration
- `benchmarks/` — latency, memory, and artifact comparisons
- `eval/` — quality evaluation harness
- `configs/` — experiment configs
- `scripts/` — operational scripts
- `examples/` — model-specific example runs

If the repo layout differs, map the actual layout to these conceptual roles before proceeding.

---

## Setup for a new autonomous run

To set up a new experiment run, work with the human to:

1. **Choose a run tag**
   - Propose a tag based on date and purpose, for example:
     - `mar26-gpt2-50m`
     - `mar26-smollm2-60m`
     - `mar26-export-int8`
   - The branch `autoresearch/<tag>` must not already exist.

2. **Create the branch**
   - Create a fresh branch from current main/master:
     - `git checkout -b autoresearch/<tag>`

3. **Read the repo for context**
   - Read the core files once at the beginning.
   - Do not repeatedly reread the entire repo every loop.
   - After initial orientation, prefer reading only the files you change and the files directly connected to them.

4. **Verify environment**
   - Confirm dependencies install or are already installed.
   - Confirm the CPU runtime backend is available or can be built.
   - Confirm dataset paths, tokenizer assets, and model checkpoints exist.
   - Confirm there is sufficient disk space for checkpoints, exports, and logs.

5. **Create run artifacts**
   - Ensure the following can exist for the run:
     - `results.tsv`
     - `run.log`
     - `artifacts/<tag>/`
     - `reports/<tag>/`
     - `checkpoints/<tag>/`

6. **Initialize results.tsv**
   - If it does not exist, create it with the header row only.

7. **Confirm the experiment lane**
   - Decide whether the run is:
     - `quick` for rapid screening,
     - `long` for worker-backed long distillation,
     - `deploy` for export/runtime benchmarking,
     - `repair` for fixing a broken pipeline.

Once setup is confirmed, begin the loop.

---

## Allowed changes

The agent **may** modify any file necessary to improve the project, including:

- compression logic
- student architecture generation
- distillation loss and schedule
- training loop
- checkpointing and resume logic
- export flow
- quantization flow
- benchmark harness
- C++ engine implementation
- CLI commands
- configs and examples
- documentation when needed for correctness

However, prefer changing the **smallest set of files** needed for a clear experiment.

---

## Disallowed behavior

The agent must **not**:

- add heavyweight dependencies without a strong reason,
- introduce GPU-only assumptions,
- rewrite the whole repository when a local fix is enough,
- break backward compatibility without documenting it,
- claim improvements without running benchmarks,
- keep a complicated change that delivers negligible benefit,
- silently ignore crashes or export failures,
- keep experimental code paths that cannot be reproduced.

---

## Token and context budget

The agent must be disciplined about token/context usage.

### Hard rules

1. Do **not** dump entire large files into context unless necessary.
2. Read the repo broadly **once**, then work incrementally.
3. Prefer diffs, targeted snippets, and direct file edits.
4. Keep experiment descriptions short and concrete.
5. Do not flood context with full logs; extract only the relevant lines.

### Working budget

Use this operating budget per loop:

- **Planning / reasoning budget**: target under **1,500 tokens** for the experiment plan.
- **Code inspection budget**: target under **4,000 tokens** per loop.
- **Log-reading budget**: target under **300 lines** total unless diagnosing a hard failure.
- **Patch budget**: prefer a small patch over a repo-wide rewrite.

### When to spend more tokens

Spend extra context only when:

- understanding the architecture is impossible without it,
- a bug spans multiple modules,
- export/runtime mismatch requires tracing across components,
- a long job failed and checkpoint recovery needs careful diagnosis.

### Context discipline rule

If you are about to reread a file you already understand, stop and ask whether you only need:

- the function signature,
- a small snippet,
- the latest diff,
- the error site.

If yes, read only that.

---

## Experiment lanes

There are four standard experiment lanes.

### 1. Quick lane

Purpose:
- cheap screening of ideas before committing worker time

Typical tasks:
- tiny distillation smoke tests
- short export validation
- benchmark harness validation
- config sanity checks
- student architecture size estimation

Recommended wall clock:
- 5 to 30 minutes

Recommended use:
- before every long job

### 2. Long lane

Purpose:
- real compression runs on the 24/7 worker

Typical tasks:
- distillation of a dense student
- structured shrinking studies
- training schedule sweeps
- low-RAM tuning
- checkpoint-based resume runs

Recommended wall clock:
- several hours to 24 hours or more

Recommended use:
- only after a quick lane validation passes

### 3. Deploy lane

Purpose:
- prove the result can be exported, quantized, loaded, and benchmarked

Typical tasks:
- ONNX export
- quantization
- validation against PyTorch outputs
- CPU engine loading
- latency and memory benchmarks

Recommended use:
- after any promising training result

### 4. Repair lane

Purpose:
- fix regressions or broken paths

Typical tasks:
- checkpoint resume bugs
- export mismatch fixes
- tokenizer/runtime issues
- benchmark correctness fixes
- C++ engine loading or generation fixes

Recommended use:
- whenever the pipeline is broken

---

## Recommended baseline approach

Unless evidence suggests otherwise, start with this baseline strategy:

1. choose a teacher model that is manageable on CPU,
2. define a **dense smaller student**,
3. distill the student from the teacher,
4. export the student,
5. quantize the exported student,
6. benchmark PyTorch vs exported runtime vs C++ engine,
7. keep only changes that improve the practical score.

This is the default strategy because it tends to produce results that are both compressible and deployable.

---

## Metrics

Every serious run must report these metrics when possible.

### Model metrics

- teacher params
- student params
- compression ratio
- artifact size on disk
- quantized artifact size on disk

### Quality metrics

Choose the main metric supported by the repo, for example:

- perplexity
- validation loss
- bits per byte
- task score
- output similarity proxy

Use one primary quality metric and keep it consistent.

### Runtime metrics

- model load time
- prompt latency
- tokens/sec
- peak RAM if measurable
- average RAM if available
- generation correctness sanity check

### Export metrics

- export success/failure
- validation drift between training framework and runtime backend
- quantization success/failure

---

## Decision policy: keep or discard

A result should be marked `keep`, `discard`, or `crash`.

### keep

Use `keep` when at least one of the following is true **without unacceptable regressions**:

- quality is better at similar runtime cost,
- runtime is faster at similar quality,
- RAM is lower at similar quality,
- artifact is smaller and still deployable,
- code is materially simpler with no real downside,
- pipeline robustness is improved significantly.

### discard

Use `discard` when:

- the change does not improve practical results,
- the gain is trivial and the complexity cost is high,
- latency improved but quality regressed too much,
- model shrank but engine/export became unreliable,
- RAM usage worsened too much for the target machine.

### crash

Use `crash` when:

- code fails to run,
- export fails completely,
- benchmark harness breaks,
- engine cannot load the artifact,
- the job exceeds the emergency timeout and must be killed.

---

## Practical acceptance thresholds

Use these default thresholds unless the human specifies otherwise.

### For quick experiments

Keep only if one of these is true:

- primary quality metric improves,
- runtime improves by at least **5%** at similar quality,
- RAM drops by at least **10%** at similar quality,
- artifact size drops by at least **10%** while deployability remains intact,
- the code becomes clearly simpler with no measurable downside.

### For long compression runs

A candidate is promising if:

- student quality is within an acceptable drop relative to teacher,
- CPU runtime improves materially after export/quantization,
- memory footprint decreases enough to matter on the target machine.

### For deployment changes

Keep only if:

- export succeeds,
- engine runs correctly,
- measured CPU behavior improves or remains acceptable,
- validation drift is small enough to trust.

---

## Result logging

Every experiment must be logged to `results.tsv` as a **tab-separated** file.

**This is mandatory for all runs**, including:
- Pure C++ engine tests
- Python training runs
- Export/quantization experiments
- Benchmark comparisons

### TSV Format

Use this header:

```tsv
commit	tag	lane	model	quality	load_s	toks_per_s	peak_ram_gb	artifact_mb	status	description
```

Column meanings:

1. `commit` — short git hash, or `workspace` if uncommitted
2. `tag` — run tag such as `mar26-gpt2-50m` or `mar27-cpp-engine`
3. `lane` — `quick`, `long`, `deploy`, `repair`
4. `model` — teacher/student identifier (e.g., `test-model-2L`, `smollm2-60m`)
5. `quality` — primary metric value (perplexity, or `N/A` for engine-only tests)
6. `load_s` — model load time in seconds
7. `toks_per_s` — throughput (tokens per second)
8. `peak_ram_gb` — peak memory in GB
9. `artifact_mb` — artifact size in MB
10. `status` — `keep`, `discard`, `crash`
11. `description` — short concrete description of the experiment

### Examples

```tsv
commit	tag	lane	model	quality	load_s	toks_per_s	peak_ram_gb	artifact_mb	status	description
1a2b3c4	mar26-gpt2-50m	quick	gpt2-student-v1	22.480	0.9	18.2	2.6	108.4	keep	6-layer dense student smoke test
2b3c4d5	mar26-gpt2-50m	deploy	gpt2-student-v1-int8	22.610	0.5	28.7	1.9	58.1	keep	onnx int8 export and cpp load path
3c4d5e6	mar26-smollm2-60m	long	smollm2-student-v2	0.000	0.0	0.0	0.0	0.0	crash	resume bug after checkpoint rotation
workspace	mar27-cpp-engine	quick	test-model-2L	N/A	0.05	533	0.01	2.6	keep	pure C++ engine test on Termux
workspace	mar27-cpp-engine	deploy	smollm2-60m-cpp	N/A	0.12	50	0.12	120	keep	C++ engine inference 60M model
```

### Logging Script

Use this pattern to log results:

```bash
# Append to results.tsv
echo -e "workspace\tmar27-cpp-engine\tquick\ttest-model-2L\tN/A\t0.05\t533\t0.01\t2.6\tkeep\tpure C++ engine test" >> results.tsv
```

Or use the CLI:

```bash
python scripts/cli.py log --tag mar27-cpp-engine --lane quick --model test-model-2L --toks 533 --status keep --desc "pure C++ engine test"
```

### When to Log

Log **immediately after** each experiment completes:
- After C++ engine benchmark
- After training run
- After export succeeds/fails
- After quality evaluation

**Do not batch logs** - log each result as it happens.

---

## Standard run artifacts

Each run should produce these files when applicable:

- `run.log` — full command output
- `reports/<tag>/summary.md` — short human-readable summary
- `reports/<tag>/metrics.json` — machine-readable metrics
- `checkpoints/<tag>/...` — intermediate checkpoints
- `artifacts/<tag>/...` — exported models and quantized artifacts
- `artifacts/<tag>/validation.json` — drift and export validation
- `artifacts/<tag>/benchmarks.json` — benchmark output

---

## Commands and execution policy

The repository may expose different commands. Discover the actual commands, then use them consistently.

Typical categories are:

- compression/train command
- export command
- quantization command
- benchmark command
- C++ engine build command
- C++ generation command

### Logging rule

Always redirect long-running commands to a log file. Example pattern:

```bash
<command> > run.log 2>&1
```

Do not flood the working context with full live logs.

### Crash diagnosis rule

If a run fails, inspect only the smallest useful slice first, for example:

```bash
tail -n 80 run.log
```

If needed, then inspect more.

---

## Emergency timeout policy

Each run must have a reasonable timeout.

### Default timeout guidance

- quick lane: kill if it exceeds **2x** the expected runtime
- deploy lane: kill if export or benchmark is clearly stuck
- long lane: allow long runtimes, but require checkpoint progress

A long job may continue for hours, but if it produces no checkpoint, no log progress, or no measurable advancement for an unreasonable period, treat it as a failure.

---

## Benchmark policy

Do not claim success until the benchmark comparison exists.

At minimum, compare:

1. original teacher
2. dense student
3. exported student
4. quantized student
5. C++ engine result

For each comparison, try to report:

- artifact size
- load time
- prompt latency
- tokens/sec
- RAM
- quality sanity check

If one stage cannot yet be benchmarked, explicitly say which stage is missing.

---

## CPU-engine policy

The C++ engine is not optional. The project goal includes a real engine path.

The preferred engine behavior is:

- load exported model
- run autoregressive generation on CPU
- expose a clean CLI
- support practical generation controls if implemented
- measure throughput and latency
- fail clearly when the artifact is incompatible

If the repo supports multiple backends, prefer the one that is:

- easiest to build on CPU,
- most robust under low RAM,
- easiest to benchmark reproducibly.

---

## Worker policy

The 24/7 worker should be treated as a scarce but valuable resource.

Use it for:

- long distillation,
- checkpointed sweeps,
- architecture studies,
- export + validation after a promising checkpoint,
- overnight benchmarks.

Do not waste it on:

- clearly broken configs,
- unvalidated small patches,
- ideas that failed the quick lane already.

### Worker requirements

Every worker job should:

- be resumable,
- checkpoint frequently,
- write a machine-readable status file,
- record enough metadata to reproduce the run,
- generate a concise summary on completion.

---

## Simplicity rule

Simplicity matters.

If two changes produce similar results, prefer the one that is:

- easier to explain,
- easier to export,
- easier to benchmark,
- easier to maintain,
- easier to run on a weak CPU machine.

A tiny gain that adds ugly complexity is usually not worth keeping.

A tiny gain from deleting code or simplifying the path is often worth keeping.

---

## Baseline policy

For any new model target, the first meaningful end-to-end pass should be:

1. establish a baseline teacher benchmark,
2. define a first dense student,
3. run a short distillation smoke test,
4. export the student,
5. benchmark export/runtime,
6. only then begin deeper experimentation.

Never skip the baseline.

---

## Experiment loop

Run this loop repeatedly.

### LOOP FOREVER

1. Check current git branch and current commit.
2. Read `results.tsv` and the latest summaries to understand what has already been tried.
3. Choose exactly one focused experimental idea.
4. Make the smallest reasonable patch.
5. Commit the change or otherwise snapshot the workspace state.
6. Run the experiment and redirect output to `run.log`.
7. Extract the key metrics.
8. If the run crashed, inspect the smallest useful log tail and attempt a focused repair.
9. Record the result in `results.tsv`.
10. If the result is meaningfully better, keep it and advance.
11. If the result is worse or useless, revert to the last good state.
12. Repeat.

### Focus rule

Each loop should test **one main idea** only, for example:

- smaller FFN ratio
- fewer layers
- better distillation temperature
- lower-RAM dataloader behavior
- better export path
- int8 calibration fix
- C++ engine cache optimization

Do not mix many ideas in one loop unless they are inseparable.

---

## Suggestions for good early experiments

If the repo is immature, start here:

1. **Teacher baseline benchmark**
2. **Student parameter budget estimator**
3. **Dense student v1** with conservative shrinkage
4. **Short distillation smoke test**
5. **ONNX export validation**
6. **INT8 quantization path**
7. **C++ engine loader + greedy generation**
8. **Benchmark harness comparing teacher vs student vs quantized**
9. **Checkpoint resume robustness test**
10. **Worker queue + summary reporting**

If the repo is already mature, then try:

- layer-depth sweeps,
- FFN width sweeps,
- distillation temperature sweeps,
- hidden-size vs latency tradeoff studies,
- quantization calibration dataset improvements,
- runtime cache and threading improvements.

---

## Suggestions for what probably works best on this project

For this specific project type, these are usually the strongest bets:

- build a **dense student** instead of relying on raw pruning,
- shrink **depth + width + FFN** carefully,
- keep tokenizer compatibility where practical,
- export early and often,
- measure **real CPU throughput**, not only model size,
- quantize only after the student is reasonably good,
- keep the C++ engine path simple and robust first, fancy later.

These are usually weak bets:

- aggressive unstructured sparsity as the main strategy,
- exotic layers that complicate export,
- giant config systems before the basic loop works,
- large dependency additions for marginal gains,
- optimizing paper metrics without checking CPU behavior.

---

## Human escalation policy

Escalate to the human only when necessary, for example:

- the target metric is ambiguous,
- a required dataset or checkpoint is missing,
- the repo structure is too incomplete to infer safely,
- two directions are both strong and require a product choice,
- the environment is broken in a way the agent cannot repair.

When escalating, be concise and concrete.

Do not ask the human questions that can be resolved by inspecting the repo.

---

## Final rule

The purpose of this project is not to produce the most interesting compression idea.

The purpose is to produce a **working compression toolchain** and a **working CPU inference engine** that together make a model **smaller, faster, and cheaper to run**.

If a change helps that outcome, keep it.
If it does not, discard it.

