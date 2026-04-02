# CLAUDE.md — Project Rules for MIMIC (FRAME)

## Research Notes (MANDATORY)

Every session that involves experiments, training runs, architectural changes, debugging findings, or infrastructure work MUST be documented in `docs/research-notes-YYYY-MM-DD.md`. Do not wait until the end — write notes as findings emerge.

### When to write notes

- At the start of a session if prior sessions are undocumented
- After each significant finding, training result, or debugging insight
- Before ending a session — summarize what happened and what's next
- If multiple days pass without notes, backfill from git history and context

### Structure

Every research notes file follows this format:

```markdown
# Research Notes — YYYY-MM-DD

## Context

One paragraph: what was the state coming in? Link to prior notes.
What were the active runs/goals?

---

## Finding N: [descriptive title]

State the finding clearly in 1-2 sentences. Then provide:
- **Data**: exact numbers, metrics, step counts — never approximate
- **Evidence**: what experiment/run/log showed this
- **Implications**: what does this mean for the project

Use tables for comparing runs or metrics.

---

## [Phase/Action/Experiment sections]

Describe what was done, why, and what happened. Include:
- Exact commands used for training/eval
- Commit hashes for code changes
- Machine assignments (which GPU machine ran what)
- Wandb run names/links where applicable

---

## Next Steps

Numbered list of what to do next, in priority order.
```

### Formatting rules

- **Findings are numbered sequentially across all notes files** (not per-file). If the last file ended with Finding 9, the next file starts with Finding 10.
- **Always cite exact numbers** — never say "about 87%" when the log says "87.3%"
- **Include commands** — the exact training/eval command used, not a paraphrase
- **Tables for comparisons** — whenever comparing runs, configs, or metrics
- **Link prior notes** — every file's Context section links to the previous notes file
- **Machine references** — use the letter designations from GPUS.md (Machine C, E, F, G)

### What to document

- Training run configs, results, and termination reasons
- Architecture/code changes with commit hashes
- Debugging investigations and root causes found
- Closed-loop eval results (damage dealt/received, stocks, behavior observations)
- Data pipeline work (uploads, processing, sharding)
- Key decisions and their rationale

### What NOT to put in research notes

- Code implementation details (that's what git diff is for)
- Speculative plans (use plan files for that)
- Duplicate content from memory files

---

## Code Style

- All new CLI flags must be toggles with backward-compatible defaults
- Never automate file deletion in pipeline scripts
- Always back claims with exact data — no approximate numbers
- Install deps and verify CUDA+imports before launching on new machines
- Document and push code before launching training runs
- Character normalization: Zelda/Sheik → ZELDA_SHEIK, Popo/Nana → ICE_CLIMBERS

## Project Structure

- `mimic/` — model, dataset, features, frame encoder
- `train.py` — training loop with all CLI flags
- `inference.py` — closed-loop inference against Dolphin
- `tools/` — data processing, diagnostics, evaluation scripts
- `docs/` — research notes, architecture diagrams, sweep results
- `data/` — local data directories (not in git)
- `checkpoints/` — saved model weights (not in git)
