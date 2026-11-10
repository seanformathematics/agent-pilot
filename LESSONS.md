# Lessons — what this work teaches about running agent experiments

Meta-lessons Sean wants to draw from the project, separate from the physics.
Each entry says what we are watching, where the evidence lives, and what
question it should eventually answer.

---

## 1. How much storage does a trustworthy record cost?

**Why it matters.** Every run keeps a full record: the log, every script and
its full output, and a tar of each agent's workspace. The toy run is tiny,
but Ising runs, multi-agent topologies and Phase 2/3 calculations will be
much larger. We want to see the growth coming, not discover it when a disk
or a git push fails.

**What is measured now** (in each run's `log.jsonl`):
- every execution: `stdout_bytes`, `stderr_bytes` (full sizes, before capping)
- every agent's workspace archive: `bytes`, the cap (`max_bytes`), whether it
  was truncated, and member count and total member size from the inspection

**What M1 adds** (one line per run in `runs/index.jsonl`): bytes of
`log.jsonl`, of `outputs/` as a whole, and of each workspace archive.

**Questions to answer as runs accumulate:**
- How does storage scale with number of agents, turns, and output volume?
- Which part dominates: log, execution outputs, or workspace archives?
- Are the caps (output per stream, workspace archive size) ever hit, and is
  anything lost when they are?
- At what point do outputs leave git, keeping only their SHA-256 there
  (as CLAUDE.md already anticipates)?

---

## 2. How much of an agent's scratch space should the record capture?

**Why it matters.** Besides `/work`, an agent can write to `/tmp` and
`/dev/shm` inside its container. Both keep their contents between
`run_python` calls, and neither is in the end-of-agent archive. So an agent
can hold state the record never shows: a second, invisible workspace.

**What we do now (decided 2026-11-10, a deliberate middle ground).** After
every `run_python` call the harness lists `/work`, `/tmp` and `/dev/shm`:
type, size, modification time and path of every entry, never contents. The
listing is `outputs/<agent>/<exec_id>.files.jsonl` (hashed in the log); the
log entry carries a per-folder summary. Nothing is wiped.

**What we could do if it turns out to matter:**
- snapshot contents (or their hashes) per call, not just names and sizes
- archive `/tmp` and `/dev/shm` at the end alongside `/work`
- wipe them after each call so `/work` is the only persistent space
- record diffs between consecutive listings

**Questions to answer:**
- Do agents actually use `/tmp` or `/dev/shm` to carry state? (The listings
  will show it.)
- What do practitioners at Anthropic do about agent scratch space in
  evaluation harnesses? (Ask, if the residency happens.)

**Related, resolved 2026-11-10:** `/dev/shm` was about 4 GB under OrbStack
but 64 MB under most other Docker setups, and `/tmp` was fixed at 256 MB in
code. Both are now set in the config (`tmp_mb`, `shm_mb`), so they are
recorded in every run and identical across machines. Lesson: anything the
environment decides silently is an uncontrolled variable; make it a config
value.
