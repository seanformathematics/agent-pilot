# Milestones — harness build for the toy test and Phase 1

Each milestone ends in a **checkpoint**: the full test suite passes (including
the Docker tests), the work is committed, and the commit is tagged `mN-<name>`.
A checkpoint is the point we can always return to and say exactly what
existed.

**SEAN** marks where your input is needed. **Blocking** means the milestone
cannot finish without it.

---

## M0 — Scaffold, sandbox, harness core ✅

Done when: the layout exists; the sandbox, harness and agent loop work against
the fake client and dummy logger; the required isolation tests pass.

- [x] Layout, toy experiment in `tests/fixtures/toy_sum/`, config format
- [x] Sandbox: hardened container, code on stdin, output via pipes into
      harness-owned files, no lingering processes, zombie reaper
- [x] Harness: tool validation and dispatch, submissions, flags, budgets
- [x] Agent: tool-use loop, logged harness-side API retries
- [x] Required tests covered so far: emissions logged, malformed calls
      rejected and logged, log unreachable from the container, timeouts and
      truncation, submission without code execution flagged, agents isolated

Checkpoint: `m0-harness-core`

---

## M1 — Run setup and the `single` topology

Done when: one command runs a config end to end with the fake client and a
real container, and produces a complete run folder.

- [ ] Refuse to start if the git tree is not clean
- [ ] Create `runs/<date>_<experiment>_<topology>_<nnn>/` with exact copies
      of the config and design, `commit.txt`, `log.jsonl`, `outputs/`
- [ ] First log entries: config, commit, model, Docker image ID, host and
      sandbox Python versions, SDK version, system prompts
- [ ] `single` topology: start container, run agent, stop container, always
      tidy up, even after an error
- [ ] After the run: import the analysis entry point, call it with accepted
      submissions, write the final hash file, append to `runs/index.jsonl`
      (also for failed and aborted runs)
- [ ] Command-line entry point, e.g. `python -m harness.run <config>`

SEAN (non-blocking, defaults in brackets):
- What happens to the agent's `/work` after a run? [archive it into the run
  folder, marked untrusted, with hashes]
- Tracebacks: plain stdin (no source lines) or a small launcher that restores
  them? [plain stdin, as specified]
- Cap on output stored per execution? [100 MB per stream, then stop writing
  and note it]
- Add `harness/config.py` to the CLAUDE.md layout? [yes]

Checkpoint: `m1-single-run`

---

## M2 — Real logger and verifier

Done when: the harness uses `harness/logger.py`, and `verify.py` passes on a
real run log and fails on tampered copies.

- [ ] **SEAN, blocking:** write `logger.py` and `verify.py`
- [ ] Tests for them (I write these): chain verifies; an edited entry is
      detected; a deleted or reordered line is detected; a truncated log is
      detected through the recorded final hash; a half-written last line is
      detected
- [ ] Swap the dummy logger out of the run path (tests may keep it)
- [ ] `verify.py` runs automatically at the end of every run; result logged
      in the index

Checkpoint: `m2-hash-chain`

---

## M3 — First live run (toy experiment, real API)

Done when: one real `single` run of `toy_sum` completes, verifies, and is
committed to `runs/`.

- [ ] **SEAN, blocking:** go-ahead to spend API money, and a spending limit
- [ ] **SEAN, blocking:** `ANTHROPIC_API_KEY` in your shell environment
      (never in a file)
- [ ] One marked test that calls the real API, skipped by default
- [ ] Run it; read the log together; check `verify.py`, the index line and
      the analysis result
- [ ] Commit the run folder

Checkpoint: `m3-first-live-run`

---

## M4 — `independent` topology

Done when: n fresh single-agent runs of one config each produce their own run
folder and index line, and cannot see each other.

- [ ] SEAN: one run folder per agent, or one folder holding n agents?
      [one folder per agent, sharing a batch id]

Checkpoint: `m4-independent`

---

## M5 — Ising readiness

Done when: an Ising config loads, validates, and runs one agent end to end.

- [ ] **SEAN, blocking:** `experiments/ising/` — design, prompts, config,
      submission model, sanity checks, analysis
- [ ] **SEAN, blocking:** the package list for the sandbox image (now numpy
      and scipy). Changing the image is an experimental change, so this is
      fixed before the first Ising run
- [ ] I check only that the harness can run it; I do not edit anything in
      `experiments/`

Checkpoint: `m5-ising-ready`

---

## Later (not yet planned in detail)

- `one_shot` and `hierarchical` topologies. `hierarchical` needs design
  decisions from SEAN first: message routing, round limits, what a supervisor
  can see.
