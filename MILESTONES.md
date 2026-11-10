# Milestones — harness build for the toy test and Phase 1

Each milestone ends in a **checkpoint**: the full test suite passes (including
the Docker tests), README.md is brought up to date (status line, rules,
interfaces, architecture), the work is committed, and the commit is tagged
`mN-<name>`.
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

## M1 — Run setup and the `single` topology ✅

Done when: one command runs a config end to end with the fake client and a
real container, and produces a complete run folder.

- [x] Refuse to start if the git tree is not clean (`runs/` excluded, so
      runs can be batched between commits; see note below)
- [x] Create `runs/<date>_<experiment>_<topology>_<nnn>/` with exact copies
      of the config and design, `commit.txt`, `log.jsonl`, `outputs/`,
      `final_hash.txt`
- [x] First log entries: config, commit, Docker image ID, sandbox Python
      version and package list, host Python, SDK version, system prompts
- [x] `single` topology: start container, run agent, stop container, always
      tidy up, even after an error or Ctrl-C
- [x] After the run: import the analysis entry point, call it with accepted
      submissions, write the final hash file, append to `runs/index.jsonl`
      (also for failed and aborted runs). Analysis results go in the index
      only, never in the log
- [x] Command-line entry point: `python -m harness.run <config>`
- [x] When an agent ends: tar of `/work` streamed out of the container into
      the run's outputs (size-capped, hashed into the log), inspected without
      extraction, then container and workspace removed
- [x] Index line records storage: bytes of the log, of `outputs/`, and of
      each workspace archive (see LESSONS.md §1)
- [x] Stored output per execution capped (`max_stored_output_mb`; toy: 100),
      with the agent still shown the true end of its output
- [x] Nothing that can fail before the run creates a run folder: config,
      referenced files, imports, topology, logger trial write, clean tree

Note on the clean tree: a finished run writes into `runs/`, so a literal
reading would block the next run until a commit. `runs/` holds no code,
prompts or configs, so excluding it keeps the rule's purpose intact.

SEAN (non-blocking, defaults in brackets):
- ~~What happens to the agent's `/work` after a run?~~ Decided: tar archive,
  capped, hashed, never auto-extracted; then the workspace is deleted
- Tracebacks: plain stdin (no source lines) or a small launcher that restores
  them? [plain stdin, as specified]
- Set `/tmp` and `/dev/shm` sizes in the config so runs match across
  machines (OrbStack gives `/dev/shm` ~4 GB; most Docker setups 64 MB)?
  [yes; see LESSONS.md §2]
- Add `harness/config.py` and `harness/run.py` to the CLAUDE.md layout, and
  note the `runs/` exclusion in its "Clean tree required" rule? [yes, once
  you agree; CLAUDE.md is yours]
- Update CLAUDE.md's Sandbox "Execution" bullet: code now runs through
  `/opt/harness/launcher.py` (still arrives on stdin) [yes, once you agree]
- Change CLAUDE.md's "Commit `runs/` to git after each batch of runs" to
  "the harness commits each run when it ends; push after each batch"?
  [yes, once you agree]

Checkpoint: `m1-single-run`

---

## Between M1 and M2

- [x] Submissions must be finite; divergence reported via a schema field
      (pattern: `tests/fixtures/float_schemas.py`); logged before accepted
- [x] After every call, list `/work`, `/tmp`, `/dev/shm` (LESSONS.md §2)
- [x] Toy prompt states the sandbox's real rules (README rule)
- [x] Every run committed to git when it ends: its folder and the index
      line only; never pushed automatically
- [x] README.md: rules, interfaces, architecture at a glance
- [x] Test: the API key never enters the container
- [x] Launcher in the image: tracebacks with source lines; fixes
      `multiprocessing`, which failed for code piped in on stdin under
      Python 3.14

---

## M2 — Real logger and verifier

Done when: the harness uses `harness/logger.py`, and `verify.py` passes on a
real run log and fails on tampered copies.

- [ ] **SEAN, blocking:** write `logger.py` and `verify.py`
- [ ] **SEAN:** `verify.py` also rehashes every file in `outputs/` and compares
      with the hash recorded in the log, and reports files with no log entry
      (decided 2026-11-09). Where the hashes are recorded:
      - `execution` entries: `data.files` = `{file name: sha256}`, files in
        `outputs/<agent>/` (the `.py`, `.out`, `.err` and `.files.jsonl`
        listing of each execution)
      - `workspace_archive` entries: `data.file` + `data.sha256` (the tar), and
        `data.errors_file_sha256` for `<file>.err`, in `outputs/<agent>/`
      - the `agent` field of each entry names the `<agent>` folder
- [ ] Tests for them (I write these): chain verifies; an edited entry is
      detected; a deleted or reordered line is detected; a truncated log is
      detected through the recorded final hash; a half-written last line is
      detected; an altered, missing or extra file in `outputs/` is detected
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
