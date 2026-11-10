# Milestones — harness build for the toy test and Phase 1

Each milestone ends in a **checkpoint**:

1. The full test suite passes, including the Docker tests.
2. **Invariant re-verification.** Walk through every invariant and rule in
   CLAUDE.md and README §1 and confirm each still holds in the code. Check
   that each "Enforced by" entry is still true; anything that drifted is
   fixed or raised with Sean before tagging. (`tests/test_docs.py`
   automatically checks that every test the README cites still exists.)
3. README.md is brought up to date (status line, rules, interfaces,
   architecture).
4. The work is committed, the commit message records the re-verification
   result, and the commit is tagged `mN-<name>`.
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
- ~~`/tmp` and `/dev/shm` sizes~~ Done 2026-11-10: `tmp_mb`, `shm_mb` and
  `memory_mb` in the config
- ~~CLAUDE.md updates~~ Done 2026-11-10: layout, Runs rules (clean tree
  ignores `runs/`; harness commits each run), Sandbox reduced to principles,
  milestone re-verification rule

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
- [x] `/work` is an in-memory filesystem with a quota (`work_mb`); nothing
      from the host is mounted, so an agent can't fill the laptop's disk
- [x] Launcher in the image: tracebacks with source lines; fixes
      `multiprocessing`, which failed for code piped in on stdin under
      Python 3.14

---

## M2 — Real logger and verifier ✅

Done when: the harness uses `harness/logger.py`, and `verify.py` passes on a
real run log and fails on tampered copies.

- [x] **SEAN, blocking:** write `logger.py` and `verify.py`
- [x] Contract for `verify` (agreed 2026-11-10): `verify(log_path,
      expected_final_hash=None) -> (ok, problems)`. `problems` lists **every**
      problem, each a dict with `kind`, `line` (1-based, or `None`) and
      `detail`; file problems add `file` (relative to `outputs/`). Kinds:
      `edited_entry`, `broken_link`, `unreadable_line`,
      `incomplete_last_line`, `final_hash_mismatch`, `file_altered`,
      `file_missing`, `file_extra`. Checks are local (each line against its
      own hash and the previous line's *stored* hash), so one tampered line
      is reported once. Never raises. Full spec: top of `tests/test_verify.py`
- [x] **SEAN:** `verify.py` also rehashes every file in `outputs/` and compares
      with the hash recorded in the log, and reports files with no log entry
      (decided 2026-11-09). Where the hashes are recorded:
      - `execution` entries: `data.files` = `{file name: sha256}`, files in
        `outputs/<agent>/` (the `.py`, `.out`, `.err` and `.files.jsonl`
        listing of each execution)
      - `workspace_archive` entries: `data.file` + `data.sha256` (the tar), and
        `data.errors_file_sha256` for `<file>.err`, in `outputs/<agent>/`
      - the `agent` field of each entry names the `<agent>` folder
- [x] Tests for them: `tests/test_verify.py` (15 tests, including lines
      not in exact canonical form, e.g. a hidden duplicate key). They skip
      until both files work, then run automatically. Checked for
      consistency against a throwaway reference outside the repo
- [x] Tamper trials on a real run folder: edited, missing, extra,
      symlinked and CRLF-converted outputs; edited, garbage, half-written,
      reformatted and deleted log lines. Each is reported as the right kind
      at the right line; the untouched folder verifies clean
- [x] `.gitattributes`: `\n` line endings everywhere, and `runs/` never
      converted, so a Windows checkout can't make outputs look altered
- [x] Swap the dummy logger out of the run path. The command line uses
      `Logger`; the Docker end-to-end tests use the real logger and
      verifier, and a tampered copy of that real run fails verify (unit
      tests keep the dummy)
- [x] `verify.py` runs automatically at the end of every run; the index
      line gets `verify: {ok, problem_count, problems (first 10)}`. A crash or
      malformed answer becomes a `verifier_error` problem. Before any run
      folder exists, the logger and verifier get a trial run on a one-entry
      scratch log, so an unfinished verifier is caught early

Checkpoint: `m2-hash-chain`

---

## M3 — First live run (toy experiment, real API) ✅

Done when: one real `single` run of `toy_sum` completes, verifies, and is
committed to `runs/`.

- [x] **SEAN, blocking:** go-ahead to spend API money, and a spending limit.
      Agreed 2026-11-10: a Console spending limit of $20–25/month on the
      project's key, and no more than ~5 live toy runs without checking in.
      Expected cost ~$0.10 per toy run (claude-opus-5-5, $4 / $20 per million
      input / output tokens); worst case ~$4–5, capped by `max_total_tokens`
- [x] **SEAN, blocking:** `ANTHROPIC_API_KEY` in your shell environment
      (never in a file)
- [x] One marked test that calls the real API, skipped by default:
      `tests/test_live_api.py` (runs into a temp folder, commits nothing)
- [x] Pre-run API check on the command line (`check_api_access`): key set,
      key accepted, model exists. Free (no tokens). On failure, no run folder
      is created
- [x] A fake key appears nowhere in a run folder, including log, outputs and
      workspace tar (`test_api_key_appears_nowhere_in_a_run_folder`)
- [x] Run it; read the log together; check `verify.py`, the index line and
      the analysis result. 2026-11-10: `2026-11-10_toy_sum_single_001`,
      completed in 2 turns (`run_python`, then `submit_result`), verify ok,
      analysis `estimate 1555, passed true`; key absent from the run folder
- [x] Commit the run folder (the harness does this itself, commit
      `5a4aa62`) and record the token count and cost in LESSONS.md (§3:
      2,459 tokens, $0.0125; about 8× under the estimate)

**Runbook for the first live session**

1. Sean: set the Console spending limit; in the terminal, `export
   ANTHROPIC_API_KEY=...` (never in a file). OrbStack running.
2. `git status` is clean (the harness refuses otherwise) and
   `.venv/bin/python -m pytest -q` is green.
3. Smoke test, not recorded: `.venv/bin/python -m pytest -m api
   tests/test_live_api.py -v -s` (one toy run into a temp folder).
4. The recorded run: `.venv/bin/python -m harness.run
   tests/fixtures/toy_sum/config.yaml`. It checks the key first, then runs,
   verifies, writes the index line and commits the run. It never pushes.
5. Read the log together; check verify, the index line and the analysis.
6. Re-verify invariants, update README and this file, tag
   `m3-first-live-run`. Sean pushes.

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
