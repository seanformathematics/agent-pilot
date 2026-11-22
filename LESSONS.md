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

**What the first milestone (M1, see MILESTONES.md) added** (one line per
run in `runs/index.jsonl`): bytes of
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

---

## 3. What does a run cost, in tokens and money?

**Why it matters.** Budgets are set in tokens (`max_total_tokens`) and
agreed in dollars. Estimates made before any live run can be far off; the
real numbers come from the `usage` field of every `api_response` entry in
the log, and `tokens_used` in the index.

Prices used below are list prices per million input / output tokens, as
given in Anthropic's model documentation on 2026-11-18, with no prompt
caching (the harness doesn't use it): Opus 5.5 $4 / $20; Sonnet 5.5
$2 / $10; Haiku 5.5 $0.10 / $0.50, or $0.50 / $2.50 for any request whose
prompt is over 100K tokens.

**Toy runs** (claude-opus-5-5, effort `low`):

| Run | Turns | Input | Output | Total | Cost |
|-----|-------|-------|--------|-------|------|
| smoke test (temp folder, not recorded), 2026-11-10 | 2 | — | — | 2,453 | ~$0.013 |
| `2026-11-10_toy_sum_single_001` | 2 | 2,293 | 166 | 2,459 | $0.0125 |
| `2026-11-10_toy_sum_independent_001` (3 agents) | 2 each | 6,879 | 500 | 7,379 | $0.0375 |

**First observation.** The pre-run estimate was ~$0.10 per toy run; the
real cost is about 8× lower. Input dominates (~93% of tokens): every turn
resends the system prompt, tool schemas and the history so far, so input
grows with the number of turns. Ising runs will have many more turns and
larger tool outputs, so expect cost to scale faster than linearly in turns.

**Independent runs scale linearly, as expected.** Three agents cost three
single runs (2,459–2,460 tokens each); nothing is shared between them, so
there is no saving and no overhead. On a trivial task the three agents
wrote identical code and gave identical answers; spread between agents
only becomes informative on harder tasks.

**Ising runs** (computed from the `usage` of every `api_response`, at the
prices above; Haiku requests over 100K input tokens charged at the higher
rate):

| Run | Model, effort | Agents | Cost |
|-----|---------------|--------|------|
| `2026-11-13_ising_prototype_single_001` | Opus, medium | 1 | $0.12 |
| `2026-11-14_ising_via_sonnet_solo_single_001` | Sonnet, medium | 1 | $0.10 |
| `2026-11-14_ising_sonnet_independent_independent_001` | Sonnet, medium | 10 | $0.54 |
| `2026-11-16_ising_haiku_at_medium_independent_001` | Haiku, medium | 10 | $0.095 |
| `2026-11-17_ising_haiku_at_medium_pinned_independent_002` | Haiku, medium | 10 | $0.086 |
| `2026-11-16_…_haiku_at_high_…_001`, `2026-11-17_…_high_…_001`, `…_high_pinned_…_001` | Haiku, high | 10 each | $0.13–0.15 |
| `2026-11-17_…_haiku_at_xhigh_…_001`, `2026-11-18_…_xhigh_pinned_…_001` | Haiku, xhigh | 10 each | $0.29–0.33 |
| `2026-11-17_…_haiku_at_max_…_002`, `2026-11-18_…_max_pinned_…_001` | Haiku, max | 10 each | $2.08–2.66 |

At medium effort a 10-agent Haiku batch cost about a sixth of the Sonnet
batch: a twentieth of the price per token, but about four times the
tokens. At max effort the same batch cost about 4–5× the Sonnet batch:
output grew about sevenfold over high effort, every turn resends the
whole history, and 14–20
requests per batch crossed the 100K-token line into the higher rate.

**Questions to answer as runs accumulate:**
- How do tokens scale with turns, and with the size of tool results?
- Would prompt caching (system prompt + tools) change the cost materially?
- What `max_total_tokens` is right for an Ising run?

---

## 4. Grading code written for one agent silently ignores the rest

**What happened.** `experiments/ising/analysis.py` was written while every
run had one agent, and grades `submissions[-1]`. The harness passes it every
accepted submission in the run (each tagged with its `agent_id`). In the
first independent batch, `2026-11-14_ising_sonnet_independent_independent_001`
(10 Sonnet agents, all accepted), the index line therefore described only
`ising-10`: `rms_pull` 1.37 and one pull ≥ 3. Recomputed over all ten
agents, the pooled rms pull is 1.02 across 140 points. Nothing failed and
nothing warned: the run "passed", the record was complete, and nine
agents were simply not graded.

**Why it matters.** The record is only as good as what reads it. Moving
from one agent to many changed what "the result" of a run means, and code
built for one agent didn't notice. Any analysis or summary that picks one
item from a list (the last, the first, the best) is a place where this can
happen again: more topologies, more agents and resubmissions all multiply
the items.

**What to do.**
- Analysis grades every submission it is given, per agent, and reports
  pooled statistics alongside.
- When a run's shape changes (topology, number of agents, resubmissions),
  check the code that reads it, not only the code that writes it.
- Cheap guard: the analysis states how many submissions it received and how
  many it graded, so a mismatch is visible in the index line.

**Questions, and where we landed:**
- What does "passed" mean for a run of n agents: all of them, a majority,
  the pooled statistics? *Current answer:* report completion, precision
  and calibration as separate rates with uncertainties. Judge calibration
  per agent with a χ²-style test over its 14 points, not its single worst
  pull. Pool pulls only where the reference error is small, and define
  `passed` as an explicitly stated combination of these. In future tasks,
  unlike this one, precision could also decide completion.
- Should the harness check that the number graded equals the number of
  accepted submissions? *Done* (commit `4547e97`): the harness compares
  the analysis's `n_graded` and `graded` with the accepted submissions and
  records the result as `analysis_coverage` in the index line. A shortfall
  is recorded, not treated as a failure, so it can be investigated.

---

## 5. A 4σ outlier that moved when we looked again

**What happened.** In `2026-11-14_ising_sonnet_independent_independent_001`
(10 Sonnet agents), `ising-10` reported U = 0.38594 ± 0.00085 at T = 2.5.
Against the reference (0.38206 ± 0.00039) that is a pull of **4.14σ**. The
chance that the worst of 140 honest pulls reaches 4.14 is about 0.5%, so it
looked like an agent understating its error. Re-analysed on 2026-11-15:

- **Against its peers, not the reference**, ising-10 is 2.82σ from the
  weighted mean of the other nine. The worst of 140 points reaching that
  happens about half the time (p ≈ 0.49). Nothing unusual.
- **The rest of the gap belongs to the reference.** The other nine agents'
  weighted mean at T = 2.5 is 0.38340 ± 0.00029, which is 2.7σ above the
  reference. All ten together sit 3.3σ above it. Every agent's pull at T = 2.5
  is shifted by the same amount (mean pull +1.47 across agents).
- Testing the reference against the agents' consensus at every T (at each
  T, the inverse-variance weighted mean of the ten agents minus the
  reference, divided by their combined error, squared and summed over the
  14 temperatures) gives χ² = 24.0 for 14 degrees of freedom (p = 0.046).
  Almost all of the excess comes from T = 2.5, which contributes 11.2;
  without it, χ² = 12.8 for 13 degrees of freedom.

So "agent ising-10 is 4σ off" turned into "the reference value at T = 2.5
is about 3σ low, or all ten agents share a bias there, and ising-10 is
an ordinary 2.8σ fluctuation on top". The surprise was real, but it was in
the wrong place.

**Why it happened.** A pull, (U − U_ref)/√(U_err² + σ_ref²), mixes two
sources of noise: the agent's and the reference's. The reference's part is
the same for every agent, so with many agents it stops averaging out and
shows up as a shared offset. Treating 140 pulls as 140 independent draws
hides that. At T ≤ 2.2 it gets worse: the median agent's error there is
0.06–0.55 times σ_ref (depending on T), so their pulls are almost pure
reference noise (for example, every agent gets about +1.65 at T = 2.1).

**What to do.**
- With several agents, split the comparison in two: agents against each
  other at each T (tests the agents; no reference needed), and the agents'
  consensus against the reference (tests the reference). Report both.
- Before calling an agent anomalous, ask whether the other agents share the
  shift at that point.
- Treat the reference as a measurement with its own error bars, open to
  being checked. With more agents it will be the least precise number in
  the comparison at some temperatures.

**Questions to answer:**
- Is the reference at T = 2.5 really low? An independent recomputation
  there (new seeds, longer run) would decide it.
- How should calibration be judged under topologies where agents are not
  independent (e.g. hierarchical, where a supervisor may propagate one
  method to every worker), so that "agreeing with peers" means less?

---

## 6. We set a precision target; the agents ran far past it, evenly

**What happened.** The task says "the target is U_err <= 0.005 for each value
of T", and the prompt says "you are done when every U_err is at most the
target". The reference (`N16.json`) was built to be about 10× better than
that (error limit 5e-4), on the assumption that agents would aim near the
target. They didn't. In the 12 agent runs we had when this was written
(2026-11-16: Opus prototype, Sonnet solo, 10 Sonnet independent):

- **Every agent beat the target at every T**, by 1.3–6× at the hardest
  temperature and by 2,000–13,000× at the easiest. The worst error in the
  10-agent batch was 3.8e-3.
- **No agent sized its sample count to how hard each T was.** Eight used
  a single count for all 14 temperatures. Four Sonnet agents used two
  coarse levels, fewer samples at low T and more at high T (for example
  80k and 2M). Either way the count was sized for the hardest
  temperatures: each agent's largest count was 200k to 2M samples. Where
  errors are naturally tiny (low T), they ran far longer than needed.

*Update 2026-11-18:* the 76 accepted Haiku submissions since then (nine
batches, efforts medium to max) also beat the target at every T; the
worst error among them is 3.3e-3.
- As a result the agents' errors at low T are 10–20× *smaller* than the
  reference's (whose absolute stopping rule ended after 1,000 clusters
  there), and only 2–4× larger at high T, not the ~10× the design expected.
  This is what made the pulls in §5 reference-dominated.

**Why it probably happened.** Reading the prompt as the agent sees it:
- The target is a floor ("at most"), never a ceiling. Nothing says
  overshooting is undesirable.
- Compute is effectively free to the agent: no cost is stated, only turns
  and a per-call time limit. More samples is the safe choice against an
  "at most" criterion, especially since U_err is itself uncertain.
- One loop over T with a fixed count is the simplest plan. Sizing per T
  needs a pilot run first, and nothing in the task rewards that.

So the agents did what the task asked; the task didn't ask what we
meant. The precision measure is saturated (every agent passes), so it
can't separate agents.

**What to do.**
- Size the reference against what agents actually do, not the target:
  a large fixed budget at every T, or a relative stopping rule.
- Decide whether precision is "met the target" or "met it without wasting
  compute". If the latter, say so in the task (e.g. aim for U_err close to
  the target, and report the compute used), and record overshoot
  (target ÷ U_err) and per-T sample allocation.
- When writing a done criterion, ask what an agent that follows it to the
  letter, with free compute, will do.

**Questions to answer:**
- What are we studying: whether agents get the physics right, whether
  their error bars are honest, or how they spend effort? Each needs a
  different task statement, and the Ising model may support all three.
- Does an explicit efficiency instruction change behavior (pilot runs,
  per-T allocation), and does it cost calibration?
- Is the 2D Ising model rich enough to keep studying these questions, or
  is it time for a harder problem? (Sean's view, 2026-11-16: there is
  much more to explore with it.)

---

## 7. The host computer is part of the instrument

**What happened.** In `2026-11-16_ising_haiku_at_medium_independent_001`,
agent `ising-07` started a `run_python` call at 11:26:34 (local time; the
log says 19:26:34 UTC) with the maximum 600 s timeout. At 11:29:09 the Mac
went into idle sleep. It was plugged in, but that doesn't prevent idle sleep
(the power log says "Idle Sleep … Using AC"). From the power log
(`pmset -g log`) and the agent's log:

| local time | Mac | ising-07 |
|---|---|---|
| 11:26:34 | awake | call starts |
| 11:29:09 | idle sleep | |
| 11:37:50–11:38:30 | brief dark wake, then sleep | |
| 11:53:30 | dark wake (background only), about 40 min | |
| ≈12:00:35 | | the harness's 620 s backstop runs out |
| 12:00:45 | | cleanup kill times out after 10 s; agent ends; archive fails |
| 12:33:43 | full wake (user activity) | container finally stops; `run_error` |

The backstop counts on the monotonic clock, which on macOS stops while the
computer sleeps. Adding up only the awake and dark-wake periods, its 620 s
ran out at about 12:00:35, ten seconds before the first failure was logged.
The container never answered, even in dark wake. Apparently Docker
Desktop's virtual machine stays frozen until the Mac fully wakes, because
`docker rm` finished in the same second the screen came on.

**What the record shows.** The log goes straight from `tool_call` to
`workspace_archive_failed`, with no `execution` and no `tool_result`. The run's
verification reports three files with no log entry
(`ising-07-0002.py/.out/.err`): the hung call's files, written before the
harness died. The analysis lists `ising-07` as `not_run`, although it did
run. Nothing was tampered with and the record is honest about the gap,
but nothing in it says *why*. That had to be reconstructed from the Mac's
power log.

**Why it matters.**
- A lost agent is a lost data point, and it is lost to the host, not to the
  agent. Counted as an agent failure, it would make Haiku look worse than it
  is.
- The harness treated "the container didn't answer" as a programming error
  (an exception) rather than an event to record. One fragile cleanup call
  ended the agent.
- Every timeout in the harness assumed the computer stays awake. Durations
  in the log quietly leave out time asleep.

**What we did** (commit `f202ed8`).
- On macOS the harness holds `caffeinate -i -s` for the whole run, refuses to
  start if it can't, and records that it did (`keep_awake` in the first
  log entry). The lid must stay open.
- A cleanup kill the container doesn't answer is retried once. If it still
  fails, the execution is logged, then `sandbox_unresponsive`, and only then
  does the agent end.
- Every execution records its wall-clock duration next to the monotonic
  one. A gap of more than 30 s is flagged `host_clock_gap`.
- The terminal shows a progress line for each milestone, so a stalled run
  is visible while it happens.

**Still open.**
- Agents that ended because of the host should get their own status (e.g.
  `host_incident`), not the generic `error` or `not_run`, so that analysis
  can separate them from agent failures.
- A power cut or crash kills the harness itself, which can't log its own
  death. It leaves a run folder with no `final_hash.txt` and no index line,
  and nothing currently notices one. The harness should refuse to start
  while such a folder exists, and a small command should record it as
  `interrupted`.

**Questions to answer:**
- How should a batch that lost an agent to the host be compared with one
  that didn't: rerun the agent, or report n = 9?
- What else does the record assume about the host (disk, network, Docker
  restarts, clock changes), and which of these can it detect for itself?

---

## 8. The weaker model exercised the harness more

**What happened.** The Haiku batch ran on the same design, prompt and
limits as the 10-agent Sonnet batch. On physics the two look similar (pooled
rms pull 1.11 vs 1.02; one point beyond 3σ each), but their process
differed:

| | Haiku (10 agents¹) | Sonnet (10 agents) |
|---|---|---|
| Tokens | 640k | 155k |
| Turns per agent | 4–13 | 2–4 |
| `run_python` calls per agent | 2–12 | 1–3 |
| Calls hitting the 600 s limit | 3 | 0 |
| Rejected tool calls | 6 | 0 |

¹ Tokens are the whole run's total. The per-agent ranges cover the 9
agents that ran to the end; `ising-07` was lost to the host (§7).

The rejections were:
- three `run_python` calls with no `code` field (ising-05; in two of them
  the code ended up inside `purpose`, see §9; the third had purpose "x"
  and no code);
- two submissions missing required fields (`seeds`, `error_method`), each
  fixed on resubmission;
- one attempt to submit again after an accepted submission.

The hung call that led to §7 was also Haiku's: 8 worker processes on a
1-CPU sandbox, sized for hours of work.

**Why it matters.** Rejection logging, resubmission limits and the timeout
path were built and tested, but Sonnet never used them in a real run.
Haiku did, and they held. A model that makes more mistakes is a cheap
stress test of the harness: a twentieth of Sonnet's price per token, and
about a sixth of the Sonnet batch's cost per batch at medium effort (§3).
A lot of the harness's value lies in paths that capable models rarely
take.

**What to do.**
- Before an expensive batch (Opus), consider a Haiku batch on the same
  config as a dress rehearsal for the harness, not only as a data point.
- Report tool-use errors (rejections by type, timeouts, resubmissions) per
  model. They separate models even where the physics measures are
  saturated (§6).

**Questions to answer:**
- Does Haiku at a higher effort level make fewer tool-use errors, or just
  spend more tokens?
- Do Haiku's error bars hold up per agent under the χ² calibration test,
  given that several agents rounded low-T values to 5 decimals and
  reported `U_err = 1e-05` (about 10× the true error there)?

## 9. The tool-call format is a channel that can corrupt data silently

**What happened.** In the first Haiku-high batch
(`2026-11-16_ising_haiku_at_high_independent_001`, same design and limits
as Haiku-medium, `effort: high`), two of ten agents,
ising-03 and ising-07, ended without an accepted submission. Both had
finished, well-calibrated results in `/work`. Both lost all three
submission attempts to the same mistake. The model writes a tool call in a
markup format that the API turns into JSON. Both agents closed the
`error_method` parameter with `</error_method>` instead of `</parameter>`.
The parser read that tag as ordinary text, so the next parameter (opening
tag and value) ended up inside the `error_method` string:

```
...far below the 0.005 target.</error_method>\n<parameter name="seeds">[1507, 1607, ...]
```

Pydantic correctly rejected the call: `seeds` was missing. Its message
truncates the input it shows (`'C...rom biasing the error.'`), though, so
neither agent could see where its markup had gone wrong. Each repeated the
slip, leaked markup again on the third try, and ran out of attempts.
ising-07's final message diagnoses the cause correctly, after the fact;
ising-03's says its third attempt had correct tags, which it did not.

Searching every logged tool input in those two batches for parameter
markup inside a string field:

| | Haiku medium | Haiku high |
|---|---|---|
| `submit_result` calls with leaked markup | 3 of 12 | 10 of 20 |
| Agents affected | 2 | 5 (3 recovered) |
| Agents lost to it | 0 | 2 |

This also explains most of §8's rejections. Two of medium's three
`run_python` calls with no `code` were `</purpose>` closing the `purpose`
field and swallowing `code`, and both of its "missing field" submissions
were this slip too.

**Update 2026-11-18, all nine Haiku batches** (every effort level run
once with and once without CPU pinning, plus a repeat of high without).
A call counts as leaked if any field other than `code` contains
`<parameter …>`, `</parameter>` or a closing tag named after a field:

| Effort | Batches | `submit_result` calls leaked | `run_python` calls leaked | Agents lost to the resubmission limit |
|---|---|---|---|---|
| medium | 2 | 3 of 22 | 18 of 128 | 0 of 19² |
| high | 3 | 38 of 61 | 10 of 177 | 7 of 30 |
| xhigh | 2 | 19 of 36 | 14 of 168 | 4 of 20 |
| max | 2 | 3 of 21 | 52 of 277 | 1 of 20³ |
| Sonnet, medium | 1 | 0 of 10 | 0 of 17 | 0 of 10 |

² One medium agent was lost to the host (§7), not to a leak.
³ A second max agent (`2026-11-17_ising_haiku_at_max_independent_002`,
ising-04) ran out of token budget while repeating a `run_python` leak 25
times.

Every Haiku batch leaked. The overall rate (14–20% of all calls) doesn't
track effort, but *where* the leaks land does: in `submit_result`, and so
costing agents, at high and xhigh; mostly in `run_python`, where a
rejected call costs a turn rather than the result, at max. These are two
or three batches per level, and they differ in more than effort: CPU
pinning, and the harness itself (the first three batches, medium and both
unpinned highs, ran before streaming and the 64k response cap were added;
the other six ran after). Leaks
also persist: within an agent, a call that follows a leaked call leaks far
more often than one that follows a clean call, so calls are not
independent and per-call rates overstate the sample size.

**The silent cases.** When the swallowed parameter is required, validation
fails and the record shows it. When it is optional with a default, the
call is *accepted*. Across the nine batches six accepted calls carry
leaked markup:
- four submissions where `notes` was swallowed into `report` or kept a
  stray `</notes>` (medium ising-06, which first showed this; high pinned
  ising-07 and ising-09; xhigh ising-04);
- two `run_python` calls where `timeout_s` was swallowed into `purpose`.
  In `2026-11-18_ising_haiku_at_max_pinned_independent_001` ising-01 asked
  for 60 s, the default, so nothing changed. In
  `2026-11-18_ising_haiku_at_xhigh_pinned_independent_001` ising-02 asked
  for 300 s; the call passed validation and ran under the 60 s default.
  It finished in 1.6 s, so nothing visible happened, and the agent never
  remarked on it. (Its next call left out `timeout_s`, timed out at 60 s,
  and the agent raised the timeout to 600; that call had no slip.)
  **The agent's intent was changed without any error.**

Grading reads only `points`, so no score changed, but nothing in the runs
flagged the corrupted fields. They were found only by searching the logs.
Any optional field that changes execution or grading can be altered
silently this way.

**Why it matters.** We had treated the tool schema as a guarantee about the
data: if it validates, it is what the agent meant. It isn't. Validation
checks shape, not intent. The model's own formatting is a step between what
it means and what we record, and that step fails in a way the schema
can't always see. Extra thinking didn't remove it: it happened at every
effort level (see the table above).

The resubmission limit (3) is what turned a formatting slip into a lost
agent. It is a hyperparameter of the design, and this is the first batch
where it decided an outcome.

**What to do** (decided 2026-11-16; to apply after the Opus-medium batch, so
Opus gets the same feedback as Haiku and Sonnet):
- **Feedback is part of the instrument.** A rejection should name the
  fields received and the fields expected, without truncation, using only
  the schema. Changing what agents are told is an experimental change and
  goes in design.md.
- **Flag, don't reject.** The harness logs a `tool_input_markup` flag
  whenever any tool call has a string field containing tool-call syntax.
  The agent never sees it. Legitimate code can contain such text, so it is
  a flag, not a rejection.
- **Check old runs as analysis.** The logs hold every raw input, so the
  same check runs over finished runs without touching `runs/`.
- **Strict tool use (`strict: true`) is an arm, not a fix.** It would
  likely prevent missing-field rejections, so it would remove a failure
  mode we are measuring. If used, it is a deliberate design setting.
- **Report per model:** leaked-markup calls, and agents lost to the
  resubmission limit.

**Answers to §8's questions** (first written 2026-11-16 from one batch
per level; updated 2026-11-18):
- Higher effort did not reduce tool-use errors (table above). In the
  first comparison, Haiku-high had 12 rejected submit attempts against
  medium's 3, and used 29% more tokens (827k vs 640k) with fewer
  `run_python` calls (53 vs 60) and timeouts (1 vs 3).
- The 5-decimal rounding came back. The first high batch reported full
  precision (`U_err` down to about 4e-7, pooled rms pull 1.05 over 112
  points against medium's 1.11), but in the second high batch ising-10
  again rounded U to 5 decimals with `U_err = 1e-05`, and in an xhigh
  batch ising-08 did the same with `5e-06`. The smallest reported errors
  elsewhere are about 2e-7.

**Questions to answer:**
- Does Opus leak markup at all? (The same log check over the 10-agent
  Sonnet batch finds none in 27 tool calls.)
- Does the leak rate depend on the field's content, for example long free
  text with angle brackets (`<m^4>`)? Almost every submission contained
  such text, so these batches can't say.
- With complete rejection feedback, do agents recover on the next try?
