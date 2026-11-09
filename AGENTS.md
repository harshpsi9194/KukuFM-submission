# AGENTS.md — Promo Finder (Microdrama Clip Selection)

This file governs how any coding agent (Claude Code, Cursor, etc.) should work in this
repo. Read this before writing any code. Violating these rules produces output that
looks plausible but is wrong in ways that are hard to catch by eye — follow them exactly.

## 0. Project in one sentence
Given one or more screenplay files (timestamped beats of CAMERA/ACTION/DIALOGUE/SOUND),
find 20–60s promo clips (single or multi-segment) that open on a hook, end on a
cliffhanger, and are understandable to someone who has never seen the show — then rank
them as a diverse portfolio, not a top-N list of near-duplicates.

## 1. Non-negotiable architectural rules

1. **Never let one LLM call do generation + scoring + ranking.** These are three
   separate stages with separate prompts, separate output schemas, and separate code
   paths. If you catch yourself writing a prompt that asks for "the best clips," stop —
   that's the forbidden single-shot approach.
2. **Never let an LLM invent a timestamp.** The LLM selects beat IDs from the parsed
   structure. Timestamp resolution (beat ID → seconds) is pure Python, deterministic,
   and testable. If a number doesn't trace back to a beat ID in the parsed JSON, it's a
   bug.
3. **Clip boundaries snap to beat edges only.** Never produce a start/end that falls
   inside a beat's interval. Opening boundary = some beat's start. Closing boundary =
   some beat's end.
4. **Duration is a filter, not a scoring input.** Generate candidates ignoring the
   20–60s window, then filter. Don't let the LLM trade off quality for duration — that
   produces worse clips that happen to be the right length.
5. **Events within a beat can overlap in time.** Do not flatten CAMERA/ACTION/DIALOGUE
   into one sequential list per timestamp. Each beat is a container of simultaneous
   events. See `schema.md` for the exact structure. Getting this wrong silently breaks
   hook detection for every visual-only hook (e.g., a reveal shot under dialogue).
6. **Standalone comprehension is a gate, never a weighted-in score.** A clip that fails
   comprehension is disqualified outright, not averaged down with a good hook score.
   Do not merge this into a composite number.
7. **Stage 4 Ranking: Pure absolute rubric ranking (0–10) across 4 narrative dimensions, exactly 1 LLM call, ZERO round-robin matches.** Absolute scores are grounded against the explicit anchor rubric in `prompts/batched_ranking_v1.txt` with deterministic Python tie-breaking (duration closeness to 35s sweet spot, emotional salience, hook score), eliminating O(n²) pairwise tournament latency and ensuring 100% comparability within an episode in 1 call.
8. **Portfolio selection happens after ranking, as a separate MMR pass.** Do not let the
   top-N by raw quality be your final output — dedupe near-identical hooks via
   embedding similarity before finalizing.
9. **Spoiler level is reported, not auto-filtered.** Compute it (0–3), attach it to
   output, never silently drop a high-spoiler clip without surfacing why.
10. **Script-derived evidence is not verified performance.** Never phrase output as if
    you know how a line was delivered on screen. The source is a screenplay. Say so in
    any user-facing explanation of a score.

## 2. Code organization rules

- `parser/` — screenplay text → structured JSON (beats with nested events). Pure
  Python, no LLM calls, fully unit-testable. This is the highest-leverage place to get
  right; bugs here corrupt everything downstream silently.
- `candidates/` — hook detection, cliffhanger detection, bidirectional window search,
  multi-segment stitching. LLM calls live here, but boundary math does not — boundary
  math is Python, not LLM output.
- `scoring/` — all rubric-based LLM evaluation (scroll-stop potential, curiosity gap,
  comprehension gate, spoiler level, boundary-variant comparison). Each metric is its
  own function with its own prompt, independently testable and independently loggable.
- `ranking/` — pairwise comparison + MMR portfolio selection. No new "quality"
  judgments happen here — this stage only combines scores computed upstream.
- `eval/` — human-labeled eval set, IoU/Kendall's tau computation, ablation harness.
  This is not optional scaffolding — treat it as a first-class part of the system.
- `prompts/` — every LLM prompt as a separate versioned file, not inlined as Python
  strings scattered through the codebase. Any rubric change should be a diff in one
  place.

## 3. Prompt-writing rules (for prompts the agent generates)

- Every rubric prompt must include 2–3 **concrete worked examples** at different score
  levels, not just adjective descriptions ("strong," "weak"). Pull these examples from
  the actual episode 1 sample the first time you write a prompt, then generalize.
- Every scoring prompt must force **structured output** (JSON schema), never free text
  that gets parsed with regex.
- Never ask an LLM "is this a good promo" — always ask about one specific, named
  dimension (hook visual strength, curiosity gap, comprehension, spoiler level,
  boundary cleanliness) in isolation.
- Comprehension-gate prompts must ask about **situational** comprehension (can the
  viewer tell who is being betrayed/threatened), not character name recall.

## 4. Testing and validation rules

- Parser must have unit tests against malformed/missing timestamp input before any
  candidate-generation code is written against it.
- Before tuning the scoring rubric, build the human-labeled eval set (even 15–20 clips
  across 2–3 episodes is enough) — don't iterate on rubrics blind.
- Every pipeline run must log: which beats were considered, which candidates were
  generated, every score per candidate, and the MMR similarity matrix used for final
  selection. If a result can't be traced back to specific beat IDs and scores, the
  logging is insufficient — fix that before adding features.
- Report metrics by name: Hook precision@K, temporal IoU, Kendall's tau / Spearman
  correlation. Don't substitute vague language ("seems to work well") for these.

## 5. Things that will look right but are wrong — watch for these specifically

- A hook detector that only reads DIALOGUE blocks and ignores CAMERA/ACTION. It will
  work fine on dialogue-heavy episodes and silently fail on visual-only reveals.
- A cliffhanger that "sounds unresolved" but actually reveals the answer to its own
  question one line later in the same clip (over-reveal). Check the clip's own end
  state, not just the sentence pattern.
- Absolute LLM scores that look consistent within one run but aren't comparable across
  episodes — don't rank across episodes using absolute scores, only pairwise.
- A portfolio of "top 5 by score" that is actually the same beat with 5 slightly
  different boundaries. This is the MMR step's entire purpose — don't skip it to save
  time.
- Treating "no good promo in this episode" as a bug to fix rather than a valid output.
  Build the abstention path from the start, don't bolt it on later.
- Multi-episode stitches that assume knowledge from a skipped episode. Always check: is
  every fact referenced in segment 2 either shown in segment 1 or inferable without
  outside context?

## 6. Scope discipline

Build in this order and do not skip ahead: parser → candidate generation → scoring →
pairwise ranking → MMR portfolio → eval harness. Do not start on character relationship
graphs, narrative dependency graphs, or emotional trajectory scoring until the above is
working end-to-end on at least one full episode. Those are explicitly scoped-down or
deferred features — see `SPEC.md` for what's in vs. out.
