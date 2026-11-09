# Build Prompt: Microdrama Promo Finder

Paste this as the task prompt to your coding agent (Claude Code or similar). It assumes
`AGENTS.md` is in the repo root and will be read automatically — if your harness doesn't
auto-read it, paste its contents first.

---

## Task

Build a tool that reads microdrama screenplay files (timestamped beats of CAMERA,
ACTION, DIALOGUE, SOUND FX) and outputs ranked promo clip recommendations: 20–60 second
clips (single continuous segment or a few joined segments) that open on a hook, end on a
cliffhanger, and are understandable to a viewer who has never seen the show.

Build this as a staged pipeline, not a single LLM call. Follow `AGENTS.md` strictly —
it encodes failure modes we've already identified; don't rediscover them the hard way.

## Stage 1 — Parser

Input: raw screenplay text in the format shown in the sample episode (scene headers,
then blocks like `**CAMERA:**`, `**ACTION:**`, `**DIALOGUE/CHARACTER NAME:**`, each with
`Start:`/`End:` timestamps).

Output: structured JSON. Group overlapping events into **beats** — a beat is a time
window containing one or more simultaneous events. Schema:

```json
{
  "episode": "Episode 1",
  "beats": [
    {
      "beat_id": "B001",
      "start": 0.0,
      "end": 4.8,
      "events": [
        {"type": "camera", "description": "...", "start": 0.0, "end": 4.8},
        {"type": "action", "description": "...", "start": 1.0, "end": 4.8}
      ]
    }
  ]
}
```

Rules:
- Merge events into the same beat only if their time intervals overlap meaningfully
  (not just touching at an endpoint).
- Preserve each event's own start/end even inside a shared beat — do not collapse to
  one timestamp per beat.
- If timestamps are missing from a source file, fall back to a words-per-second
  estimate (~2.5 words/sec) to generate approximate start/end — but mark these beats
  `"timestamp_source": "estimated"` so downstream stages can discount confidence
  accordingly. Never silently treat estimated and given timestamps as equally reliable.
- Write unit tests against: missing timestamps, out-of-order blocks, dialogue with no
  preceding action, consecutive CAMERA blocks with no dialogue at all.

## Stage 2 — Candidate generation

Two independent detectors, each its own prompt + structured output:

**Hook detector.** For each beat near the start of a potential clip, score:
- Dialogue-hook signal (shocking line, reveal, threat — from DIALOGUE events)
- Visual-hook signal, i.e. Scroll-Stop Potential (unusual/surprising visual, conflict
  legible without sound, subject visually prominent, creates a question, distinct from
  a typical genre scene — from CAMERA/ACTION events)
Report both signals separately per beat. Do not merge into one "hook score" yet.

**Cliffhanger detector.** For each beat, assess whether it ends on unresolved tension.
Score as a Curiosity Gap:
- What specific question would a viewer have after this point?
- Is the question important enough to motivate continuing?
- Does the beat (or clip ending here) reveal too much, defusing its own question?
Classify the cliffhanger type: identity reveal / betrayal / threat / romantic
uncertainty / revenge / hidden secret / imminent action / unexpected arrival / moral
dilemma / unanswered revelation.

**Window search (bidirectional).**
- Forward: from each strong hook beat, search forward up to 60s for a strong
  cliffhanger beat.
- Backward: from each strong cliffhanger beat, search backward up to 60s for a strong
  hook beat.
- Union the results, dedupe identical [start_beat, end_beat] pairs.

**Multi-segment stitching.** Only propose a multi-segment candidate when a single
segment scores below a threshold on either hook or cliffhanger AND a plausible second
segment (same character/conflict thread) would clearly raise both. Generate the
stitched candidate alongside the single-segment version — don't replace it — and let
scoring decide which wins.

Candidate boundaries at this stage are **beat IDs**, not timestamps.

## Stage 3 — Scoring

Run each candidate through independent scorers, each producing a structured score
(not merged yet):

1. **Comprehension gate** (pass/fail, not a weighted score): can a first-time viewer
   identify who is being betrayed/threatened/challenged using only what's in the clip,
   without needing a character's name or prior episode knowledge? Fail → candidate is
   dropped from further ranking, logged with the specific missing context.
2. **Emotional salience** — score relative to other candidates *within the same
   episode*, not on an absolute scale (corrects for melodrama inflation in this genre).
3. **Spoiler level** (0–3): 0 = no meaningful spoiler, 1 = minor development, 2 = major
   relationship/betrayal, 3 = major twist/resolution. Attach to output; do not auto-drop.
4. **Boundary-quality pass**: generate 2–3 boundary variants per surviving candidate
   (shift start/end to neighboring beat edges), and pick the cleanest opening (hook
   within first 3s, no dead air) and cleanest ending (cuts exactly at the unresolved
   moment, doesn't accidentally include the answer) under the duration constraint.
5. For multi-episode / multi-segment candidates: a continuity check — can segment 2 be
   understood using only information available in segment 1 or earlier in the same
   promo?

## Stage 4 — Ranking

Pairwise/tournament comparison among surviving candidates (within and across episodes),
using the per-dimension scores from Stage 3 as grounding context for the comparison
prompt — not a single composite number computed by Python. The LLM judges "A vs. B,
which is the stronger promo overall," shown both candidates' dialogue/action text and
their individual scores.

## Stage 5 — Portfolio selection (MMR)

From the ranked list, select the final output set using Maximal Marginal Relevance:

```
Score(c) = λ * Quality(c) - (1 - λ) * Similarity(c, Selected)
```

- `Quality(c)`: the candidate's rank/score from Stage 4.
- `Similarity(c, Selected)`: embedding similarity between this candidate's
  hook+cliffhanger text and already-selected candidates'.
- Start λ around 0.7 (favor quality, but meaningfully penalize redundancy); report what
  you used and why.

## Stage 6 — Output

Per promo: episode(s), ordered list of `[start, end]` segment timestamps, hook text +
beat ID, cliffhanger text + beat ID, one-line rationale, spoiler level, comprehension
status, and a note on whether the hook was dialogue-driven, visual-driven, or both. The
editor should be able to verify the recommendation from this output alone, without
rewatching the episode.

## Stage 7 — Evaluation

- Build a small human-labeled set: pick 15–25 clips you'd personally call editor-grade
  across 2–3 episodes, with your own start/end timestamps.
- Compute: Hook precision@K, temporal IoU between your tool's boundaries and your human
  picks, Kendall's tau between the tool's ranking and your human ranking.
- Run an ablation: disable pairwise ranking (use raw scores instead), disable MMR,
  disable the comprehension gate — report how output quality/overlap with your human
  set changes in each case.
- Write up a failure-mode section with categorized, counted examples: false hooks
  (loud ≠ shocking), mistimed cuts, context-dependent clips that slipped past the gate,
  stitching incoherence, visual hooks missed because scoring leaned on dialogue only.

## Constraints

- Model choice: use a cheaper/faster model for Stage 2 (candidate generation, which
  runs over many beats) and reserve the strongest model for Stage 4 (pairwise ranking,
  which runs over far fewer items).
- Every LLM prompt lives in its own file under `prompts/`, not inline in Python.
- Every number in the final output must be traceable to a beat ID and a specific
  scoring call — no unexplained numbers.
- Do not implement character relationship graphs, full narrative dependency graphs, or
  emotional trajectory scoring until Stages 1–6 work end-to-end on at least one full
  episode. If time remains after that, these are documented in `SPEC.md` as optional
  extensions — implement only with time to spare, and only after the core pipeline has
  been run and evaluated.

## Deliverables

1. Working pipeline (Stages 1–6) runnable on any screenplay file in the given format.
2. Eval harness (Stage 7) with real numbers from your own human-labeled set.
3. A short architecture write-up: the 7-stage diagram, the central thesis ("a good
   scene is not automatically a good promo"), 3–5 concrete before/after examples
   showing the portfolio-diversity step actually changing the output, and the failure
   analysis with categorized counts.
