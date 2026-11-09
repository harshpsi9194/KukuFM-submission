"""
mmr.py — Stage 5: Maximal Marginal Relevance (MMR) Portfolio Selection.

Per AGENTS.md Rule 8:
"Portfolio selection happens after ranking, as a separate MMR pass. Do not let the
top-N by raw quality be your final output — dedupe near-identical hooks via
embedding similarity before finalizing."

Design:
1. Narrative Signatures:
   Embeds functional promo drivers (Hook Dialogue + Visual Rationale, Cliffhanger
   Specific Question + Type, Situational Conflict Summary), explicitly excluding
   raw script line dumps (avoids artificial dialogue overlap) and crude thread labels
   (avoids type-collapsing).
2. Semantic Embeddings:
   Computed using BAAI/bge-small-en-v1.5 (384-dimensional dense semantic vectors,
   L2-normalized) via fastembed.
3. Hard Deduplication Constraint (Option B1):
   Disqualifies any candidate with Cosine Similarity >= max_sim_threshold (0.90)
   against any already selected promo. This strictly blocks near-identical hooks
   (e.g. CAND_EP001_52 at 0.9630 similarity to CAND_EP001_47).
4. MMR Objective Function:
   MMR(c) = λ * Q(c) - (1 - λ) * max_{s in S} Sim(c, s)
   where Q(c) = (N + 1 - Rank(c)) / N is the normalized Stage 4 absolute rubric rank quality.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def build_narrative_signature(
    candidate: Dict[str, Any],
    scored_candidate: Dict[str, Any],
) -> str:
    """Build structured narrative text representation for dense embedding."""
    hook = candidate.get("hook_signal", {})
    cliff = candidate.get("cliffhanger_signal", {})
    comp = scored_candidate.get("comprehension", {})

    hook_dial = hook.get("dialogue_hook_rationale", "")
    hook_vis = hook.get("visual_hook_rationale", "")
    cliff_q = cliff.get("specific_question", "")
    cliff_type = cliff.get("cliffhanger_type", "")
    sit_sum = comp.get("situational_summary", "")

    return (
        f"Hook: {hook_dial} {hook_vis}\n"
        f"Cliffhanger: {cliff_q} ({cliff_type})\n"
        f"Situational Summary: {sit_sum}"
    )


def compute_embeddings(
    signatures: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> np.ndarray:
    """Compute normalized dense embeddings using fastembed."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        venv_site = root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if venv_site.exists() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))
        from fastembed import TextEmbedding

    model = TextEmbedding(model_name)
    embs = np.array(list(model.embed(signatures)), dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return embs / norms


def compute_cosine_similarity_matrix(norm_embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix from normalized embeddings."""
    return np.dot(norm_embeddings, norm_embeddings.T)


def select_mmr_portfolio(
    rankings: List[Dict[str, Any]],
    sim_matrix: np.ndarray,
    k: int = 5,
    lambda_param: float = 0.55,
    max_sim_threshold: float = 0.90,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Select diverse promo portfolio using MMR with hard deduplication.

    Args:
        rankings: Ordered list of Stage 4 ranking dicts.
        sim_matrix: Pre-computed cosine similarity matrix.
        k: Number of promos to select.
        lambda_param: Quality vs diversity trade-off parameter.
        max_sim_threshold: Hard similarity cutoff above which candidates are
                           flagged as near-duplicates and disqualified.

    Returns:
        selected_portfolio: List of selected promo dictionaries with audit metadata.
        disqualification_log: List of all disqualified candidate events.
    """
    n_candidates = len(rankings)
    ranks = [r.get("rank", i + 1) for i, r in enumerate(rankings)]
    denom = max(1.0, float(n_candidates))
    quality_scores = [(denom + 1.0 - r) / denom for r in ranks]
    cand_ids = [r["candidate_id"] for r in rankings]

    pool = list(range(n_candidates))
    selected: List[int] = []
    portfolio: List[Dict[str, Any]] = []
    disqualification_log: List[Dict[str, Any]] = []

    # Pick 1: Highest quality candidate (Rank #1)
    best_initial = int(np.argmax(quality_scores))
    selected.append(best_initial)
    pool.remove(best_initial)

    r0 = rankings[best_initial]
    boundary_0 = r0.get("recommended_boundary", f"{r0.get('start_beat_id')} -> {r0.get('end_beat_id')}")
    portfolio.append({
        "selection_order": 1,
        "candidate_id": r0["candidate_id"],
        "rank": r0.get("rank", 1),
        "recommended_boundary": boundary_0,
        "narrative_thread": r0.get("narrative_thread", ""),
        "duration": r0.get("duration", 0.0),
        "quality_score": round(quality_scores[best_initial], 4),
        "max_similarity_to_selected": 0.0,
        "most_similar_selected_id": None,
        "mmr_score": round(lambda_param * quality_scores[best_initial], 4),
        "selection_rationale": "Rank #1 Absolute Rubric Leader."
    })

    # Iterative MMR selection for picks 2 to k
    for step in range(2, k + 1):
        best_score = -1e9
        best_idx = None
        best_sim = None
        best_sim_source = None

        for c in pool:
            sims = [(sim_matrix[c, s], cand_ids[s]) for s in selected]
            max_s, max_src = max(sims, key=lambda x: x[0])

            boundary_c = rankings[c].get("recommended_boundary", f"{rankings[c].get('start_beat_id')} -> {rankings[c].get('end_beat_id')}")
            if max_s >= max_sim_threshold:
                disqualification_log.append({
                    "step": step,
                    "candidate_id": cand_ids[c],
                    "rank": rankings[c].get("rank", c + 1),
                    "recommended_boundary": boundary_c,
                    "similarity": round(float(max_s), 4),
                    "duplicate_of": max_src,
                    "reason": f"Exceeded near-duplicate threshold ({max_s:.4f} >= {max_sim_threshold:.2f})"
                })
                continue

            q = quality_scores[c]
            score = lambda_param * q - (1.0 - lambda_param) * max_s
            if score > best_score:
                best_score = score
                best_idx = c
                best_sim = max_s
                best_sim_source = max_src

        if best_idx is None:
            # Fallback if threshold eliminates all remaining candidates
            break

        selected.append(best_idx)
        pool.remove(best_idx)
        r_chosen = rankings[best_idx]
        boundary_chosen = r_chosen.get("recommended_boundary", f"{r_chosen.get('start_beat_id')} -> {r_chosen.get('end_beat_id')}")

        portfolio.append({
            "selection_order": step,
            "candidate_id": r_chosen["candidate_id"],
            "rank": r_chosen.get("rank", step),
            "recommended_boundary": boundary_chosen,
            "narrative_thread": r_chosen.get("narrative_thread", ""),
            "duration": r_chosen.get("duration", 0.0),
            "quality_score": round(quality_scores[best_idx], 4),
            "max_similarity_to_selected": round(float(best_sim), 4),
            "most_similar_selected_id": best_sim_source,
            "mmr_score": round(float(best_score), 4),
            "selection_rationale": (
                f"Selected at step {step}: Q={quality_scores[best_idx]:.4f}, "
                f"MaxSim={best_sim:.4f} (to {best_sim_source}), MMR={best_score:+.4f}."
            )
        })

    return portfolio, disqualification_log

