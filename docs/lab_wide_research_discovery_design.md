# Lab-Wide Condensed-Matter Research Discovery: Design and Feasibility Report

**Status:** Design proposal — no implementation performed.
**Repository:** `arxiv-digest` v0.4.0 (`src/arxiv_digest/`, ~4,100 LOC)
**Date:** 2026-08-26
**Scope:** Response to `new_requirements.md`. Audits the current system, then proposes an architecture for expanding it from a single-researcher tool into a lab-wide research intelligence system for a condensed-matter group of ~5–30 members, running on consumer hardware.

---

## 1. Executive Summary

The existing system is **substantially better engineered than most tools of this kind** — strict Pydantic models, deterministic ranking, atomic artifact writes, injectable I/O, `mypy --strict`, a real test suite, and a working CI pipeline. The redesign should therefore be an *extension* of this codebase, not a rewrite. The provider-Protocol pattern already in `summarization.py` is, as it happens, exactly the right seam for local LLM integration.

The audit found four concrete defects that matter far more than any missing feature, and all four are measurable.

**Finding 1 — Retrieval has no category constraint at all.** `arxiv_client.build_search_query()` emits `all:"term"` with no `cat:` clause, so every query searches all of arXiv. This is the root cause of the monopole problem. Measured against the live API: `all:"magnetic monopole"` returns 1,830 papers, of which only 448 (24%) are in `cond-mat.*` while 458 (25%) are `hep-ph`. Adding a single `AND cat:cond-mat.*` clause removes 76% of that noise **at the source**, before any scoring runs.

**Finding 2 — The score floor lets irrelevant papers into the digest.** Ranking weights are normalized to sum to 1.0, and `novelty` is `1.0` for any paper not previously recommended, contributing a flat `0.10`. `recency` contributes up to another `0.10`. A completely irrelevant paper submitted late in the window therefore scores **≥ 0.20 from relevance-free components alone** — which equals `adjacent_min_score` (0.20) and comfortably exceeds `wildcard_min_score` (0.12). This is not hypothetical. In the repository's own committed run, `data/selection-2026-07-21-2026-07-28.json` rank 10 is *"Dark Monopoles, Bounds on Hidden Sectors, and Cosmological Implications"* (`hep-ph`, keyword score 0.00, category score 0.00) and rank 5 is an X-ray Thomson scattering plasma paper (`physics.plasm-ph`). The reported failure mode is reproducible in checked-in data.

**Finding 3 — Keyword relevance is structurally crushed.** `ranking.keyword_relevance()` divides by the sum of *all* configured term weights (currently 35.0). A paper matching the single highest-weighted term, `spin ice` (3.0), in its title scores `3.0/35 = 0.086`. Observed keyword scores across the stored run span 0.00–0.09. Weighted at 0.25, keyword evidence contributes at most ~0.02 to a final score — less than the constant novelty bonus. The strongest available relevance signal is the weakest contributor.

**Finding 4 — Material-name matching silently fails on real arXiv text.** `ranking.tokenize()` splits on `[^\W_]+`, so `Ho$_2$Ti$_2$O$_7$` — how arXiv actually renders it — tokenizes to `('ho','2','ti','2','o','7')` and never matches the configured token `ho2ti2o7`. The two highest-weighted materials in `config/research_profile.yaml` (`Ho2Ti2O7` and `ZnFe2O4`, 3.0 each) are dead weight that *also* inflates the denominator in Finding 3, suppressing every other term. Verified directly against the repo's own tokenizer; the repo's own stored data confirms the premise, containing the title `Stacking-dependent anisotropic altermagnetism in V$_{1/3}$NbS$_2$`.

Separately: **"semantic relevance" is not semantic.** `ranking.semantic_relevance_scores()` is corpus-local TF-IDF cosine similarity — pure lexical overlap. It carries the largest weight (0.45) but cannot distinguish *"emergent magnetic monopoles in spin ice"* from *"primordial magnetic monopoles"*, because it has no notion of meaning. Embeddings are genuinely needed here, and `summarization.py` already defines an unused `EmbeddingProvider` Protocol for them.

**The recommendation is Option B, the hybrid pipeline** — deterministic gating, then embeddings, then a small local LLM used only where it earns its cost — sequenced so that *filtering quality is fixed before profiles are added*. The ordering matters: adding 10–30 researcher profiles on top of a broken domain filter multiplies the false positives by the number of profiles instead of dividing the work.

One measurement is load-bearing for the whole design: **arXiv publishes only ~517 `cond-mat.*` papers per week** (measured for 2026-08-17 → 2026-08-24). That is small enough to abandon keyword-driven retrieval entirely and instead ingest the *complete* weekly cond-mat firehose, ranking locally. This eliminates a whole class of problem — query-term coverage bias — and makes recall independent of whether anyone remembered to add the right search term. It also fits the specified hardware comfortably: ~500 abstracts embed in well under a minute on either GPU.

---

## 2. Audit of the Existing Implementation

Answering the ten questions from `new_requirements.md` §10 directly.

### 2.1 How papers are currently discovered

`pipeline.retrieve_candidates()` → `ArxivClient.fetch()` (`src/arxiv_digest/arxiv_client.py:246`). For each of the 6 query groups in `config/research_profile.yaml`, two streams are issued:

1. `build_search_query()` — terms OR'd together, `AND submittedDate:[start TO end]`, sorted by `submittedDate`
2. `build_term_query()` — the same terms with **no date bound**, sorted by `lastUpdatedDate`, to catch revisions

Results are pooled, filtered to the window by explicit timestamp comparison, deduplicated via `normalization.deduplicate_papers()`, and truncated to `max_candidate_count` (200).

The client itself is well built: 3 s pacing (`_pace()`), bounded exponential-backoff retries on transient status codes, in-process response caching, a contact email in the User-Agent, and a scan cap that prevents runaway pagination.

### 2.2 Which APIs/data sources are used

**Only the official arXiv Atom API** (`https://export.arxiv.org/api/query`), parsed with `feedparser`. No Semantic Scholar, OpenAlex, Crossref, INSPIRE, or journal feeds. No PDF retrieval — the system is deliberately abstract-only, and `models.ABSTRACT_SUMMARY_BASIS` is enforced by validator so summaries cannot claim otherwise. That discipline is worth preserving.

OpenAI is an optional summary provider (`summarization.OpenAISummaryProvider`), off by default (`config/app.yaml` sets `provider: offline`).

### 2.3 How keywords are currently represented

`config.ResearchProfile` (`src/arxiv_digest/config.py:229`), loaded from `config/research_profile.yaml`:

| Field | Type | Role |
|---|---|---|
| `priority_topics` | `list[str]` | free text; only used in the TF-IDF profile document |
| `exact_phrases` | `dict[str, float]` | weighted phrases |
| `materials` | `dict[str, float]` | weighted material names |
| `methods` | `dict[str, float]` | weighted method names |
| `categories` | `dict[str, float]` | weighted arXiv categories |
| `negative_terms` | `list[str]` | unweighted penalty terms |
| `queries` | `list[QueryConfig]` | retrieval terms, *maintained separately from* the scoring terms |

The scoring code (`ranking._positive_terms()`) merges `exact_phrases`, `materials`, and `methods` into one flat dict by `max()` weight, discarding the facet distinction entirely. **The three-way split is presentational only** — it has no effect on ranking. This matters for §4: the schema already gestures at a facet structure without using it.

Note also that `queries` and the scoring terms drift independently, so a term can be scored but never retrieved, or retrieved but never scored. `negative_terms` currently holds only `financial market` and `social network` — nothing that would catch cosmology or collider physics.

### 2.4 What filtering currently exists

Four mechanisms, all weak against the stated problem:

1. **Window filtering** (`arxiv_client.fetch()`) — timestamp bounds. Correct and reliable.
2. **Deduplication** (`normalization.deduplicate_papers()`) — by canonical arXiv ID, then by punctuation-insensitive title key, with deterministic winner selection in `_preferred_paper()`. Well done.
3. **History exclusion** (`history.filter_recent_history()`) — suppresses papers recommended within `history_exclusion_days` (90), with an `allow_updated_resurfacing` escape for genuine new versions. Solid.
4. **Score thresholds** (`selection.recommendation_type()`) — `direct ≥ 0.28`, `adjacent ≥ 0.20`, `wildcard ≥ 0.12`; below 0.12 is rejected.

**There is no category filter, no domain classifier, and no hard exclusion rule anywhere in the pipeline.** Category information influences only `ranking.category_relevance()`, a soft 0.10-weighted term. An `astro-ph.CO` paper receives `category_relevance = 0.0` and loses at most 0.10 — which, per Finding 2, the flat novelty bonus hands straight back.

### 2.5 How ranking works

`ranking.rank_papers()` (`src/arxiv_digest/ranking.py:161`) computes a fixed five-component weighted sum, weights normalized to 1.0 by `config.RankingWeights.normalize()`:

```
final = 0.45·semantic + 0.25·keyword + 0.10·category + 0.10·recency + 0.10·novelty
        − min(0.5, 0.20 · |negative_term_matches|)
```

Component by component:

- **`semantic_relevance`** — `_tfidf_matrix()` over the candidate corpus plus a synthetic profile document, then cosine similarity. Lexical, not semantic. Also **corpus-dependent**: IDF is computed over whichever 18–200 papers happened to be retrieved, so the same paper scores differently in different weeks. That breaks score comparability across runs, which in turn destabilizes the fixed thresholds in §2.4.
- **`keyword_relevance`** — see Finding 3.
- **`category_relevance`** — `max(matched weight) / max(configured weight)`; 1.0 for anything touching `cond-mat.str-el`, 0.0 outside the configured set. Binary in practice.
- **`recency`** — linear position within the retrieval window. Rewards papers for being submitted on a Friday.
- **`novelty`** — `0.5` if previously recommended, else `1.0`. This is a *repeat-suppression flag*, not novelty in any research sense, and it is the source of the score floor.
- **`feedback_affinity`** — hardcoded `0.5` at `ranking.py:176` and **never used in the score**. `RankingWeights` has no field for it; the weight named `feedback_or_novelty` multiplies `novelty` alone. It is a placeholder propagated into every artifact.

Tie-breaking is deterministic (score, then `updated_at`, then ID). Good practice; keep it.

### 2.6 Whether embeddings are used

**No.** `summarization.EmbeddingProvider` (Protocol) and `summarization.OpenAIEmbeddingProvider` exist and are unit-tested (`tests/test_summarization.py:131`), and `SummarizationConfig.openai_embedding_model` is a valid config key — but nothing in `ranking.py`, `selection.py`, or `pipeline.py` ever calls `.embed()`. The interface is built and wired to nothing. Convenient: the seam already exists.

### 2.7 Whether LLM integration exists

**Yes, and it is well designed.** `summarization.OpenAISummaryProvider` uses the OpenAI Responses API with structured outputs (`responses.parse(..., text_format=PaperSummary)`), bounded validation retries, and a strict prompt forbidding claims beyond the abstract. `models.PaperSummary` enforces this with word-count validators, an exact-match `summary_basis` check, and `reject_inspection_claims()`, a regex validator that rejects phrasing implying the PDF was read.

`summarize_selection()` degrades gracefully: a provider failure falls back to `DeterministicSummaryProvider` per paper, never dropping a paper, and counts fallbacks for observability.

**This architecture transfers to local models with almost no change.** A `LocalLLMSummaryProvider` satisfying the same Protocol, backed by Ollama's `format` parameter (JSON Schema → GBNF grammar), reuses `PaperSummary` verbatim as both output schema and validation gate.

### 2.8 How metadata/history is stored

Flat files, no database:

- `data/candidates-*.json`, `data/ranked-*.json`, `data/selection-*.json` — per-window Pydantic-validated snapshots
- `data/history.jsonl` — append-only `HistoryRecord` lines
- `reports/*.{md,html,json}` plus `reports/latest.*`
- All writes atomic via temp-file-then-`replace()` (`pipeline.write_model()`, `reporting._write_atomic()`)
- `history.append_history()` is idempotent and rejects run-ID/window collisions

The GitHub Actions workflow (`.github/workflows/weekly_digest.yml`) commits `data/history.jsonl` and dated report JSON back to the branch through a strict path allowlist. Careful work.

**Scaling limit:** history is keyed by `arxiv_id` alone, with no researcher dimension. There is no persistent per-paper store — full `Paper` records live only inside per-run snapshots, so "have we ever seen this paper?" requires scanning every snapshot file.

### 2.9 How easy it is to support multiple researcher profiles

**Currently: not possible without structural change.** `config.Settings` holds exactly one `profile: ResearchProfile`. `load_settings()` takes a single `profile_path`. Every downstream function — `rank_papers()`, `select_diverse()`, `topic_key()`, `summarize_selection()`, `build_digest_artifact()` — takes one `ResearchProfile`. `DigestArtifact` carries scalar `profile_name`, `profile_version`, `profile_hash`.

The naive workaround — run the pipeline N times with N profile files — fails badly:

- **N× the arXiv traffic**, at 3 s/request pacing, over heavily overlapping result sets
- **N separate history files**, or one file with no way to tell whose history a record is
- **N× the summarization cost**, re-summarizing the same paper per researcher
- No group view, no cross-researcher deduplication, no way to say "3 people match this paper"

The fix is not "loop over profiles" but **"separate the corpus from the audience"**: fetch and score papers once against a group model, then compute cheap per-researcher affinities on top. §5 and §11 develop this.

### 2.10 Where the architecture fails when scaling to a lab

| # | Failure mode | Cause | Severity |
|---|---|---|---|
| 1 | Off-domain papers reach the digest | No category filter at retrieval (§2.4); score floor ≥ 0.20 (Finding 2) | **Critical** |
| 2 | Recall depends on query-term coverage | Only papers matching one of 6 hand-written query groups are ever seen | **Critical** |
| 3 | Cannot represent multiple researchers | Single `ResearchProfile` throughout (§2.9) | **Critical** |
| 4 | Scores not comparable across runs | Corpus-local TF-IDF IDF (§2.5) makes fixed thresholds drift | High |
| 5 | Keyword signal too weak to matter | Sum-of-all-weights denominator (Finding 3) | High |
| 6 | Material matching silently broken | LaTeX subscripts vs. tokenizer (Finding 4) | High |
| 7 | Cannot answer "who is this for?" | No researcher dimension in scores or history | High |
| 8 | Per-user summarization cost scales linearly | Summaries regenerated per profile | Medium |
| 9 | Feedback cannot be recorded | `feedback_affinity` is a hardcoded constant (§2.5) | Medium |
| 10 | One digest, one recipient list | `DeliveryConfig.to_emails` is a flat list; no subscribers, no personalization | Medium |
| 11 | `topic_key` creates a bucket for the noise | `selection.topic_key()` returns `category:hep-ph` when no query matches | Medium |
| 12 | No persistent paper store | Full records live only in per-run snapshots (§2.8) | Medium |

Failure 11 deserves emphasis. Because `topic_key()` assigns off-domain papers their own topic bucket, the `max_papers_per_topic` diversity cap actively *protects* them from being crowded out by better cond-mat papers. In the stored run, the `hep-ph` monopole paper sat alone in bucket `category:hep-ph` and took the wildcard slot. **The diversity mechanism amplified the filtering bug** — a good illustration of why §6 argues for gating before ranking rather than tuning weights.

---

## 3. Requirements Derived from the Goals

Numbered for traceability to later sections.

### Functional

| ID | Requirement | Source |
|---|---|---|
| F1 | Represent 5–30 researchers with individual interest profiles | §1 of requirements |
| F2 | Maintain a group-level interest profile distinct from the union of individuals | §1 |
| F3 | Accept lightweight input (5–15 free-form keywords) and expand it into a richer profile | §8 |
| F4 | Produce a weekly group digest: Highly Recommended / By Research Area / Cross-Disciplinary | §2 |
| F5 | Per paper: title, authors, arXiv info, date, URL/DOI, summary, why-relevant, matched interests/members, relevance score, novelty score | §2 |
| F6 | Reject astrophysics/HEP/cosmology false positives while retaining condensed-matter emergent monopoles | §5 |
| F7 | Run routine processing on local LLMs without paid API calls | §3 |
| F8 | Composite ranking, not a single LLM score | §6 |
| F9 | Diversity control so the top 10 is not ten near-identical papers | §6 |
| F10 | Record lightweight feedback (👍/👎/save/more-like-this) and let it influence ranking | §9 |
| F11 | Generate HTML and text digests; maintain a subscriber list; support eventual scheduled delivery | §2 |
| F12 | Eventually answer natural-language questions over the collected paper database | §3 |

### Non-functional

| ID | Requirement | Rationale |
|---|---|---|
| N1 | Full weekly run completes in < 30 min on the RTX 4050 laptop | Practicality on stated hardware |
| N2 | Peak VRAM ≤ 5.5 GB on the laptop (6 GB card, display overhead) | §4 of requirements |
| N3 | Every ranking decision explainable in terms a physicist can audit | "understandable and easy to debug" |
| N4 | Determinism preserved wherever the current system has it | Existing repo discipline; reproducible run IDs |
| N5 | Adding a researcher requires no code change | Lab usability |
| N6 | Degrade gracefully when the local LLM is unavailable | Mirrors existing fallback design |
| N7 | No new heavyweight infrastructure unless corpus size justifies it | §7 of requirements |
| N8 | Preserve the abstract-only evidence discipline | Existing `ABSTRACT_SUMMARY_BASIS` invariant |

### Explicit non-goals for this phase

- No email is sent, and no mailing system is implemented (requirements §2). Note that **SMTP delivery already exists** in `delivery.py` and the `email` job of the CI workflow — §12 therefore describes what would need to change, not what to build now.
- No PDF full-text ingestion.
- No authenticated web application.

---

## 4. Recommended Lab/Researcher Interest Representation

### 4.1 Evaluating the "each member gives keywords" proposal

The instinct is right, but plain keywords alone are insufficient — for three reasons the audit makes concrete.

1. **Keywords have no domain anchor.** `monopole` is the canonical case. A keyword list cannot express "this word, in this scientific sense". That has to come from context, category, or a classifier (§6).
2. **Keywords conflate different kinds of thing.** "Neutron scattering" (a technique), "pyrochlore" (a material class), "spin ice" (a phenomenon), and "Monte Carlo" (a method) behave differently in matching. A paper that matches two *techniques* and no topic is usually less relevant than one matching a topic plus a material. Flat keyword lists cannot express that, and §2.3 shows the current code flattens the facets it does have.
3. **Lexical matching is brittle** — Finding 4 is the proof.

But the *input* should still be keywords. The resolution is to **separate the input format from the internal representation**: members type free-form keywords; the system expands them into a structured profile, which they then review.

### 4.2 Recommended: two-layer representation

**Layer 1 — Authored profile (what a human writes and reviews).**

```yaml
# profiles/researchers/a_okafor.yaml
id: a_okafor
name: Dr. Amara Okafor
email: a.okafor@example.edu
active: true

# The only REQUIRED field. 5-15 free-form keywords.
interests:
  - frustrated magnetism
  - spin ice
  - pyrochlore
  - neutron scattering
  - magnetic monopoles
  - Monte Carlo simulation
  - rare-earth magnets

# Everything below is OPTIONAL. Generated by profile expansion,
# then edited by the researcher.
facets:
  primary_topics:    [frustrated magnetism, spin ice]
  secondary_topics:  [magnetic excitations, spin liquids]
  materials:         [pyrochlore, "Ho2Ti2O7", "Dy2Ti2O7", rare-earth titanates]
  techniques:        [inelastic neutron scattering, diffuse scattering, magnetometry]
  methods:           [classical Monte Carlo, loop algorithms, spin-wave theory]
  ai_ml:             []

exclude:
  - cosmological monopoles
  - collider searches

weight: 1.0          # relative voice in the group profile
digest_opt_in: true
```

**Layer 2 — Compiled profile (what the pipeline uses; generated, never hand-edited).**

```python
@dataclass(frozen=True)
class CompiledProfile:
    id: str
    centroid: NDArray[np.float32]           # mean of facet embeddings, L2-normalized
    facet_vectors: dict[str, NDArray]       # per-facet centroids for explainability
    lexical_terms: dict[str, float]         # expanded surface forms -> weight
    exclusion_vectors: list[NDArray]        # embedded exclusion phrases
    source_hash: str                        # hash of the authored profile
```

Compilation is a separate, cached step (`arxiv-digest profiles compile`). It runs when a profile changes, not per digest. Deterministic given the same embedding model and authored file, satisfying **N4**.

### 4.3 Why facets earn their complexity

Facets are not decoration; they do three specific jobs:

1. **Facet-aware matching.** A paper matching *primary_topics* is more relevant than one matching only *techniques*. Weighting facets separately (§10) is the cheapest way to encode that, and it is a single dot product per facet.
2. **Explainability (N3).** "Matched your *materials* (pyrochlore) and *techniques* (diffuse neutron scattering)" is auditable. "Cosine similarity 0.71" is not.
3. **Disambiguation.** The `materials` and `techniques` facets are precisely the condensed-matter context signals that resolve `monopole` (§6.3). Facets are how the positive-context mechanism gets its vocabulary for free.

Facets should be **generated, not demanded**. Requirement N5 and the user's "keep it lightweight" constraint are met by making `interests` the only mandatory field.

### 4.4 Profile expansion (keywords → facets)

This is a genuinely good use of a local LLM: one-off, low-volume, human-reviewed, and structured. For 30 researchers it is 30 calls, once.

```
expand_profile(interests: list[str]) -> facets
  1. LLM call, grammar-constrained to the facets schema:
       "Classify each term into: primary_topics, secondary_topics,
        materials, techniques, methods, ai_ml. For each term add at most
        3 standard synonyms or surface variants used in condensed-matter
        literature. Do not invent research interests not implied by the input."
  2. Deterministic post-processing:
       - Expand chemical formulae into LaTeX/Unicode/plain variants (Finding 4)
       - Attach known arXiv categories per facet from a static mapping
       - Drop any generated term whose embedding cosine to the nearest
         input keyword is < 0.5   (hallucination guard)
  3. Write to the profile YAML under `facets:` with a
     `# generated, please review` header.
  4. Researcher reviews and edits. Their edits are never overwritten:
     re-running expansion only fills empty facets unless --force.
```

Step 2's third clause matters: it bounds the LLM's contribution to terms that are *semantically close to something the human actually wrote*, which turns an unreliable generator into a reliable-enough suggester. Step 4 keeps the human in authority.

The formula-variant expansion in step 2 is the direct fix for Finding 4:

```python
def formula_variants(formula: str) -> list[str]:
    """Ho2Ti2O7 -> [Ho2Ti2O7, Ho$_2$Ti$_2$O$_7$, Ho₂Ti₂O₇, Ho 2 Ti 2 O 7]"""
```

with matching done against a subscript-normalized form of the paper text rather than against raw tokens.

### 4.5 Group profile: computed, but with a manual override

**Yes, maintain both** (F2). The group profile is not merely the union of individuals — the union is too broad, and a paper interesting to everybody slightly is often more valuable to a group digest than one fascinating to exactly one person.

```yaml
# profiles/group.yaml
name: Condensed Matter Group
description: >-
  Experimental and computational condensed-matter physics: quantum materials,
  magnetism and frustrated magnetism, neutron and Raman scattering, transport,
  crystal growth, thin films, spintronics, correlated electrons, and
  machine-learning methods applied to materials.

# The domain gate. Deliberately broad — this defines "is this our field at all",
# not "is this interesting". See section 6.
domain:
  include_categories: [cond-mat.*, physics.ins-det, physics.comp-ph, physics.app-ph, mtrl-th]
  soft_categories:    [quant-ph, physics.optics, physics.chem-ph, cs.LG, stat.ML]
  exclude_categories: [astro-ph.*, hep-ph, hep-th, hep-ex, hep-lat, gr-qc, nucl-th, nucl-ex, math.*, q-bio.*, q-fin.*, econ.*]

# Areas used for the "By Research Area" digest sections (F4).
research_areas:
  - {id: magnetism,   label: "Magnetism & Frustrated Systems"}
  - {id: neutron,     label: "Neutron & X-ray Scattering"}
  - {id: quantum_mat, label: "Quantum Materials & Correlated Electrons"}
  - {id: spintronics, label: "Spintronics & Transport"}
  - {id: synthesis,   label: "Crystal Growth & Thin Films"}
  - {id: computation, label: "Computational Methods"}
  - {id: ml_physics,  label: "ML/AI for Condensed Matter"}

# Optional: shared interests not owned by any one member.
group_interests: [quantum materials, materials characterization]

aggregation:
  member_weighting: equal      # equal | by_weight | seniority
  breadth_bonus: 0.15          # reward for matching many members (section 10)
```

The group centroid is the weighted mean of member centroids, plus embedded `group_interests`, plus embedded `description`. The `domain` block is deliberately *not* derived from members — it is a stable statement of the field, so a single member's adjacent interest cannot silently widen the domain gate for everyone. That separation is what stops one person's `cs.LG` interest from admitting the whole of machine learning.

The `research_areas` list defines both the digest sections (F4) and the diversity quotas (§10.4). Each area gets a centroid embedded from its label plus the facet terms of members who match it.

---

## 5. Proposed Paper-Discovery Architecture

### 5.1 The central change: firehose ingestion, not keyword search

The current design asks arXiv "give me papers matching my terms" and gets back an unbounded, category-blind result set. The proposed design asks "give me this week's condensed-matter papers" and does all the relevance work locally.

Measured weekly volumes justify this:

| Query | Result |
|---|---|
| `cat:cond-mat.*` submitted 2026-08-17 → 2026-08-24 | **517 papers** |
| `cat:cond-mat.*` all time | 412,368 papers |

517 papers/week is ~26,000/year. Embedding 517 abstracts takes well under a minute on either GPU. Storage for a full year of metadata plus 1024-d float16 embeddings is on the order of 100 MB. **The entire relevant literature is small enough to hold locally.**

The benefits are structural rather than incremental:

- **Recall stops depending on query wording.** Today, a spin-ice paper that never uses the phrase "spin ice" is invisible. Under firehose ingestion it is retrieved and then judged semantically.
- **Adding a researcher requires no new queries** — satisfies N5 directly.
- **Scores become corpus-stable.** Ranking against a fixed group model rather than corpus-local IDF fixes failure mode 4.
- **arXiv load drops.** ~11 paged requests/week at 100/page, versus 12 query streams paging independently.

Keyword queries do not disappear entirely — they remain useful as a **supplementary recall channel** for cross-listed papers whose primary category sits outside cond-mat but which are genuinely condensed-matter work (a `quant-ph` paper on superconducting qubit materials, say). Those are fetched with an explicit category guard and then subjected to the same gate.

### 5.2 Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 0  INGEST                                    ~517 papers/wk │
│  arXiv API: cat:cond-mat.* AND submittedDate:[window]            │
│  + supplementary: (soft categories) AND (group lexical terms)    │
│  → dedupe (existing normalization.deduplicate_papers)            │
│  → upsert into papers store                                      │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 1  HARD GATE (deterministic, no ML)          ~517 → ~500    │
│  • drop if primary_category in exclude_categories AND             │
│    no cond-mat cross-list                                         │
│  • drop if history says "recommended < N days ago"                │
│  • drop if title/abstract hits a hard-exclusion rule (6.4)        │
│  Cost: microseconds. Fully auditable. Logged with reason.         │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 2  EMBED + DOMAIN SCORE                      ~500 papers    │
│  • embed(title + abstract) → cache by arxiv_id+version            │
│  • domain_score = f(category evidence, group centroid sim,        │
│                     context terms, exclusion sim)   [see 6.5]     │
│  • drop if domain_score < hard_floor                              │
│  Cost: ~30-60 s on RTX 4050.                                      │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 3  RELEVANCE SCORING (vectorized, no LLM)    ~350 papers    │
│  • group_similarity     = cos(paper, group_centroid)              │
│  • researcher_affinity  = max/top-k over member centroids         │
│  • area_assignment      = argmax over research-area centroids     │
│  • lexical_score        = facet-weighted term matching            │
│  • recency, novelty, feedback  (section 10)                       │
│  → composite score; keep top ~60 for LLM stages                   │
│  Cost: a few matrix multiplies. Milliseconds.                     │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 4  LOCAL LLM RERANK (borderline band only)   ~40 papers     │
│  Only papers in the uncertainty band get an LLM opinion.          │
│  Confident accepts and confident rejects skip this entirely.      │
│  Cross-encoder rerank (Qwen3-Reranker-0.6B) or small-LLM judge.   │
│  Cost: ~1-3 min.                                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 5  SELECT: diversity, quotas, MMR            ~30 papers     │
│  Highly Recommended (10) + per-area sections + cross-disciplinary │
│  (extends existing selection.select_diverse)                      │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 6  SUMMARIZE (local LLM, selected papers only) ~30 papers   │
│  PaperSummary via grammar-constrained JSON. Reuses existing       │
│  Pydantic model and validators unchanged.                         │
│  Cost: ~5-10 min on RTX 4050 with a 4B model.                     │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 7  RENDER: group digest + per-member sections               │
│  Markdown / HTML / JSON. Delivery deferred (section 12).          │
└──────────────────────────────────────────────────────────────────┘
```

The funnel shape is the point: **517 → 500 → 350 → 60 → 40 → 30**. Expensive stages only ever see a small, pre-filtered set. The LLM is asked ~70 questions per week, not 517.

### 5.3 Assessment of the architecture proposed in the requirements

The requirements sketch this chain:

```
Sources → Metadata/category filter → Hard exclusion → Broad CM classifier
       → Embedding similarity → Local-LLM rerank → Novelty/diversity → List
```

**This is essentially correct**, and the proposal above follows it. Four refinements:

1. **Move the category filter into the query itself**, not just post-retrieval. `AND cat:cond-mat.*` cuts the monopole result set by 76% before a single byte is scored (§6.2). Filtering at the source is strictly better than filtering after.
2. **Merge "broad CM classifier" with embedding similarity.** They use the same embedding pass; running them as separate stages doubles the cost for no benefit. Compute the embedding once, derive both the domain score and the relevance score from it (§6.5).
3. **Make LLM reranking conditional, not universal.** Reranking all ~350 survivors on a 6 GB laptop is the difference between a 3-minute and a 40-minute run. Only the uncertainty band needs an LLM (§8.3).
4. **Add a persistent paper store before Stage 2.** Novelty detection, deduplication against past weeks, and later RAG (F12) all require a corpus that outlives one run. Without it, "is this an incremental follow-up?" is unanswerable.

One caution on ordering: **novelty/diversity must come after LLM reranking, not before** — otherwise diversity quotas are computed over a candidate set the reranker is about to reorder, and you get the failure already visible in §2.10 item 11, where diversity protects a bad paper.

---

## 6. Strategy for Preventing Astrophysics/Particle-Physics False Positives

This is the highest-value fix in the report, so it is treated in the most detail. The layered design below is justified by direct measurement against the live arXiv API.

### 6.1 The problem, quantified

| Query | Total | In `cond-mat.*` | Fraction |
|---|---|---|---|
| `all:"magnetic monopole"` | 1,830 | 448 | **24%** |
| `all:"spin ice"` | 931 | 921 | **99%** |

Breakdown for `magnetic monopole`: `hep-ph` 458 (25%), `astro-ph.*` 148 (8%), `cond-mat.str-el` 214.

Two things follow. First, the leakage is real and large. Second — and more useful — **ambiguity is a per-term property, not a global one.** `spin ice` is safe; `magnetic monopole` is not. A single global exclusion policy is the wrong shape. This motivates §6.6.

### 6.2 Layer 1 — Category constraints at query time (highest value, lowest cost)

The arXiv API supports the `cat:` prefix and `ANDNOT`, with wildcards. Measured effectiveness of each option:

| Strategy | Query | Results | Reduction |
|---|---|---|---|
| Baseline | `all:"magnetic monopole"` | 1,830 | — |
| **Positive guard** | `all:"magnetic monopole" AND cat:cond-mat.*` | **448** | **76%** |
| Grouped exclusion | `all:"magnetic monopole" ANDNOT (cat:hep-ph OR cat:hep-th OR cat:astro-ph.*)` | 743 | 59% |
| Single exclusion | `all:"magnetic monopole" ANDNOT cat:hep-ph` | 1,372 | 25% |

**The positive guard is decisively better than exclusion**, and it is also more robust: it does not require enumerating every irrelevant archive, and it will not be defeated by a new archive appearing.

An important implementation warning surfaced by testing. **Chained `ANDNOT` clauses do not compose reliably:**

```
all:"magnetic monopole" ANDNOT cat:hep-ph                       -> 1372   (1830-458 exactly)
all:"magnetic monopole" ANDNOT cat:hep-ph ANDNOT cat:astro-ph.CO -> 1424   (INCREASED)
all:"magnetic monopole" ANDNOT (cat:hep-ph OR cat:astro-ph.CO)   -> 1356   (consistent)
```

Adding a second chained `ANDNOT` *increased* the result count — the parser does not associate as expected. **Always group negations in parentheses.** This is exactly the sort of thing that would silently degrade a filter for months, so it belongs in a regression test.

Recommended query shape:

```python
def build_search_query(query, window, domain) -> str:
    terms = " OR ".join(f'all:"{escape(t)}"' for t in query.terms)
    include = " OR ".join(f"cat:{c}" for c in domain.include_categories)
    exclude = " OR ".join(f"cat:{c}" for c in domain.exclude_categories)
    return (
        f"({terms}) AND ({include}) "
        f"ANDNOT ({exclude}) "          # single grouped negation, never chained
        f"AND submittedDate:[{start} TO {end}]"
    )
```

For the firehose channel (§5.1) the term group is omitted entirely.

**One caveat that must not be skipped.** Category filtering alone would discard genuinely relevant cross-listed work — a `quant-ph` primary paper on superconducting qubit materials, or a `cs.LG` paper introducing an ML potential for magnetic materials. Hence `soft_categories` in the group profile: papers whose primary category is soft are admitted to Stage 1 but must earn their place through the semantic gate rather than being auto-accepted. Category filtering is a strong prior, not a verdict.

### 6.3 Layer 2 — Positive context requirements for ambiguous terms

Category filtering does not solve everything. An `astro-ph.CO` paper cross-listed to `cond-mat.stat-mech` for its numerical methods will pass a category check. So ambiguous terms carry a context requirement:

```yaml
# config/disambiguation.yaml
ambiguous_terms:
  monopole:
    requires_any_of:                    # >=1 must appear in title/abstract
      [spin ice, pyrochlore, emergent, frustrated, magnetic moment,
       Dirac string, spinon, Coulomb phase, dumbbell model,
       neutron scattering, condensed matter, lattice]
    blocked_by_any_of:                  # any of these => reject outright
      [primordial, cosmological, inflation, early universe, GUT,
       't Hooft-Polyakov, collider, dark matter, cosmic ray, relic abundance]
    default_when_neither: penalize      # accept | penalize | reject

  frustration:
    requires_any_of: [magnetic, lattice, spin, antiferromagnet, geometrical]
    blocked_by_any_of: [protein folding, optimization landscape]

  transport:
    requires_any_of: [electrical, thermal, charge, spin, conductivity, resistivity, Seebeck]
    blocked_by_any_of: [public transport, supply chain, neutrino]

  phase transition:
    requires_any_of: [order parameter, critical exponent, magnetic, structural, quantum critical]
    blocked_by_any_of: [electroweak, QCD, cosmological, deconfinement]
```

The `default_when_neither: penalize` setting matters. Rejecting on absence of positive context is too aggressive — abstracts are short and an author may simply not use the expected vocabulary. Penalizing and letting Layer 4 decide is the calibrated choice.

### 6.4 Layer 3 — Hard exclusion rules (narrow, high-precision, auditable)

A small set of rules applied at Stage 1, each of which must be individually defensible and individually testable:

```yaml
hard_exclusions:
  - id: cosmological-monopole
    when_all:
      - {field: categories, none_of: [cond-mat.*]}
      - {field: text, any_of: [primordial, cosmological, early universe, inflation]}
      - {field: text, any_of: [monopole, defect, domain wall]}
    reason: "Cosmological topological defects, not condensed-matter excitations"

  - id: collider-physics
    when_all:
      - {field: categories, none_of: [cond-mat.*]}
      - {field: text, any_of: [LHC, ATLAS, CMS, collider, luminosity, GeV, TeV]}
    reason: "High-energy collider physics"

  - id: pure-astro
    when_all:
      - {field: primary_category, any_of: [astro-ph.CO, astro-ph.GA, astro-ph.SR, astro-ph.HE, astro-ph.EP]}
      - {field: categories, none_of: [cond-mat.*]}
    reason: "Astrophysics with no condensed-matter cross-listing"
```

The `none_of: [cond-mat.*]` guard on every rule is the safety catch: **a paper cross-listed to cond-mat is never hard-rejected**, no matter what words it contains. That is what preserves *"magnetic monopole excitations in spin ice"* while rejecting *"primordial magnetic monopoles in cosmology"*. Every rejection is logged with its rule `id` and `reason`, so a mis-rejection is diagnosable rather than mysterious (N3).

Keep this list short. Hard rules are the layer most likely to cause silent recall loss; the semantic layers should do the nuanced work.

### 6.5 Layer 4 — Semantic domain classification

After the cheap layers, compute a domain score in [0, 1] combining four signals:

```python
def domain_score(paper, group, embeddings) -> tuple[float, dict[str, float]]:
    # 1. Category evidence — strong prior
    if any_match(paper.categories, group.domain.include_categories):
        cat = 1.0
    elif any_match(paper.categories, group.domain.soft_categories):
        cat = 0.5
    else:
        cat = 0.0

    # 2. Semantic similarity to the group's domain description
    sim = cosine(embeddings[paper.id], group.domain_centroid)      # ~0.2-0.8

    # 3. Positive condensed-matter context density
    ctx = min(1.0, matched_context_terms(paper) / 3.0)

    # 4. Similarity to known off-domain exemplars (negative evidence)
    off = max(cosine(embeddings[paper.id], v) for v in group.offdomain_centroids)

    score = 0.40*cat + 0.35*normalize(sim) + 0.15*ctx - 0.30*max(0.0, off - 0.45)
    return clamp(score, 0.0, 1.0), {"cat": cat, "sim": sim, "ctx": ctx, "off": off}
```

`offdomain_centroids` are embeddings of short prototype descriptions — "primordial magnetic monopoles and topological defects in the early universe", "searches for magnetic monopoles at particle colliders", "black hole thermodynamics and holography" — a handful of sentences, no training required. The `max(0.0, off - 0.45)` form means off-domain similarity only bites once it is clearly above baseline, avoiding a constant penalty on everything.

Papers below `hard_floor` (start at 0.35, calibrate per §19) never reach ranking. Crucially, **the domain score is a gate, not a ranking term.** Mixing it into the composite score is what produced Finding 2: a soft penalty can always be outvoted by other components. A gate cannot.

### 6.6 Layer 5 — Measured per-term ambiguity (a cheap, distinctive addition)

The arXiv API can measure a term's ambiguity directly, offline, once:

```python
def term_ambiguity(term: str) -> float:
    """Fraction of arXiv hits for `term` that fall outside cond-mat.
    0.0 = unambiguous (spin ice), 1.0 = never condensed matter."""
    total    = arxiv_count(f'all:"{term}"')
    in_scope = arxiv_count(f'all:"{term}" AND cat:cond-mat.*')
    return 1.0 - (in_scope / total) if total else 0.0
```

Measured: `spin ice` → 0.01, `magnetic monopole` → 0.76.

Run this once per term when a profile is compiled (two API calls per term, cached indefinitely). Then:

- terms with ambiguity < 0.15 are trusted for lexical matching
- terms above 0.40 automatically require a context guard (§6.3) and get their lexical weight scaled by `(1 - ambiguity)`
- the compile step **warns the researcher**: *"'magnetic monopole' matches 1,830 arXiv papers, only 24% in condensed matter. A context requirement has been added automatically."*

This turns disambiguation from a hand-maintained list into a measured property, and it scales: when a new lab member adds `skyrmion` or `Majorana` or `holography`, the system flags the risk without anyone having to anticipate it. For a 30-person lab this is the difference between a config file someone must curate forever and one that maintains itself.

### 6.7 Layered summary

| Layer | Mechanism | Cost | Catches | Risk |
|---|---|---|---|---|
| 0 | Query-time `cat:` guard | Free | ~76% of monopole noise | Loses uncross-listed relevant work |
| 1 | Hard exclusion rules | Microseconds | Explicit cosmology/collider | Over-broad rules cut recall |
| 2 | Context requirements | Microseconds | Ambiguous terms in wrong sense | Short abstracts lack context |
| 3 | Semantic domain score | ~1 ms/paper after embedding | Subtle off-domain work | Needs calibration |
| 4 | LLM adjudication (band only) | ~2 s/paper | Genuinely hard cases | Slow; use sparingly |
| 5 | Measured term ambiguity | 2 API calls/term, cached | Future unanticipated terms | Needs occasional refresh |

No single layer is sufficient; each catches what the previous one structurally cannot. Layers 0–2 are deterministic and testable, which is where the reliability comes from — layers 3–4 handle the residue.

**Worked example.**

*"Magnetic monopole excitations in the spin ice Ho₂Ti₂O₇ probed by neutron scattering"* — `cond-mat.str-el`.
L0 passes (cond-mat). L1: no rule fires (cond-mat guard). L2: `monopole` present, context `spin ice`/`neutron scattering` present → pass. L3: cat=1.0, sim high, ctx=1.0, off low → domain_score ≈ 0.92. **Accepted, ranked highly.**

*"Primordial magnetic monopoles and the cosmological constant problem"* — `hep-ph`, cross-listed `astro-ph.CO`.
L0 excluded by query guard. Had it arrived by another channel: L1 rule `cosmological-monopole` fires (not cond-mat, `primordial`, `monopole`) → **rejected with a logged reason.**

*"Machine learning interatomic potentials for magnetic materials"* — `cs.LG` primary, cross-listed `cond-mat.mtrl-sci`.
L0: soft category, admitted. L1: no rule. L2: n/a. L3: cat=1.0 (cond-mat cross-list), sim high → **accepted.** This is the case that pure category filtering would have lost, and the reason `soft_categories` exists.

---

## 7. Embedding Strategy

### 7.1 Should embeddings be central? Yes.

Four of the twelve functional requirements (F3, F6, F9, F12) are not satisfiable without them, and the audit's core weakness — a lexical score labelled "semantic" and carrying the highest weight — is precisely an embedding-shaped hole. The `EmbeddingProvider` Protocol already in `summarization.py:43` is the seam.

Embeddings serve, in rough order of value:

1. **Researcher/group matching** (F1, F2) — the core relevance signal
2. **Domain classification** (F6) — §6.5
3. **Diversity via MMR** (F9) — real semantic redundancy, not shared vocabulary
4. **Near-duplicate and follow-up detection** — cosine > 0.95 against the corpus
5. **Novelty detection** — distance from the historical centroid
6. **Clustering** for the "By Research Area" sections
7. **Semantic search / RAG** over the archive (F12)

### 7.2 Model recommendations

Two credible strategies. Given the hardware constraints, they are complementary rather than competing.

**Recommended default: `Qwen3-Embedding-0.6B`**

| Property | Value |
|---|---|
| Parameters | ~600M |
| VRAM (FP16) | ~1.2 GB; ~0.7 GB at Q8 |
| Dimensions | 1024 (supports Matryoshka truncation) |
| Context | 32k tokens — an entire abstract fits with room to spare |
| Licence | Apache 2.0 |
| Notable | Instruction-aware: the query prompt can be tailored per task |

The instruction-awareness is the deciding feature. The same model can produce a *domain* embedding ("Represent this paper for condensed-matter domain classification") and a *topical* embedding ("Represent this paper for matching against research interests") from one set of weights. That directly serves §6.5 and §10 without a second model in VRAM. At 1.2 GB it leaves ~4.3 GB free on the laptop for a generation model, which is what makes the two-model pipeline in §9.4 fit at all.

**Strong domain-specific alternative: PhysBERT**

Pre-trained on ~1.2M arXiv physics papers, BERT-base scale (~110M, ~0.4 GB), reported to outperform general-purpose models on physics-specific tasks. Its 512-token limit is a non-issue for abstracts. Two attractions here: it is tiny, and its training distribution *is* this corpus.

Also worth benchmarking: **SPECTER2** (citation-contrastive, purpose-built for academic retrieval, ~110M) and **SciNCL**, which are the standard scientific-document embeddings. Their citation-based objective encodes "papers a physicist would consider related", which is closer to the actual task than generic semantic similarity.

**Recommendation: benchmark, don't guess.** §19 specifies the experiment. The prior is that a general 0.6B model with instructions beats a 110M domain model on nuanced interest matching, while the domain model may win on the narrow "is this condensed matter?" classification. If so, use both — PhysBERT costs 0.4 GB and can run on CPU.

Not recommended here: `Qwen3-Embedding-4B/8B` (2.5–8 GB crowds out the generation model for a marginal gain on abstracts), and general-purpose small models like `EmbeddingGemma` (308M) or `BGE-M3` (567M), which are fine but have no scientific-text advantage over the above.

### 7.3 Storage: no vector database

**Recommendation: NumPy arrays in a single `.npy` file, plus SQLite for metadata. Nothing more.**

The arithmetic settles it. At 26,000 papers/year and 1024-d float32:

```
26,000 × 1024 × 4 bytes = 106 MB/year   (53 MB at float16)
```

Five years of accumulated literature is ~500 MB — it fits in RAM on a 16 GB laptop with room to spare. A brute-force cosine search over the full matrix is a single `numpy` matmul:

```python
scores = embeddings @ query_vector          # (26000, 1024) @ (1024,)
top_k = np.argpartition(-scores, k)[:k]     # ~5 ms for a year of papers
```

Approximate nearest-neighbour indexing exists to avoid O(n) search when n is in the millions. At n = 26,000, exact search is already faster than the index build, and it is exactly correct, trivially debuggable, and adds no dependency. FAISS, Chroma, Qdrant, and LanceDB would each add operational surface for no measurable gain — a direct violation of N7.

**Revisit only if** the corpus exceeds ~500k vectors (≈20 years of cond-mat) or sub-10 ms latency over multiple concurrent users becomes a requirement. If that day comes, `sqlite-vec` is the smallest step up, because the metadata is already in SQLite.

```
data/
  papers.db              SQLite: papers, categories, scores, feedback, history
  embeddings.npy         float16 matrix, row order = papers.embedding_row
  embeddings_index.json  {arxiv_id: row} plus the model name and dimension
```

Embeddings are cached by `(arxiv_id, version, model_name)`, so re-running a week costs nothing and changing embedding models triggers a clean rebuild rather than silently mixing vector spaces.

---

## 8. Local LLM Roles Within the Pipeline

### 8.1 Assignment principle

The requirements are explicit: *"I do not want to use an LLM where conventional code or embeddings would be faster and more reliable."* The rule applied throughout is —

> **Use the cheapest mechanism that is reliable for the task. Escalate only where the cheaper mechanism demonstrably fails, and only for the papers where it fails.**

This is not merely about cost. Deterministic code is testable and debuggable (N3); an LLM in a hot path is neither.

### 8.2 Task assignment table

| Task | Mechanism | Why |
|---|---|---|
| Category filtering | **Deterministic** | arXiv categories are curated metadata. An LLM here is strictly worse. |
| Deduplication (exact) | **Deterministic** | Already implemented and correct (`normalization.py`) |
| Recency, history exclusion | **Deterministic** | Arithmetic |
| Hard exclusion rules | **Deterministic** | Must be auditable and testable |
| Term ambiguity measurement | **Deterministic** (API counts) | §6.6; measurement beats judgement |
| Near-duplicate detection | **Embeddings** | Cosine > 0.95; an LLM adds nothing |
| Interest matching | **Embeddings** | The core use case; ~1 ms/paper |
| Domain relevance (bulk) | **Embeddings** + rules | §6.5; 500 papers/run makes an LLM impractical |
| Research-area assignment | **Embeddings** (argmax over area centroids) | Deterministic given fixed centroids; free |
| Clustering for diversity | **Embeddings** | MMR / agglomerative |
| Novelty detection | **Embeddings** + history | Distance from historical centroid |
| **Borderline domain calls** | **Small local LLM (1–4B)** | Only the uncertainty band, ~40 papers/run |
| **Reranking top candidates** | **Cross-encoder or small LLM** | Genuine gain over bi-encoder cosine |
| **Concept/entity extraction** | **Small local LLM (3–4B)** | Structured output; materials, techniques, phenomena |
| **Paper summaries** | **Local LLM (7–9B preferred)** | Quality is user-visible; ~30 papers/run |
| **"Why relevant to our group"** | **Local LLM (7–9B)** | Requires reasoning over profile + paper |
| Profile expansion | **Local LLM (7–9B), one-off** | 30 calls total, human-reviewed |
| Query expansion | **Local LLM, one-off, cached** | Not per-run |
| Digest assembly | **Deterministic templating** | Jinja2 already does this well |
| NL Q&A over corpus (F12) | **Local LLM + RAG** | Phase 6 |
| Anything requiring frontier reasoning | **External API, opt-in** | Rare; keep the existing OpenAI provider as an escape hatch |

### 8.3 The uncertainty band — where the LLM actually earns its cost

The single most important efficiency decision. Rather than reranking everything:

```python
CONFIDENT_ACCEPT = 0.72
CONFIDENT_REJECT = 0.38

def needs_llm_adjudication(paper, scores) -> bool:
    if scores.domain < CONFIDENT_REJECT:      return False   # clearly off-domain
    if scores.composite > CONFIDENT_ACCEPT:   return False   # clearly relevant
    if paper.hit_ambiguous_term_without_context: return True  # section 6.3
    if scores.researcher_max > 0.6 and scores.domain < 0.6: return True  # conflict
    return CONFIDENT_REJECT <= scores.composite <= CONFIDENT_ACCEPT
```

Expected effect: ~40 of ~350 papers reach the LLM (~11%). At ~2 s each that is ~80 s, versus ~12 minutes for all 350. On a 6 GB laptop this is the difference between a usable tool and one nobody runs.

The band edges should be **calibrated from labelled data**, not guessed (§19.1). Set them where the embedding score's precision/recall actually degrades.

### 8.4 Prompt patterns

All local LLM calls use grammar-constrained JSON (Ollama `format` parameter → GBNF), which makes malformed output mechanically impossible. Throughput cost is ~5–15%; the reliability gain is total. Pydantic still validates afterwards — the grammar guarantees *shape*, not *sense*.

**Domain adjudication** (small model, ~1–4B):

```
system: You classify physics papers by field. Answer only from the title,
        abstract, and arXiv categories provided.
user:   Title: {title}
        Categories: {categories}
        Abstract: {abstract}

        Is this substantially condensed-matter or materials physics research?
        Emergent quasiparticle excitations in solids (magnetic monopoles in
        spin ice, spinons, skyrmions) ARE condensed matter. Fundamental
        particles, cosmology, and collider physics are NOT.

schema: {"is_condensed_matter": bool,
         "confidence": float,
         "subfield": str|null,
         "reason": str}     # <= 25 words
```

The one-line disambiguation clause in the prompt does most of the work; it directly encodes the distinction the requirements ask for.

**Relevance explanation** (7–9B model, selected papers only):

```
system: Explain relevance using only the provided abstract and profile.
        Never claim to have read the full paper. If the connection is weak,
        say so plainly.
user:   PAPER: {title} / {abstract}
        GROUP INTERESTS: {group_description}
        MATCHED MEMBERS: {member_names_and_matched_facets}
schema: PaperSummary            # existing Pydantic model, unchanged
```

Reusing `PaperSummary` means every existing validator — the 35-word takeaway ceiling, the exact `summary_basis` match, `reject_inspection_claims()` — applies to local output automatically. The safety properties the repo already has are inherited, not rebuilt. That is the strongest argument for keeping the existing provider abstraction.

### 8.5 What local LLMs should *not* do here

- **Score papers numerically.** Small models are poorly calibrated; asking for "relevance: 0.0–1.0" yields clustered, unstable values. Ask for a *decision plus reason*, or use a cross-encoder trained for ranking.
- **Be the sole relevance signal.** Explicitly ruled out by the requirements, and correctly so.
- **Extract structured metadata that arXiv already provides.** Authors, categories, and dates come from the API. Never re-derive them.
- **Process every candidate.** §8.3.

---

## 9. Local Model Recommendations for the Specified Hardware

### 9.1 Hardware reality check

| | Laptop (Dell XPS 15) | Desktop |
|---|---|---|
| GPU | RTX 4050 Laptop | RTX 4060 (desktop) |
| **VRAM** | **6 GB** GDDR6 | **8 GB** GDDR6 |
| Memory bus | 96-bit @ 16 Gbps | 128-bit @ 18 Gbps |
| **Bandwidth** | **~192 GB/s** | **~288 GB/s** |
| TGP | 35–115 W (configurable) | 115 W |
| System RAM | 16 GB | 32 GB |

Two points are commonly underestimated and drive everything below.

**The laptop has 6 GB, not 8.** The RTX 4050 Laptop GPU ships with 6 GB on a 96-bit bus. Windows plus a display consumes roughly 0.5–1 GB, so the practical budget is **~5.0–5.5 GB** (requirement N2). This rules out comfortable 8B inference alongside an embedding model.

**Bandwidth, not compute, sets the token rate.** Decode is memory-bound: `tokens/s ≈ bandwidth ÷ model_bytes × ~0.75`. The laptop's 192 GB/s is only 67% of the desktop's 288 GB/s, and its configurable TGP means a thin-chassis XPS 15 may sustain well below peak under sustained load. Expect the low end of every range below on the laptop.

### 9.2 Candidate models (current as of August 2026)

| Model | Params | Q4_K_M size | Laptop (6 GB) | Desktop (8 GB) | Est. tok/s (lap/desk) | Best for |
|---|---|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | 4B | ~2.5 GB | ✅ Comfortable | ✅ Comfortable | ~45 / ~65 | **Classification, extraction, rerank** |
| **Gemma 4 E4B** | ~4B eff. | ~2.8 GB | ✅ Comfortable | ✅ Comfortable | ~40 / ~60 | Alternative small; strong instruction-following |
| **Qwen3-8B** | 8B | ~4.9 GB | ⚠️ Tight | ✅ Good | ~28 / ~42 | **Summaries, explanations** |
| **Granite 4.2 8B** | 8B | ~4.9 GB | ⚠️ Tight | ✅ Good | ~28 / ~42 | Enterprise-tuned; strong structured output |
| **Ministral-3 8B** | 8B | ~4.9 GB | ⚠️ Tight | ✅ Good | ~30 / ~45 | Edge-optimized |
| **Gemma 4 12B Unified** | 12B | ~7.3 GB | ❌ | ⚠️ Very tight | — / ~25 | Only with short context |
| **Qwen3-14B** | 14B | ~8.6 GB | ❌ | ❌ (Q3 only) | — / ~18 | Not recommended |
| **Gemma 4 26B-A4B** | 26B MoE (3.8B active) | ~15 GB | ❌ | ⚠️ CPU offload | — / ~15–20 | Interesting; needs 32 GB RAM |
| **Qwen3-Embedding-0.6B** | 0.6B | ~0.7 GB (Q8) | ✅ | ✅ | 500+ abstracts/min | **Embeddings** |
| **Qwen3-Reranker-0.6B** | 0.6B | ~0.7 GB | ✅ | ✅ | ~50 pairs/s | **Cross-encoder rerank** |
| **PhysBERT** | 110M | ~0.4 GB | ✅ (CPU fine) | ✅ | very fast | Physics-domain embeddings |

Sizes are approximate GGUF Q4_K_M weights and exclude KV cache. Budget an extra ~0.5–1.5 GB for context at 4–8k tokens.

Notes on specific choices:

- **Qwen3-4B-Instruct-2507** is the workhorse. It is one of the few 4B-class models with a genuinely refreshed instruct tune, supports 128k context (irrelevant here, but harmless), and handles grammar-constrained JSON reliably. At 2.5 GB it coexists with the embedding model on the laptop.
- **Gemma 4 26B-A4B** (MoE, 3.8B active) is tempting on paper: MoE means only ~3.8B parameters are active per token, so it is fast for its quality. But the *full* ~15 GB of weights must be resident somewhere. On the 32 GB desktop with `--cpu-moe` offloading, it is feasible and worth benchmarking for summary quality. On the 16 GB laptop it is not.
- **12–14B dense models are not recommended.** On 8 GB they require Q3 or aggressive offload, and Q3 degrades exactly the structured-output reliability this pipeline depends on. The requirements asked whether ~12–14B is realistic "with aggressive quantization": the honest answer is that a Q4 8B will outperform a Q3 14B on these tasks, and be twice as fast.

### 9.3 Runtime recommendation: Ollama

| Runtime | Verdict |
|---|---|
| **Ollama** | **Recommended.** Trivial install on Windows, model management, HTTP API, native JSON-Schema structured output (→ GBNF), automatic VRAM/CPU layer splitting, keeps models warm between calls. |
| llama.cpp | The engine underneath Ollama. Use directly only for fine-grained control over offload; costs build complexity on Windows. |
| LM Studio | Good GUI for interactive exploration. Less suited to scripted pipelines. |
| Transformers | Too slow unquantized; too much VRAM. Useful for PhysBERT/SPECTER2 embeddings via `sentence-transformers`. |
| vLLM | **Not appropriate.** Built for high-concurrency serving; poor Windows support; PagedAttention gains need far more VRAM. This workload is a weekly batch of ~70 calls. |

Decisive factor: Ollama's `format` parameter accepts a JSON Schema directly and compiles it to a GBNF grammar internally, so `PaperSummary.model_json_schema()` can be passed straight through. The existing structured-output design ports with minimal change.

For embeddings, `sentence-transformers` in-process is preferable to Ollama's embedding endpoint — batching control matters when embedding 500 abstracts, and PhysBERT/SPECTER2 are only available that way.

### 9.4 Recommended per-stage assignment

**Yes, use different models per stage** — the requirements ask, and the answer is clearly yes. Sequential loading keeps peak VRAM low.

**Laptop (6 GB) — the binding constraint:**

```
Resident throughout:  Qwen3-Embedding-0.6B (Q8)          ~0.7 GB
Stage 2-4:            Qwen3-4B-Instruct-2507 (Q4_K_M)    ~2.5 GB + ~0.5 KV
                                                   peak  ~3.7 GB   ✅
Stage 6 (unload 4B, load 8B):
                      Qwen3-8B (Q4_K_M)                  ~4.9 GB
                      (embedding model unloaded first)
                                                   peak  ~5.4 GB   ⚠️ tight but viable
```

Alternative for the laptop, if 8B proves unstable under memory pressure: use **Qwen3-4B for summaries too**. Quality drops noticeably on the "why relevant" reasoning, less so on plain summarization. A reasonable compromise is 4B everywhere on the laptop, 8B on the desktop, with the desktop as the primary weekly runner.

**Desktop (8 GB) — comfortable:**

```
Resident:             Qwen3-Embedding-0.6B (Q8)          ~0.7 GB
Stages 2-5:           Qwen3-4B-Instruct (Q4_K_M)         ~2.5 GB
                                                   peak  ~3.7 GB
Stage 6:              Qwen3-8B (Q4_K_M), embed unloaded  ~4.9 GB + 1.0 KV
                                                   peak  ~5.9 GB   ✅
```

Both fit. The desktop can additionally keep a larger context window, which matters if summaries later include related-work context from the corpus.

### 9.5 Estimated weekly runtime

RTX 4050 laptop, ~517 candidates:

| Stage | Work | Estimate |
|---|---|---|
| 0. Ingest | ~11 paged arXiv requests @ 3 s pacing | ~1 min |
| 1. Hard gate | Pure Python over 517 records | < 1 s |
| 2. Embed | 500 abstracts, 0.6B model, batched | ~1 min |
| 3. Score | NumPy matmuls | < 1 s |
| 4. LLM rerank | ~40 papers × ~2 s (4B) | ~1.5 min |
| 5. Select | MMR + quotas | < 1 s |
| 6. Summarize | ~30 papers × ~20 s (8B, ~250 tok out) | ~10 min |
| 7. Render | Jinja2 | < 1 s |
| **Total** | | **~14 min** |

Comfortably inside N1's 30-minute budget, with headroom for the laptop running at reduced TGP. On the desktop, expect ~8–9 minutes. Summarization dominates, which is the right place for the cost to sit — it is the only stage whose output the reader sees directly.

---

## 10. Ranking and Diversity Strategy

### 10.1 Assessment of the proposed formula

The requirements propose:

```
FinalScore = w1·TopicSimilarity + w2·ResearcherSimilarity + w3·Recency
           + w4·SourceQuality + w5·Novelty + w6·LLMRelevance
           + w7·GroupBreadth − w8·ExclusionPenalty
```

The shape is sound — a composite of independent signals is right, and much better than a single LLM score. Four changes are needed, three of them lessons from the audit.

**(a) Exclusion must be a gate, not a subtracted term.** This is Finding 2 restated. Any penalty inside a weighted sum can be outvoted by the other terms; the current `− min(0.5, 0.20·n)` penalty demonstrably fails to keep `hep-ph` papers out. Domain exclusion belongs in Stage 1–2 as a hard filter (§6). Drop `w8·ExclusionPenalty` entirely.

**(b) Remove `SourceQuality`.** Everything comes from arXiv, so it is a constant. If a journal-reference signal is wanted later, treat it as a small bonus, not a scoring dimension. Adding a term that is the same for every paper only dilutes the others.

**(c) `Recency` should be a mild tiebreaker, not a scoring component.** All candidates already come from a one-week window (§5.1), so recency spans at most 7 days and mostly encodes day-of-week. The current linear-position formula gives a Friday paper a 0.10 advantage over a Monday paper of equal merit — larger than the entire realistic range of the keyword score (Finding 3). Cap its influence sharply.

**(d) `Novelty` must mean novelty.** The current implementation is a repeat-suppression flag that creates the score floor. Replace it with a corpus-relative measure (§10.3) and, critically, **remove the constant bonus for unseen papers** — repeat suppression is history's job (`filter_recent_history()` already does it well).

### 10.2 Proposed composite score

```python
def composite_score(paper, scores, weights) -> float:
    # --- GATES (binary, before any arithmetic) ---
    if scores.domain < DOMAIN_FLOOR:        return 0.0   # section 6
    if scores.hard_excluded:                return 0.0
    if scores.duplicate_of is not None:     return 0.0

    # --- RELEVANCE (dominant, ~75%) ---
    relevance = (
        0.35 * scores.group_similarity        # cos(paper, group centroid)
      + 0.30 * scores.researcher_affinity     # softmax-weighted top-k members
      + 0.10 * scores.lexical                 # facet-weighted term match
    )

    # --- QUALITY / INTEREST MODIFIERS (~25%) ---
    modifiers = (
        0.10 * scores.llm_relevance           # 0.5 neutral when not adjudicated
      + 0.08 * scores.novelty
      + 0.07 * scores.group_breadth
    )

    base = relevance + modifiers               # in [0, 1]

    # --- MULTIPLICATIVE ADJUSTMENTS (cannot manufacture relevance) ---
    base *= (0.95 + 0.05 * scores.recency)     # <= 5% swing
    base *= (1.0 + 0.15 * scores.feedback)     # feedback in [-1, 1], section 13
    return clamp(base, 0.0, 1.0)
```

Two structural properties are deliberate and both address audit findings.

**Multiplicative modifiers cannot create relevance.** Recency and feedback scale an existing score rather than adding to it. A paper with `relevance ≈ 0` stays near 0 no matter how recent it is. This is the direct structural fix for Finding 2 — there is no longer any floor.

**`llm_relevance` defaults to 0.5 (neutral) when the paper was not adjudicated** (§8.3), so skipping the LLM neither rewards nor punishes a paper. This preserves comparability between adjudicated and non-adjudicated papers — without it, the uncertainty-band optimization would systematically disadvantage exactly the papers it skipped.

### 10.3 Component definitions

**`group_similarity`** — cosine between the paper embedding and the group centroid, rescaled. Raw cosines from a good embedding model cluster in a narrow band (~0.25–0.75 for in-domain text), so rescale against the *current run's* distribution using robust percentiles:

```python
p10, p90 = np.percentile(all_similarities, [10, 90])
normalized = np.clip((sim - p10) / (p90 - p10 + 1e-9), 0, 1)
```

This restores dynamic range. Note it reintroduces a mild corpus dependence (failure mode 4), so **persist the percentiles per run** in the artifact and use a trailing 8-week median for threshold decisions. That keeps the thresholds stable while the scores stay discriminative.

**`researcher_affinity`** — not a plain max, which would let one enthusiastic member dominate, and not a mean, which would wash out specialists. Use a softmax-weighted top-k:

```python
def researcher_affinity(paper_vec, members, k=3, temperature=0.1):
    sims = np.array([cos(paper_vec, m.centroid) for m in members])
    top = np.argsort(-sims)[:k]
    w = np.exp(sims[top] / temperature)
    return float(np.dot(sims[top], w / w.sum()))
```

This is dominated by the best match but rewards a paper that several members like. Also record `matched_members` — the members above a per-member threshold — which F5 requires for the digest ("which group members it matches").

**`lexical`** — facet-weighted, and fixing Finding 3 by normalizing against *matched potential* rather than the sum of all configured weights:

```python
FACET_WEIGHTS = {"primary_topics": 1.0, "materials": 0.8, "techniques": 0.7,
                 "secondary_topics": 0.6, "methods": 0.6, "ai_ml": 0.5}

def lexical_score(paper, profile) -> float:
    text = subscript_normalized(f"{paper.title} {paper.title} {paper.abstract}")
    hits = 0.0
    for facet, terms in profile.facets.items():
        fw = FACET_WEIGHTS[facet]
        for term in terms:
            for variant in surface_variants(term):        # Finding 4
                if whole_term_match(variant, text):
                    hits += fw * (1.0 - term_ambiguity(term))   # section 6.6
                    break
    # Saturating, NOT divided by total configured weight
    return 1.0 - math.exp(-hits / 2.5)
```

The saturating form means matching 3 strong terms scores ~0.70 and 6 terms ~0.91 — real dynamic range — instead of the current 0.09 ceiling. It also means adding more profile terms never dilutes existing matches, which is the specific pathology of the current denominator and the reason profiles cannot grow safely today.

**`novelty`** — genuine corpus-relative novelty:

```python
def novelty(paper_vec, corpus_vecs, recent_window_vecs) -> float:
    max_sim_ever   = max_cosine(paper_vec, corpus_vecs)        # all history
    max_sim_recent = max_cosine(paper_vec, recent_window_vecs) # last 8 weeks
    if max_sim_ever > 0.95:  return 0.0                        # near-duplicate
    return clamp(1.0 - 0.7*max_sim_recent - 0.3*max_sim_ever, 0.0, 1.0)
```

This distinguishes "the fifth incremental spin-ice paper this month" from "the first paper connecting spin ice to a new measurement technique" — which is what the requirements actually asked for when they mentioned identifying incremental follow-ups.

**`group_breadth`** — fraction of members with affinity above threshold, mildly compressed so a universally-mild-interest paper does not beat a specialist's essential paper:

```python
breadth = (n_members_above_threshold / n_active_members) ** 0.5
```

### 10.4 Diversity

Three mechanisms, layered, applied *after* reranking (§5.3):

**(a) Near-duplicate suppression.** Cosine > 0.95 against anything already selected or in recent history → drop. Cheap and unambiguous.

**(b) Semantic MMR.** The existing `selection.select_diverse()` already implements MMR correctly; the only change needed is replacing `_normalized_binary_vectors()` (binary bag-of-words) with real embeddings. Same algorithm, far better redundancy detection — binary token overlap treats two papers sharing boilerplate vocabulary as similar, while embeddings capture actual topical redundancy.

**(c) Area quotas with diminishing returns.** Rather than a hard cap, apply a decay so the *n*-th paper from an area must be substantially better to displace a first paper from another:

```python
def area_adjusted(score, area_counts, area, decay=0.75):
    return score * (decay ** area_counts[area])
```

With `decay = 0.75`, the 4th spin-ice paper is discounted to 42% of its raw score. This is strictly better than the current hard `max_papers_per_topic` cap because a genuinely exceptional fourth paper can still win — the cap is a preference, not a prohibition.

**Important:** replace `selection.topic_key()`'s fallback of `f"category:{primary_category}"` with assignment to the nearest research-area centroid. That fallback is failure mode 11 — it manufactures a private topic bucket for exactly the off-domain papers that should be crowded out, and then the quota logic protects them.

**(d) Per-researcher representation.** After the main top-10 is chosen, guarantee each active member at least one paper in their personal digest section, drawn from their highest-affinity unselected candidates. This is a selection guarantee in the *rendering* layer, not a ranking change — which keeps the group ranking honest while ensuring nobody opens the digest to find nothing for them.

### 10.5 Worked ranking trace

```
Paper: "Emergent magnetic monopole dynamics in the spin ice Dy2Ti2O7
        probed by AC susceptibility"     [cond-mat.str-el, cond-mat.mtrl-sci]

GATES        domain_score = 0.94  (cat=1.0, sim=0.71, ctx=1.0, off=0.12)  PASS
             hard_excluded = False                                        PASS
             duplicate_of = None                                          PASS

RELEVANCE    group_similarity   0.82 x 0.35 = 0.287
             researcher_affinity 0.91 x 0.30 = 0.273   [okafor 0.91, chen 0.64]
             lexical             0.78 x 0.10 = 0.078   [spin ice, monopole,
                                                        Dy2Ti2O7, pyrochlore]
MODIFIERS    llm_relevance       0.50 x 0.10 = 0.050   (not adjudicated: confident)
             novelty             0.61 x 0.08 = 0.049
             group_breadth       0.45 x 0.07 = 0.032
                                        base = 0.769
ADJUSTMENTS  recency 0.71  -> x 0.9855        = 0.758
             feedback 0.0  -> x 1.0           = 0.758

FINAL 0.758   area=magnetism   members=[okafor, chen]
EXPLANATION "Matches Amara Okafor's primary topics (spin ice, frustrated
             magnetism) and materials (pyrochlore, Dy2Ti2O7); technique
             (AC susceptibility) overlaps Wei Chen's interests."
```

Every number is traceable to a named component and every component to a profile field — requirement N3 satisfied.

---

## 11. Weekly Reading-List Architecture

### 11.1 Structure

```
WEEKLY GROUP READING LIST — 2026-08-24 to 2026-08-31
Condensed Matter Group · 517 papers scanned · 31 selected

HIGHLY RECOMMENDED (10)
  Top composite scores after diversity adjustment.
  Full treatment: summary, why-relevant, matched members, scores.

BY RESEARCH AREA (2-4 papers per area, areas with content only)
  Magnetism & Frustrated Systems
  Neutron & X-ray Scattering
  Quantum Materials & Correlated Electrons
  Spintronics & Transport
  Crystal Growth & Thin Films
  Computational Methods
  ML/AI for Condensed Matter
  Compact treatment: title, authors, one-line takeaway, link.

POSSIBLY INTERESTING / CROSS-DISCIPLINARY (3-5)
  High novelty, moderate group similarity, passed the domain gate.
  Includes a short "why we are showing you this" note.

FOR SPECIFIC MEMBERS (optional per-member block)
  Amara Okafor — 3 papers
  Wei Chen — 2 papers
  ...

APPENDIX
  Near misses (5) · run statistics · profile versions · model versions
```

### 11.2 Selection algorithm

```
1. candidates = ranked papers with composite > 0
2. HIGHLY RECOMMENDED:
     MMR over candidates, lambda = 0.7
     area decay 0.75, hard cap 3 per area
     near-duplicate suppression at 0.95
     -> 10 papers
3. BY AREA:
     for each research_area:
         remaining = candidates not yet selected, assigned to this area
         take top 2-4 by composite, MMR within the area
4. CROSS-DISCIPLINARY:
     pool = novelty > 0.7 AND 0.45 < group_similarity < 0.65
            AND domain_score > 0.5
     take top 3-5 by (novelty x group_similarity)
5. PER-MEMBER (optional):
     for each active member:
         if no selected paper has this member in matched_members:
             take their top-1 unselected candidate above threshold
6. NEAR MISSES: top 5 unselected by composite (existing behaviour, keep)
```

Step 4's *band* on `group_similarity` is the key idea for the cross-disciplinary section. Papers above 0.65 are just normal recommendations; below 0.45 they are usually noise. The interesting cross-disciplinary work sits in between — recognizably adjacent but not central. Requiring `domain_score > 0.5` prevents this section from becoming the new leak path for exactly the off-domain papers §6 works to exclude. That constraint is deliberate: an "interesting outliers" section is the most natural place for a filtering regression to reappear unnoticed.

### 11.3 Per-paper record

Extending the existing `Recommendation` model:

```python
class GroupRecommendation(StrictModel):
    paper: Paper                       # unchanged
    summary: PaperSummary              # unchanged; local LLM now
    score: GroupScoreBreakdown         # replaces ScoreBreakdown
    rank: int
    section: DigestSection             # highly_recommended | area | cross_disciplinary
    research_area: str | None
    matched_members: list[MemberMatch] # NEW - satisfies F5
    why_relevant_to_group: str         # NEW - generated

class MemberMatch(StrictModel):
    member_id: str
    display_name: str
    affinity: float = Field(ge=0.0, le=1.0)
    matched_facets: dict[str, list[str]]   # {"materials": ["pyrochlore"]}
```

`matched_facets` is what makes the digest legible: *"matches Amara Okafor's materials (pyrochlore) and techniques (diffuse neutron scattering)"* rather than a bare score.

### 11.4 One digest or personalized digests?

**Recommendation: one group digest with per-member sections, initially.** Move to per-member digests only if usage justifies it.

| | Shared digest | Per-member digests |
|---|---|---|
| Summarization cost | 1× (~30 papers) | Up to N× |
| Shared awareness | High — everyone sees the same list | Low — fragments the group |
| Implementation | Extends current rendering | Needs subscriber/render matrix |
| Serendipity | Higher | Lower — filter-bubble risk |
| Individual precision | Lower | Higher |

The shared digest also serves a social function that per-member digests destroy: a group reading list gives people something to *discuss*. That is arguably the main reason a lab wants this tool rather than each member running their own.

The hybrid — one shared digest with a short personalized block appended per recipient — captures most of the individual value at ~1× the summarization cost, since the same summaries are reused across recipients. Only the rendering differs per person. That is the recommended target.

---

## 12. Potential Email-Delivery Architecture

**No email is sent and no mailing system is implemented as part of this design task.** This section describes what would be required.

**Important audit note:** SMTP delivery **already exists** — `delivery.SMTPDeliveryProvider`, the `email-report` CLI command, and the `email` job in `.github/workflows/weekly_digest.yml`, gated behind `send_email: false` by default. The work below is a modification of a working system, not a greenfield build.

### 12.1 Current limitations for group use

`DeliveryConfig.to_emails` is a flat `list[str]` populated from `DIGEST_TO_EMAIL`. Consequences: one identical email to everyone; all recipients on a single `To:` header (every member's address exposed to every other — acceptable within a lab, but poor practice and a habit not worth carrying); no unsubscribe; no per-member content; no delivery record.

### 12.2 Proposed subscriber model

```yaml
# profiles/subscribers.yaml
subscribers:
  - member_id: a_okafor
    email: a.okafor@example.edu
    format: html              # html | text | both
    sections: [highly_recommended, by_area, personal]
    frequency: weekly
    active: true
  - member_id: guest_reader
    email: reader@example.edu
    format: text
    sections: [highly_recommended]
    active: true
```

Derived from researcher profiles where possible (`email` + `digest_opt_in`), so there is one place to add a person. `subscribers.yaml` holds only the overrides and non-member readers.

### 12.3 Rendering

Keep Jinja2 — it already produces both formats via `templates/weekly_report.{md,html}.j2`, and `reporting.render_reports()` is sound. Required changes:

- Render **once per distinct `(format, sections)` combination**, not once per subscriber. With most members on the same settings this is typically 2–3 renders for 30 people.
- Email HTML must be **table-based with inline CSS** — mail clients (Outlook especially) ignore `<style>` blocks and most modern CSS. The current HTML template targets browsers and the dashboard; it will need an email-specific variant. This is the single most underestimated task in email work.
- Always send `multipart/alternative` with a real text part. `delivery.SMTPDeliveryProvider._message()` already does this correctly.
- Keep total HTML under ~102 KB or Gmail clips it. With 30 papers and generated summaries this is a live constraint, not a theoretical one — measure it.

### 12.4 Sending

**Recommendation: institutional SMTP relay.** Most universities provide one, it requires no third-party data sharing, and it needs no new dependency. Use per-recipient sends (each with a single `To:`), not one message with 30 recipients.

Transactional services (SendGrid/SES/Postmark/Resend) give better deliverability and bounce handling, but for ≤30 internal recipients that is solving a problem the lab does not have, and it puts member addresses and reading interests on a third-party service. Not worth it here.

Rate-limit sends (~1/second) and record each delivery:

```python
class DeliveryRecord(StrictModel):
    run_id: str
    member_id: str
    email_hash: str          # SHA-256; never store plaintext in artifacts
    sent_at: datetime
    format: Literal["html", "text"]
    status: Literal["sent", "failed", "skipped"]
    error_type: str | None   # exception class only, never message contents
```

Hashing addresses in the persisted record matters because `data/` is committed to the repository by the CI workflow — plaintext addresses would be published to git history.

### 12.5 Scheduling

Three options, in order of preference for this project:

1. **Keep GitHub Actions for the digest, run LLM stages locally.** The CI runner has no GPU, so this means a split: local weekly run produces the digest JSON, CI (or a local scheduled task) sends it. Preserves the existing durable-state workflow.
2. **Windows Task Scheduler on the desktop.** Simplest for a fully local pipeline. Requires the machine to be on.
3. **A small always-on machine.** Only if the lab wants guaranteed delivery.

The existing workflow's persistence allowlist and idempotent history append are good foundations; extend rather than replace them.

### 12.6 Before any email is sent

- [ ] Explicit opt-in from every recipient — no auto-subscribing colleagues
- [ ] Working unsubscribe (a documented config edit is acceptable for a lab)
- [ ] Sender domain with SPF/DKIM configured, or institutional relay
- [ ] Dry-run mode writing `.eml` files to disk for inspection
- [ ] A hard cap on recipients per run as an accident guard
- [ ] Bounce handling, or at minimum failure logging with alerting

---

## 13. Feedback and Personalization Strategy

### 13.1 Signals

The existing `AGENTS.md` already specifies a richer taxonomy (`must_read`, `relevant`, `interesting_but_peripheral`, `not_relevant`, `already_known`, `poor_recommendation`). That is more nuance than most people will supply. Recommend collapsing to four for input, mapping to a numeric value:

| Signal | Value | Meaning |
|---|---|---|
| 👍 relevant | +1.0 | Good recommendation |
| 👎 not relevant | −1.0 | Bad recommendation |
| 📌 save / read later | +0.5 | Weak positive, high confidence |
| 🔁 more like this | +1.0 + expansion | Positive *and* a profile update signal |

Keep `already_known` distinct from `not_relevant` — they mean opposite things about the *profile* (the topic is right, the paper is stale) and conflating them teaches the system the wrong lesson.

### 13.2 Mechanism: bounded, transparent, and reversible

The requirements ask for something understandable and easy to debug. Three simple mechanisms, no learned model:

**(a) Per-paper feedback → author and concept nudges.** Store feedback; derive small adjustments:

```python
def feedback_score(paper, member_feedback) -> float:
    """Returns [-1, 1]. Bounded, decaying, explainable."""
    s = 0.0
    for fb in member_feedback:
        age_weeks = (now - fb.at).days / 7
        decay = 0.5 ** (age_weeks / 12)          # 12-week half-life
        sim = cosine(paper.embedding, fb.paper_embedding)
        if sim > 0.5:
            s += fb.value * decay * (sim - 0.5) * 2
    return clamp(s, -1.0, 1.0)
```

Feedback influences a new paper only in proportion to how similar it is to the paper actually rated, decays over 12 weeks, and is hard-clamped. Applied multiplicatively at ±15% (§10.2), it can reorder borderline cases but never promote an irrelevant paper or bury a strongly relevant one. This directly satisfies the `AGENTS.md` requirement that feedback "must not permanently suppress an entire broad field based on a single negative rating".

**(b) Profile drift suggestions, human-approved.** Every ~10 feedback events, propose an edit rather than making one:

```
Suggested profile update for a_okafor:
  You marked 4 papers about "altermagnetism" as relevant.
  Add "altermagnetism" to secondary_topics?  [y/N]
```

Automatic profile mutation is where recommenders become inscrutable. Suggest-and-confirm keeps the profile a human-authored artifact and keeps N3 intact.

**(c) Aggregate diagnostics, not just per-paper learning.** Track precision@10 per member per week. If someone's 👎 rate exceeds ~40% for three consecutive weeks, that is a *profile* problem, not a ranking problem — surface it as a prompt to revise their interests. This is often more valuable than any amount of score tuning.

### 13.3 Capture

Ordered by expected response rate:

1. **One-click links in the email digest** — `mailto:` links or a tiny local HTTP endpoint. Highest response rate by a wide margin, because it requires no context switch.
2. **The existing Streamlit dashboard** (`src/arxiv_digest/dashboard/`) — currently read-only; adding feedback buttons is a natural extension and requires no new infrastructure.
3. **CLI** — `arxiv-digest feedback <arxiv_id> <signal> --member <id>` — already specified in `AGENTS.md`. Useful for power users and testing; realistically low uptake.

Storage in SQLite:

```sql
CREATE TABLE feedback (
    id          INTEGER PRIMARY KEY,
    member_id   TEXT NOT NULL,
    arxiv_id    TEXT NOT NULL,
    signal      TEXT NOT NULL CHECK (signal IN ('relevant','not_relevant','saved','more_like_this','already_known')),
    value       REAL NOT NULL,
    run_id      TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE (member_id, arxiv_id, signal)
);
```

### 13.4 What to avoid initially

- **Collaborative filtering.** With 30 users and ~30 items/week there is nowhere near enough interaction data. It would fit noise.
- **Learned ranking models (LambdaMART, neural rerankers).** Need thousands of labelled examples; a lab will produce a few hundred a year.
- **Automatic weight optimization.** Tempting, but with sparse feedback it overfits to whoever clicks most. Revisit after a year of data, if ever.

The bounded-nudge approach above will capture most of the achievable gain at a fraction of the complexity, and — more importantly — it stays debuggable when it misbehaves.

---

## 14. Comparison of Implementation Approaches

### Option A — Minimal Upgrade

Keep the current architecture; add category filters, a group profile, basic embeddings, local LLM summaries, and multi-section digest rendering. Keep TF-IDF alongside embeddings. Keep JSON/JSONL storage.

**Changes:** category guards in `arxiv_client.build_search_query()`; fix Findings 2–4 in `ranking.py`; add `EmbeddingProvider` implementation and wire it in; add `LocalLLMSummaryProvider`; extend `ResearchProfile` with a member list; extend templates.

| Axis | Assessment |
|---|---|
| Dev complexity | **Low** — ~2–3 weeks part-time |
| Compute | Minimal; embeddings + summaries only |
| Quality | **Moderate.** Fixes the monopole problem (the biggest win) but researcher matching stays coarse |
| Maintainability | Good initially; the single-`ResearchProfile` assumption becomes a growing tax |
| Debuggability | **Excellent** — mostly deterministic |
| Hardware fit | Excellent |
| Scale to 5–30 | **Poor.** Members bolted onto a single-profile design; no per-member scoring, no feedback |

**Verdict:** the right *first phase*, wrong final destination. Its work is not wasted — it is a strict subset of Option B.

### Option B — Hybrid Recommendation Pipeline ★ Recommended

The architecture of §5–§13: firehose ingestion, layered filtering, embeddings for matching, conditional local-LLM reranking, composite scoring, diversity-aware selection, researcher profiles, feedback tracking, group digest with per-member sections. SQLite + NumPy storage.

| Axis | Assessment |
|---|---|
| Dev complexity | **Moderate** — ~6–10 weeks part-time, phased and independently shippable |
| Compute | ~14 min/week on the laptop, ~9 min on the desktop (§9.5) |
| Quality | **High.** Semantic matching plus layered filtering addresses every stated failure |
| Maintainability | **Good.** Clear stage boundaries; each independently testable |
| Debuggability | **Good.** Deterministic except two bounded LLM stages, both logged with reasons |
| Hardware fit | **Excellent** — designed around the 6 GB constraint |
| Scale to 5–30 | **Good.** Per-member scoring is a matmul; adding a member costs one profile file |

**Verdict:** the right balance. Every component is justified by a specific audit finding or requirement, and nothing is included speculatively.

### Option C — Research Intelligence System

Everything in B, plus: a paper-to-paper citation/similarity graph, structured concept extraction into a knowledge base, natural-language querying, RAG over accumulated literature, evolving interest profiles, per-member personalized recommendations.

| Axis | Assessment |
|---|---|
| Dev complexity | **High** — 6+ months; concept extraction and graph maintenance are open-ended |
| Compute | Substantially higher; concept extraction over the full corpus is many LLM calls |
| Quality | Potentially highest, but **only with sustained curation** |
| Maintainability | **Concerning.** Concept schemas drift; extraction quality varies; the graph needs upkeep |
| Debuggability | **Poor.** Multi-hop reasoning failures are hard to attribute |
| Hardware fit | Marginal — RAG over a large corpus wants more VRAM for context |
| Scale to 5–30 | Good if it works; high risk of never quite working |

**Verdict:** the right *long-term direction*, wrong immediate build. Its highest-value pieces — semantic search over collected papers, NL Q&A — are cheap additions *once B's embedding infrastructure exists*, because B already stores exactly the vectors RAG needs. Build B, then add C's components individually as they prove their worth.

### Summary

| | A | **B** | C |
|---|---|---|---|
| Time to first value | 2–3 wk | 3–4 wk (Phase 1) | 3+ mo |
| Total effort | Low | Moderate | High |
| Fixes monopole problem | ✅ | ✅ | ✅ |
| Semantic matching | Partial | ✅ | ✅ |
| Real multi-researcher support | ❌ | ✅ | ✅ |
| Feedback loop | ❌ | ✅ | ✅ |
| NL query over corpus | ❌ | ❌ (Phase 6) | ✅ |
| Runs in <15 min on RTX 4050 | ✅ | ✅ | ⚠️ |
| Debuggable by a physicist | ✅ | ✅ | ⚠️ |
| **Recommended** | Phase 1 only | **★ Yes** | Later |

---

## 15. Recommended Architecture

**Build Option B, phased so each phase ships something usable.**

Five decisions carry the design.

**1. Fix filtering before adding people.** Every audit finding says the current system admits papers it should not. Adding 30 profiles to that pipeline multiplies the noise rather than dividing the labour, and it makes the noise harder to diagnose because it is now distributed across 30 people's sections. Phase 1 must land first.

**2. Ingest the whole cond-mat firehose.** 517 papers/week is small. Retrieving everything and filtering locally removes query-coverage bias permanently, makes adding researchers free (N5), and reduces arXiv load. This is the highest-leverage architectural change in the report, and it is *simpler* than what exists — fewer queries, less configuration.

**3. Gates before scores.** Domain exclusion is binary and happens before ranking (§6). Finding 2 is the proof that soft penalties inside a weighted sum do not hold. This one structural choice fixes the stated problem more reliably than any amount of weight tuning.

**4. Embeddings for matching, LLMs for explaining.** Embeddings handle all 517 papers cheaply and deterministically. The LLM sees ~70 papers/week: ~40 borderline adjudications and ~30 summaries. It is used where language generation and nuanced judgement are genuinely required, not as a scoring function.

**5. One corpus, many audiences.** Fetch, embed, score, and summarize once against a group model; compute per-member affinity as a matrix operation on top. Adding the 31st member costs one profile file and one row in a matmul — not another pipeline run.

```
                          ┌─────────────────┐
                          │  arXiv API      │
                          │  cat:cond-mat.* │
                          └────────┬────────┘
                                   │ ~517/week
                    ┌──────────────▼──────────────┐
                    │  DETERMINISTIC GATES        │   fast, testable,
                    │  categories, rules, history │   every rejection logged
                    └──────────────┬──────────────┘
                                   │ ~500
                    ┌──────────────▼──────────────┐
   profiles/  ─────▶│  EMBED + DOMAIN GATE        │◀──── Qwen3-Embedding-0.6B
   group.yaml       │  cache by (id, version)     │      (+ PhysBERT, benchmark)
   researchers/     └──────────────┬──────────────┘
                                   │ ~350
                    ┌──────────────▼──────────────┐
                    │  COMPOSITE SCORING          │   NumPy; no LLM;
                    │  group / member / lexical / │   fully explainable
                    │  novelty / breadth / fb     │
                    └──────────────┬──────────────┘
                                   │ top ~60
                    ┌──────────────▼──────────────┐
                    │  LLM RERANK (band only)     │◀──── Qwen3-4B / Reranker-0.6B
                    │  ~40 papers, ~11%           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  DIVERSITY SELECTION        │   MMR + area decay
                    │  + per-member guarantee     │   + dup suppression
                    └──────────────┬──────────────┘
                                   │ ~30
                    ┌──────────────▼──────────────┐
                    │  SUMMARIZE (local LLM)      │◀──── Qwen3-8B (desktop)
                    │  PaperSummary, grammar-     │      Qwen3-4B (laptop)
                    │  constrained JSON           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  RENDER: group + per-member │   Jinja2 (existing)
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        reports/*.md         reports/*.html        papers.db
                                                   (feedback, history,
                                                    embeddings)
```

---

## 16. Suggested Repository/Module Changes

Additive where possible. Existing modules that work well — `arxiv_client.py`, `normalization.py`, `reporting.py`, `history.py` — change minimally.

```
src/arxiv_digest/
  config.py              MODIFY  add GroupProfile, ResearcherProfile, DomainConfig;
                                 load_settings() takes a profiles directory
  models.py              MODIFY  add GroupScoreBreakdown, MemberMatch,
                                 GroupRecommendation, DigestSection;
                                 KEEP Paper, PaperSummary unchanged
  arxiv_client.py        MODIFY  category-constrained queries; grouped ANDNOT only;
                                 add fetch_category_window() for the firehose
  ranking.py             REPLACE lexical scoring (saturating, facet-weighted,
                                 formula variants); composite score; gates
  selection.py           MODIFY  embedding-based MMR; area decay;
                                 topic_key -> research-area assignment
  summarization.py       MODIFY  add LocalLLMSummaryProvider; keep the Protocol
                                 and OpenAI provider as-is
  pipeline.py            MODIFY  orchestrate the new stages
  reporting.py           MODIFY  multi-section digest rendering
  delivery.py            MODIFY  subscriber-aware (Phase 5; NOT NOW)

  profiles.py            NEW     load/validate/compile researcher + group profiles
  embeddings.py          NEW     EmbeddingProvider impls; cache; matrix store
  domain_filter.py       NEW     hard rules, context requirements, domain scoring
  disambiguation.py      NEW     ambiguous-term registry; term_ambiguity() via API
  llm/                   NEW     local LLM clients
    __init__.py
    ollama.py                    OllamaClient (grammar-constrained JSON)
    rerank.py                    cross-encoder / LLM adjudication
    concepts.py                  structured concept extraction (Phase 4)
  storage.py             NEW     SQLite schema, migrations, paper upsert
  feedback.py            NEW     capture, decay, bounded scoring

config/
  app.yaml               MODIFY  add local_llm and embedding blocks
  group.yaml             NEW     group profile, domain gate, research areas
  disambiguation.yaml    NEW     ambiguous terms, context rules, hard exclusions
  research_profile.yaml  KEEP    migrate to profiles/researchers/ (see below)

profiles/
  group.yaml             NEW
  researchers/*.yaml     NEW     one file per member
  subscribers.yaml       NEW     Phase 5
  compiled/              NEW     generated; gitignored

data/
  papers.db              NEW     SQLite
  embeddings.npy         NEW     float16 matrix; gitignored
  history.jsonl          KEEP    migrate into papers.db over time

docs/
  lab_wide_research_discovery_design.md   THIS FILE
```

**Migration path for the existing profile.** `config/research_profile.yaml` becomes `profiles/researchers/<owner>.yaml`, with `exact_phrases`/`materials`/`methods` mapping onto `facets`. A one-off `arxiv-digest profiles migrate` command can do this mechanically, so no interests are lost and the current user's tuning carries forward.

**New CLI commands:**

```bash
arxiv-digest profiles list
arxiv-digest profiles add --name "Amara Okafor" --email a.okafor@example.edu
arxiv-digest profiles expand a_okafor        # keywords -> facets (LLM, reviewed)
arxiv-digest profiles compile                # -> embeddings, cached
arxiv-digest profiles check a_okafor         # ambiguity warnings, coverage report
arxiv-digest ingest --days 7                 # firehose, populate papers.db
arxiv-digest digest --days 7 --limit 10      # full pipeline
arxiv-digest explain <arxiv_id>              # full score breakdown (N3)
arxiv-digest feedback <arxiv_id> relevant --member a_okafor
arxiv-digest search "spin ice monopole dynamics"   # Phase 6
```

`arxiv-digest explain` is worth building early. When someone asks "why did this show up?", a command that prints every gate result and score component turns a debate into a lookup — and it is the practical embodiment of N3.

**Dependency additions:** `sentence-transformers` (embeddings), `httpx` (already present — reuse for Ollama), `sqlite3` (stdlib). Notably **no vector database, no ORM, no ML framework beyond what sentence-transformers pulls in** (PyTorch). PyTorch is the one heavy addition; it is unavoidable for local embeddings and justified by §7.

---

## 17. Phased Implementation Roadmap

The requirements' proposed ordering is close to right. **One change is recommended: move the weekly digest structure earlier**, because the group digest is the visible deliverable that keeps the lab engaged, and it can be built on deterministic ranking before the LLM stages exist.

### Phase 1 — Fix discovery quality *(highest priority)*

- Category-constrained queries; grouped `ANDNOT` only, with a regression test for the chained-`ANDNOT` bug (§6.2)
- Hard exclusion rules + ambiguous-term context requirements (§6.3–6.4)
- Fix Finding 2 (remove the score floor; gates before scores)
- Fix Finding 3 (saturating lexical score)
- Fix Finding 4 (formula variants, subscript normalization)
- `arxiv-digest explain`
- **Deliverable:** the current single-user digest, with the astro/HEP leakage gone
- **Validation:** re-run 2026-07-21→28 and confirm the `hep-ph` monopole paper and the plasma paper no longer appear

*Phase 1 is worth doing even if nothing else in this report is built.* It is small, self-contained, and fixes the problem that prompted the request.

### Phase 2 — Storage and firehose ingestion

- SQLite schema; migrate `history.jsonl`
- `fetch_category_window()` for full cond-mat ingestion
- Paper upsert with version handling
- **Deliverable:** a persistent corpus; recall no longer depends on query wording

### Phase 3 — Embeddings and semantic ranking

- `EmbeddingProvider` implementations (Qwen3-Embedding-0.6B; benchmark PhysBERT/SPECTER2)
- Embedding cache and matrix store
- Semantic domain gate (§6.5)
- Replace TF-IDF `semantic_relevance` with real embedding similarity
- Embedding-based MMR in `selection.py`
- **Deliverable:** genuine semantic matching; the monopole distinction now works on *meaning*, not just categories

### Phase 4 — Lab profiles

- `ResearcherProfile` / `GroupProfile` schemas; profile loading and compilation
- Per-member affinity scoring; `matched_members`
- Research-area assignment
- Profile expansion (keywords → facets) with the LLM, human-reviewed
- Term-ambiguity measurement (§6.6)
- **Deliverable:** multi-researcher support
- **Note:** deliberately *after* embeddings — profile matching without embeddings would just be the current broken keyword matching, multiplied by 30

### Phase 5 — Weekly group digest

- Multi-section selection (§11.2)
- Area quotas, per-member guarantees
- Extended templates: group + per-member sections
- **Deliverable:** the weekly group reading list, still with deterministic summaries

### Phase 6 — Local LLM pipeline

- Ollama client with grammar-constrained JSON
- `LocalLLMSummaryProvider` reusing `PaperSummary`
- Borderline adjudication (uncertainty band)
- Cross-encoder reranking
- "Why relevant to our group" generation
- **Deliverable:** high-quality summaries and explanations with no API cost

### Phase 7 — Email delivery *(explicitly deferred)*

- Subscriber model; email-safe HTML templates; per-recipient rendering
- Dry-run mode; delivery records; opt-in verification
- Scheduling
- **Gate:** do not start until the digest content is good enough that people would want it in their inbox. Emailing a mediocre digest is worse than not emailing.

### Phase 8 — Feedback and literature intelligence

- Feedback capture (dashboard first, then email links)
- Bounded feedback scoring; profile drift suggestions
- Semantic search over the corpus
- RAG-based Q&A (F12)
- **Deliverable:** a system that improves with use

**Critical path:** Phases 1 → 2 → 3 → 4 → 5. Phase 6 can run in parallel with 4–5 (the provider interface already exists). Phases 7–8 are genuinely optional.

---

## 18. Risks and Failure Modes

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Over-filtering hides relevant work** | **High** | **High** | Log every rejection with rule/score. Weekly "rejected but high researcher-affinity" report. Never hard-reject a cond-mat cross-listed paper (§6.4). Track recall on a labelled set (§19). |
| Embedding model poor on physics text | Medium | High | Benchmark before committing (§19.2). Keep lexical scoring as an independent signal so no single model is a point of failure. |
| Local LLM output quality disappoints vs. GPT-class | Medium | Medium | Grammar-constrained output guarantees structure. Keep the OpenAI provider as an opt-in escape hatch. Blind-compare summaries (§19.3). |
| Laptop VRAM exhaustion mid-run | Medium | Medium | Sequential model loading; explicit unload between stages; a `--profile laptop` config capping model sizes. Fail loudly, not by silent CPU fallback. |
| Profiles go stale as interests drift | **High** | Medium | Quarterly review prompt; drift suggestions from feedback (§13.2b); flag members with no positive feedback in 8 weeks. |
| Low group engagement — nobody reads it | **High** | **High** | Ship Phase 5 early for feedback. Keep the digest short. Measure open/click if emailed. *This is the most likely way the project fails.* |
| Score threshold drift across runs | Medium | Medium | Persist per-run percentiles; use trailing 8-week medians for thresholds (§10.3). Alert when selected-count deviates sharply. |
| SQLite corruption / concurrent writes | Low | High | WAL mode; single-writer discipline; keep JSON snapshots as a rebuildable audit trail. |
| arXiv API changes or rate-limits | Low | High | Existing client already paces and retries. Cache raw responses. Chained-`ANDNOT` regression test guards silent query-semantics drift. |
| Feedback loop creates a filter bubble | Medium | Medium | Cap feedback at ±15%, multiplicative. Reserve the cross-disciplinary section, which ignores feedback entirely. |
| Member email addresses committed to git | Medium | Medium | Hash addresses in persisted records (§12.4). Audit the CI persistence allowlist before Phase 7. |
| Scope creep into Option C | **High** | Medium | Phase gates. Do not start Phase 8 until 1–5 have run for a month. |

Two deserve emphasis.

**Over-filtering is the mirror image of the current bug, and it is harder to notice.** Today's failure is loud — an obviously wrong paper in the list. Tomorrow's failure would be silent: a relevant paper that never appears, which nobody can miss because nobody knows it existed. The weekly "rejected but high-affinity" report is not optional polish; it is the only mechanism that makes this class of failure visible.

**Low engagement is the most probable overall failure.** The technical work here is tractable. Getting 30 physicists to read a weekly email is not. Ship early, keep it short, and measure whether anyone opens it before investing in Phase 8.

---

## 19. Recommended Experiments Before Implementation

Six experiments, ordered by how much they de-risk the design. Together roughly two to three days of work — a good investment before committing to weeks of implementation.

### 19.1 Build a labelled evaluation set *(do this first)*

Nothing else can be measured without it.

- Take 3–4 past weeks of `cat:cond-mat.*` plus deliberately included `hep-ph`/`astro-ph` monopole papers
- Label ~300 papers: `relevant` / `adjacent` / `irrelevant` / `off-domain`
- Include ~20 hard cases: emergent monopoles, ML-for-materials cross-lists, `quant-ph` materials papers
- Store as `tests/fixtures/eval_set.jsonl`

**Success:** a fixed set that every subsequent change is measured against. The existing test suite is strong on correctness but has no relevance benchmark — this fills that gap and makes the ranking changes falsifiable rather than a matter of opinion.

### 19.2 Embedding model bake-off

Compare `Qwen3-Embedding-0.6B`, `PhysBERT`, `SPECTER2`, and (as a baseline) the current TF-IDF, on:

- **Domain classification:** AUC for cond-mat vs. off-domain
- **Interest matching:** nDCG@10 against the labelled set using the existing profile
- **Speed:** abstracts/second on the RTX 4050
- **Monopole discrimination:** cosine gap between spin-ice-monopole and cosmological-monopole abstracts

**Decision rule:** pick the best domain-classification AUC for the gate and the best nDCG for matching. If they differ, run both — PhysBERT is 0.4 GB and can sit on CPU.

### 19.3 Local LLM summary quality

Generate summaries for the same 20 papers with `Qwen3-4B-Instruct`, `Qwen3-8B`, `Gemma 4 E4B`, and (as reference) the current OpenAI provider. Blind-rate on accuracy, usefulness, and hallucination.

**Also measure:** schema-validation pass rate with and without grammar constraints, and tokens/second on both machines.

**Success:** a local model whose summaries a physicist rates ≥ 4/5 and which passes `PaperSummary` validation ≥ 95% of the time.

### 19.4 Filtering ablation

On the labelled set, measure precision/recall for each cumulative layer:

| Configuration | Precision | Recall |
|---|---|---|
| Current system (baseline) | ? | ? |
| + category query guard | ? | ? |
| + hard exclusion rules | ? | ? |
| + context requirements | ? | ? |
| + semantic domain gate | ? | ? |
| + LLM adjudication (band) | ? | ? |

**Success:** ≥ 0.95 precision on off-domain rejection with ≥ 0.90 recall on relevant papers. **This is the experiment that validates or refutes §6**, and the per-layer breakdown will show which layers are actually carrying their weight — some may prove unnecessary, which is a useful finding.

### 19.5 Threshold calibration

Using the labelled set, plot score distributions for relevant vs. irrelevant papers and set `DOMAIN_FLOOR`, `CONFIDENT_ACCEPT`, and `CONFIDENT_REJECT` where the curves actually separate — rather than at the guessed values in §8.3. Measure what fraction of papers land in the uncertainty band; if it substantially exceeds ~15%, the runtime estimate in §9.5 needs revisiting.

### 19.6 Profile expansion trial

Have 3–5 lab members supply 5–15 keywords. Run expansion. Measure: how many generated facet terms they accept unchanged, how many they edit, how many are wrong. Time the whole process per person.

**Success:** ≥ 70% of generated terms accepted, < 10 minutes per researcher. If acceptance is low, fall back to a manual facet template — the profile format still works, it just costs more to fill in, and that is an acceptable outcome. Worth knowing before building the expansion pipeline.

---

## Closing Recommendation

> **If we wanted to turn the existing personal paper-discovery tool into a reliable lab-wide condensed-matter research intelligence system while keeping it practical on consumer hardware, what architecture should we actually build first, and why?**

**Build a layered-gate, embedding-first hybrid pipeline (Option B), and build it in the order Phase 1 → 2 → 3 → 4 → 5.**

The reasoning is that the system's problem is not a missing feature — it is that **the current pipeline has no mechanism that can express "this paper is not in our field."** Category information exists but only as a 10%-weighted soft term; the term "semantic" describes a lexical TF-IDF score; and a constant novelty bonus guarantees any recent paper clears the wildcard threshold. The `hep-ph` dark-monopole paper sitting at rank 10 in the repository's own committed data is not an unlucky edge case — it is what the scoring arithmetic is guaranteed to produce.

So the first thing to build is not researcher profiles, and not local LLMs. It is **gates**: a category constraint in the query itself (76% of monopole noise gone, measured, for a one-line change), hard exclusion rules that can never fire on a cond-mat cross-listed paper, and the removal of the relevance-free score floor. That is Phase 1. It is perhaps a week of work, it is almost entirely deterministic and testable, and it solves the problem that prompted this request — with or without anything else in this report.

Everything after that follows from a single measurement: **arXiv publishes only ~517 condensed-matter papers a week.** That number is what makes the whole design tractable. It is small enough to ingest completely (so recall stops depending on whether someone thought to add the right search term), small enough to embed on a laptop GPU in under a minute, and small enough that per-researcher matching for 30 people is one matrix multiplication rather than 30 pipeline runs. The right shape is therefore *one corpus, many audiences*: fetch, embed, score, and summarize once against a group model, then compute individual affinities on top.

Embeddings come before profiles in the ordering, and that sequencing is deliberate. Researcher profiles matched by keywords would inherit every defect the audit found — the crushed keyword denominator, the LaTeX-formula matching failure — and multiply them by the number of lab members. Semantic matching has to work for one person before it is worth extending to thirty.

The local LLM comes last among the core phases, and does the least work, on purpose. It sees ~70 papers a week out of 517: roughly 40 borderline adjudications and 30 summaries. Everything else — category logic, deduplication, similarity, novelty, diversity, area assignment — is deterministic code or a matrix operation, because those are faster, reproducible, and debuggable at 2 a.m. by a physicist rather than an ML engineer. The LLM is reserved for the two things it is genuinely better at than any alternative: judging genuinely ambiguous cases, and writing prose a colleague will actually read.

The existing codebase makes this considerably easier than a rewrite would be. `PaperSummary` and its validators — the word limits, the exact `summary_basis` match, `reject_inspection_claims()` — port to local models unchanged, so the abstract-only evidence discipline is inherited rather than rebuilt. The `SummaryProvider` Protocol is already the right seam. `EmbeddingProvider` is already defined and waiting to be implemented. The MMR selection is already correct and needs only better vectors. **The architecture to build first is largely the architecture that is already there, with gates added at the front, real embeddings in the middle, and a group model replacing the single profile.**

One last point on sequencing, which matters more than any technical choice in this report: ship Phase 5 — the actual weekly group digest — before Phase 6's polish. The most likely way this project fails is not a filtering bug or a VRAM limit. It is that a technically excellent digest gets built that nobody in the group reads. Put a short, decent list in front of the lab early, find out whether they want it, and let their answer decide how much of the rest is worth building.
