I want to expand the existing research-paper discovery/recommendation tool in this project from a workflow optimized around my own research interests into a tool that can serve our broader condensed-matter physics research group.

**For this task, do not implement or modify the code yet.** I first want a careful design and feasibility analysis of the existing system and a report proposing how we should evolve it.

Please inspect the current codebase, configuration files, ranking/filtering logic, paper sources/APIs, keyword handling, database/storage approach, and any existing LLM integration before making recommendations. Base your proposals on what is actually present in the repository rather than assuming an architecture.

## 1. Goal: Expand from an Individual Tool to a Lab-Wide Tool

The current system has been tuned primarily around my own interests. We want to make it useful to an entire condensed-matter physics group.

Our group's interests should remain predominantly within condensed-matter/materials physics, although there may be adjacent interests such as:

* quantum materials
* magnetism
* frustrated magnetism
* spin ice
* spin glass
* neutron scattering
* raman scattering
* spectroscopy
* transport
* single crystal growth
* monte carlo methods in condensed-matter
* thin films
* spintronics
* magnetic materials
* correlated-electron systems
* computational condensed-matter physics
* materials characterization
* machine learning for physics/materials
* AI methods relevant to physical sciences

These are examples rather than the final taxonomy.

My current idea is to ask each member of the group to provide several **research keywords/topics** describing their interests. I would like you to evaluate this approach and propose a better representation if appropriate.

For example, consider whether each researcher should have a profile containing some combination of:

* primary research topics
* secondary topics
* experimental techniques
* materials/material classes
* theoretical/computational methods
* AI/ML interests
* negative/exclusion topics
* keyword weights or importance levels

Consider whether maintaining both:

1. a **group-level interest profile**, and
2. individual **researcher profiles**

would give better recommendations.

I do not necessarily want to build a complicated recommendation system immediately. Please distinguish between an effective MVP and features that could be added later.

---

## 2. Weekly Group Reading List

We also want the tool to generate a **weekly reading list that can eventually be emailed to members of the research group**.

Explore architectures for producing something like:

### Weekly Group Reading List

**Highly Recommended**

* 10 papers most relevant to the group's current interests

**By Research Area**

* Magnetism / frustrated systems
* Neutron scattering
* Quantum materials
* Spintronics
* Computational methods
* ML/AI for condensed matter
* etc.

**Possibly Interesting / Cross-Disciplinary**

* papers slightly outside the group's normal topics but judged potentially useful

For each paper, the digest might contain:

* title
* authors
* journal / arXiv information
* publication date
* URL/DOI
* short generated summary
* why the paper may be relevant to our group
* which research interests or group members it matches
* relevance score
* perhaps a novelty/cross-disciplinary score

Please evaluate whether the same digest should go to everyone initially or whether personalized sections/digests would be practical.

Also explore an eventual email workflow, but **do not send emails or implement the mailing system at this stage**. Recommend suitable options for generating HTML/text digests, maintaining a subscriber list, and eventually scheduling weekly delivery.

---

## 3. Local LLM Integration

I want to make much greater use of **local LLMs** so that routine processing can happen locally without requiring expensive API calls.

One obvious application is generating paper summaries, but I would like you to explore other useful roles for a local model throughout the pipeline.

Potential applications include:

* paper abstract summarization
* extracting important concepts/topics from abstracts
* assigning papers to research categories
* comparing papers with researcher-interest profiles
* semantic relevance ranking
* reranking candidates returned by conventional search
* generating "Why this might matter to our group" explanations
* identifying papers that are likely duplicates or incremental follow-ups
* identifying genuinely unusual or potentially high-impact papers
* identifying interdisciplinary papers worth surfacing
* extracting materials, techniques, physical phenomena, and methods
* generating search-query expansions from researcher keywords
* detecting false-positive papers
* creating the weekly digest
* eventually answering natural-language questions over the collected paper database

Please distinguish tasks that should be handled by:

* deterministic filters/rules,
* embeddings,
* small local LLMs,
* larger local LLMs,
* and external APIs when genuinely necessary.

I do **not** want to use an LLM where conventional code or embeddings would be faster and more reliable.

---

## 4. Hardware Constraints and Local Model Recommendations

The system should remain practical on hardware I already have.

### Laptop

* Dell XPS 15, approximately 2023 generation
* NVIDIA RTX 4050 Laptop GPU
* 16 GB system RAM

### Desktop

* NVIDIA RTX 4060
* 32 GB system RAM

Assume typical consumer versions of these GPUs unless the repository contains more precise hardware information.

Recommend realistic local models and quantization levels for these systems.

Please investigate current model options rather than relying purely on older model recommendations. Consider models in roughly the:

* 1–4B
* 7–9B
* potentially ~12–14B if realistic with aggressive quantization

ranges.

Evaluate suitable recent models from families such as Qwen, Gemma, Llama, Phi, Mistral, or better alternatives that currently exist.

For each promising model, discuss:

* approximate memory requirements
* whether it should run comfortably on the laptop
* whether it is more appropriate for the desktop
* expected inference performance
* summary quality
* classification/reranking ability
* structured-output reliability
* context-window considerations

Also discuss whether it would be beneficial to use **different models for different stages**, for example:

* very small model for classification/filtering
* embeddings for candidate retrieval
* stronger 7–9B model for final summaries and relevance explanations

Consider practical runtimes such as:

* Ollama
* llama.cpp
* LM Studio
* Transformers
* vLLM if appropriate

and recommend which makes the most sense for this project.

---

## 5. Critical Filtering Problem: Avoiding Unrelated Physics Papers

I previously encountered an important problem.

Because terms such as **"monopoles"** are relevant to spin ice, searches started returning large numbers of papers from:

* astrophysics
* cosmology
* high-energy physics
* particle physics

These are generally irrelevant to what I want.

The term "monopole" illustrates the larger issue: **keyword matching without scientific context is inadequate.**

I want a robust strategy for preventing these false positives while retaining papers about emergent magnetic monopoles in condensed matter.

For example:

> "magnetic monopole excitations in spin ice"

should score highly, whereas:

> "primordial magnetic monopoles in cosmology"

should be rejected.

Please explore multiple ways of solving this problem.

Possible mechanisms include:

### Source/category filtering

For arXiv, strongly prefer relevant categories such as:

* cond-mat.*
* physics.ins-det where appropriate
* physics.comp-ph where appropriate

and reject or strongly penalize categories such as:

* astro-ph.*
* hep-*
* gr-qc

unless another strong condensed-matter signal exists.

### Positive context

Terms such as:

* spin ice
* pyrochlore
* frustrated magnetism
* neutron scattering
* condensed matter
* magnetic excitations

should reinforce the condensed-matter interpretation of ambiguous keywords.

### Negative keywords/context

Terms such as:

* primordial
* dark matter
* cosmology
* early universe
* black hole
* collider
* supersymmetry

might strongly penalize a result.

### Semantic classification

Embeddings or a small LLM could classify:

> "Is this paper substantially relevant to condensed-matter/materials physics?"

before the paper enters the main ranking pipeline.

Please compare these approaches and recommend a **layered filtering architecture** rather than relying on one method.

I am particularly interested in something conceptually like:

```text
Paper sources
      ↓
Metadata/category filter
      ↓
Hard exclusion rules
      ↓
Broad condensed-matter relevance classifier
      ↓
Embedding similarity to group/researcher interests
      ↓
Local-LLM reranking
      ↓
Novelty/diversity adjustment
      ↓
Weekly recommendation list
```

Evaluate whether this architecture makes sense and propose improvements.

---

## 6. Ranking Architecture

Explore how ranking should work once irrelevant papers have been removed.

I do not want the ranking to be entirely dependent on a single LLM score.

Consider a composite score such as:

```text
FinalScore =
    w1 × TopicSimilarity
  + w2 × ResearcherSimilarity
  + w3 × Recency
  + w4 × SourceQuality
  + w5 × Novelty
  + w6 × LLMRelevance
  + w7 × GroupBreadth
  - w8 × ExclusionPenalty
```

Do not assume these exact variables or weights are correct.

Please propose a sensible scoring/reranking architecture and explain how each component could be calculated.

Also consider **diversity**. I do not want the weekly top 10 to consist of ten nearly identical spin-ice papers simply because one topic happens to have many recent submissions.

Explore approaches such as:

* maximum marginal relevance (MMR)
* category quotas
* topic clustering
* diminishing returns for similar papers
* per-researcher representation

The goal should be to balance:

**relevance + quality + diversity + novelty.**

---

## 7. Embeddings

Evaluate whether embeddings should become an important part of the system.

Potential uses include:

* matching abstracts to research interests
* finding similar papers
* researcher-profile matching
* clustering
* deduplication
* novelty detection
* semantic search over previously collected papers

Recommend local embedding models suitable for scientific text and my hardware.

Where appropriate, compare general-purpose embedding models against scientific-domain models.

Also recommend whether a simple local vector database is worthwhile. Possible options might include:

* FAISS
* SQLite with vector extensions
* Chroma
* Qdrant
* LanceDB

Avoid adding unnecessary infrastructure if the expected database is small enough that a simpler approach would work.

---

## 8. Researcher Profile Design

Propose a clean data format for research-group members.

For example, perhaps something conceptually like:

```yaml
name: Researcher_A

primary_topics:
  - frustrated magnetism
  - spin ice

secondary_topics:
  - neutron scattering
  - magnetic excitations

materials:
  - pyrochlores
  - rare-earth magnets

techniques:
  - neutron diffraction
  - diffuse scattering

methods:
  - Monte Carlo
  - spin-wave theory

ai_ml:
  - Bayesian optimization

exclude:
  - astrophysical monopoles
  - collider physics
```

This is only illustrative.

Please determine whether storing plain keywords is sufficient or whether hierarchical/profile-based metadata would meaningfully improve the system.

I still want submitting interests to be easy for lab members, so the input process should remain lightweight.

For example, perhaps members provide 5–15 free-form keywords and the tool automatically expands them into a richer profile.

Explore this possibility.

---

## 9. Possible Human Feedback Loop

Consider whether the system should gradually learn from lightweight user feedback such as:

* 👍 relevant
* 👎 not relevant
* save/read later
* "more like this"
* "less like this"

Discuss simple ways this feedback could improve ranking without requiring a complicated machine-learning recommendation system.

The initial version should remain understandable and easy to debug.

---

## 10. Existing-System Audit

Before proposing a redesigned architecture, inspect the repository and report:

1. How papers are currently discovered.
2. Which APIs/data sources are currently used.
3. How keywords are currently represented.
4. What filtering currently exists.
5. How ranking currently works.
6. Whether embeddings are currently used.
7. Whether any LLM integration already exists.
8. How metadata/history is stored.
9. How easy it is to support multiple researcher profiles.
10. Where the current architecture is likely to fail when scaling from one user to an entire lab.

Please identify code files/functions relevant to each part of the pipeline.

---

## 11. Implementation Options

Provide at least **three plausible implementation strategies**, for example:

### Option A — Minimal Upgrade

Keep the existing architecture and add:

* group profiles
* arXiv/category exclusions
* improved keyword/context filtering
* basic embeddings
* local LLM summaries
* weekly digest generation

Prioritize simplicity and quick deployment.

### Option B — Hybrid Recommendation Pipeline

Use:

* deterministic filtering
* embeddings
* local LLM reranking
* researcher profiles
* diversity-aware ranking
* feedback tracking
* weekly personalized/group digests

This may be the best balance between sophistication and maintainability.

### Option C — More Advanced Research Intelligence System

Build a more general paper knowledge base with:

* embeddings/vector search
* structured concept extraction
* researcher profiles
* paper-to-paper similarity graph
* natural-language querying
* personalized recommendations
* evolving interest profiles
* local RAG over the accumulated literature

This could represent the long-term direction rather than the immediate implementation.

For each option compare:

* development complexity
* computational requirements
* expected recommendation quality
* maintainability
* transparency/debuggability
* suitability for my hardware
* ability to scale to perhaps 5–30 group members

---

## 12. Proposed Roadmap

Conclude with a staged roadmap.

Something broadly like:

### Phase 1 — Fix discovery quality

* domain/category filters
* exclusion rules
* better query generation
* eliminate astrophysics/HEP leakage

### Phase 2 — Lab profiles

* researcher interest input
* group profile
* researcher-paper matching

### Phase 3 — Semantic recommendation

* embeddings
* reranking
* diversity selection

### Phase 4 — Local LLM pipeline

* classification
* summaries
* relevance explanations
* structured concept extraction

### Phase 5 — Weekly digest

* digest generation
* HTML/text rendering
* mailing workflow
* scheduling

### Phase 6 — Feedback and literature intelligence

* relevance feedback
* semantic search
* RAG over saved papers
* personalized recommendations

Please modify this roadmap if a different ordering makes more technical sense.

---

# Deliverable

Produce a detailed technical/design report, preferably as a Markdown file in the repository, such as:

```text
docs/lab_wide_research_discovery_design.md
```

The report should contain:

1. **Executive summary**
2. **Audit of the existing implementation**
3. **Requirements derived from the goals above**
4. **Recommended lab/researcher interest representation**
5. **Proposed paper-discovery architecture**
6. **Strategy for preventing astrophysics/particle-physics false positives**
7. **Embedding strategy**
8. **Local LLM roles within the pipeline**
9. **Current local-model recommendations for my RTX 4050/16 GB laptop and RTX 4060/32 GB desktop**
10. **Ranking and diversity strategy**
11. **Weekly reading-list architecture**
12. **Potential email-delivery architecture**
13. **Feedback/personalization strategy**
14. **Comparison of at least three implementation approaches**
15. **Recommended architecture**
16. **Suggested repository/module changes**
17. **Phased implementation roadmap**
18. **Potential risks/failure modes**
19. **Recommended experiments or benchmarks before implementation**

When recommending models, libraries, APIs, or other software that may have changed recently, research the current options rather than relying on potentially outdated assumptions.

For the ranking/filtering architecture, include pseudocode or diagrams where they clarify the design.

Where useful, propose concrete configuration schemas and example researcher profiles.

Most importantly, **do not start implementing the redesigned system during this task**. I want us to understand the design space, identify weaknesses in the current implementation, and decide on an architecture before making code changes.

At the end of the report, provide a concise recommendation answering:

> **If we wanted to turn the existing personal paper-discovery tool into a reliable lab-wide condensed-matter research intelligence system while keeping it practical on consumer hardware, what architecture should we actually build first, and why?**
