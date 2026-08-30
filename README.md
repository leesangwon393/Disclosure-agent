# Disclosure-agent
공시자료를 근거로 사용자 질의에 답변하는 공시 에이전트




┌──────────────────── OFFLINE ────────────────────┐

DART
  ↓
Parsing
  ↓
Parent-Child Chunking
  ↓
Correction + Metadata
  │
  ├────────→ BM25 / Dense / Sparse Index ─────────┐
  │                                               │
  └────────→ Section Card                         │
             └─ Section                           │
             └─ Corpus Keywords                   │
                      │                           │
└──────────────────── │───────────────────────────│
                      │                           │
                      ▼                           ▼

┌──────────────────── ONLINE ───────────────────────────────┐

                 User Question
                       ↓
               Entity Resolver
                  "어느 회사?"
                       ↓
                Query Planner ◀──────────────┐
        "무엇을·어디서·어떻게 찾지?"                │
             ↑         ↑                     │
             │         │                     │
      Section Card   Corpus Keywords         │
             │                               │
             └──────────┐                    │
                        ↓                    │
                Metadata Filter              │
                        ↓                    │
              Hybrid Retrieval  ◀──── Index  │
                        ↓                    │
               Fusion / Reranker             │
                        ↓                    │
              Evidence Organizer             │
                        ↓                    │
                Coverage Checker             │
                  /             \
          충분함 /                 \ 부족함
              ↓                     ↓
        Route Logic          Missing Evidence
              ↓                     ↓
        Evidence Pack              Replan
              ↓                     │
             HCX                    └──────────→ Query Planner
              ↓
          Validator
              ↓
     Final Answer + Citation

└───────────────────────────────────────────────────────────┘
