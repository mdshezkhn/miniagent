# Reverse-Engineering Big Assistants — REA method applied

Method (works on any software product): **Observe → Hypothesize → Verify → Extract → Implement.**
Watch the product's observable behavior (inputs, outputs, timing, network calls),
hypothesize the mechanism, verify against public engineering talks/docs/source,
extract the *pattern* (never the code), implement it yourself.

## 1. Perplexity — the cited answer engine

| Phase | Observation | Reverse-engineered mechanism |
|---|---|---|
| Input | question typed → brief "searching" states | query REWRITTEN into optimized search queries first |
| Retrieval | multiple source chips appear fast | parallel search over own index (Vespa) + live crawl |
| Ranking | ~3-8 passages get cited | passages ranked (relevance/freshness/domain trust) |
| Synthesis | answer with [1][2] inline numbers | LLM writes FROM the passage set, citation ids enforced by prompt |
| Extras | related questions, focus modes | follow-up query generation; scoped indexes (news/academic/Reddit) |

**Extracted patterns → miniagent:** `web_search` + cite-sources prompt rule (v0.4 ✅).
Still missing: query rewriting (prompt rule below), related questions.

## 2. ChatGPT — memory + tool suite

| Phase | Observation | Reverse-engineered mechanism |
|---|---|---|
| Memory | "saved a memory" toast; facts resurface weeks later | facts stored as rows; INJECTED into system context each turn; a background "consolidation/dreaming" pass prunes & reorganizes |
| Tools | browses/executes code/draws on demand | model picks tools from schemas; results appended (a loop — same as ours) |
| Streaming | text streams, then tool cards | SSE-style delta streaming; tool events as separate frame types |

**Extracted → miniagent:** `remember` + memory.md injection (v0.4 ✅), streaming + tool
cards (v0.2 GUI ✅). Missing: memory pruning/consolidation.

## 3. Claude Code — context engineering + extensibility

| Phase | Observation | Reverse-engineered mechanism |
|---|---|---|
| Project awareness | knows repo conventions unprompted | CLAUDE.md/AGENTS.md read from cwd into system prompt |
| /commands | typed /foo expands into big prompt | markdown files on disk loaded as prompts |
| Subagents | "researching..." parallel contexts | separate agent instances with own context, results summarized back |
| Skills | domain knowledge appears on demand | md manuals loaded lazily, only when relevant |

**Extracted → miniagent:** AGENTS.md auto-context (v0.4 ✅), skills-on-demand pattern
(hard-coded doc clipping for now). NEW in v0.5: slash commands.

## v0.5 additions from this pass

1. **Slash commands** (Claude Code): `.miniagent/commands/<name>.md` → `/name` in the REPL.
2. **Query-rewrite prompt rule** (Perplexity): search guidance added to system prompt —
   rewrite the need into 1-3 precise queries before calling web_search.

## The meta-lesson

Every "magical" feature reduced to one of three primitives:
a TOOL, a PROMPT RULE, or a FILE READ INTO CONTEXT. No product required changing
the loop. Your agent is now proof: five "big features" shipped, loop untouched.
