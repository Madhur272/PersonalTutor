# Extractor scrapers

This folder contains small scraper scaffolds for turning web article pages into
extractor-style chunk JSON suitable for ingestion by the KG builder and retriever.

Files:
- `byjus_scraper.py` - minimal requests+bs4 scraper for Byju's pages (best-effort)
- `vedantu_scraper.py` - minimal requests+bs4 scraper for Vedantu pages
- `toppr_scraper.py` - minimal requests+bs4 scraper for Toppr pages

Usage:

    python extractor/scrapers/byjus_scraper.py <article-url>

Notes:
- Many modern sites rely on JavaScript. These scripts use requests+bs4 and will
  only work for statically-rendered content. For JS-heavy pages, use Playwright
  or Selenium to render the page before extracting paragraphs.
- The output is written to `extractor/<site>_demo_chunks.json` and follows the
  structure expected by `retrieval/kg_builder.build_knowledge_graph`.
