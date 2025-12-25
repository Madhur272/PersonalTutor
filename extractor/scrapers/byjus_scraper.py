"""Minimal scraper scaffold for Byju's pages.

This is a best-effort scraper and not hardened for production. It demonstrates how
to fetch an article page and chunk textual paragraphs into the extractor schema
expected by the KG builder and retriever.

Usage (example):
    python -m extractor.scrapers.byjus_scraper "https://byjus.com/example-article"

Note: some sites require JavaScript to render content; this script uses requests+bs4
and will not work for heavy JS sites. For those, consider using Playwright or Selenium.
"""
import sys
import os
import json
import requests
from bs4 import BeautifulSoup

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def page_to_chunks(url, book='Byju', chapter='article', class_level=9):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    # remove scripts/styles
    for s in soup(['script', 'style', 'noscript', 'iframe']):
        s.extract()
    article = soup.find('article') or soup.find('main') or soup
    paras = [p.get_text(separator=' ', strip=True) for p in article.find_all('p') if p.get_text(strip=True)]
    chunks = []
    for i, p in enumerate(paras, start=1):
        chunks.append({
            'chapter': chapter,
            'chunk_id': i,
            'page_number': i,
            'text': p,
            'class': class_level,
            'subject': 'English',
            'book': book
        })
    return chunks

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python byjus_scraper.py <url>')
        sys.exit(1)
    url = sys.argv[1]
    chunks = page_to_chunks(url)
    out = os.path.join(proj_root, 'extractor', 'byjus_demo_chunks.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    print(f'Wrote {len(chunks)} chunks to {out}')
