"""Minimal Vedantu scraper scaffold.

This file provides a small helper to fetch a Vedantu article page and chunk it into
extractor-style JSON. Vedantu pages may require JS in some cases; this is a simple
requests+bs4 approach that works for static pages.
"""
import sys
import os
import json
import requests
from bs4 import BeautifulSoup

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def page_to_chunks(url, book='Vedantu', chapter='article', class_level=9):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
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
        print('Usage: python vedantu_scraper.py <url>')
        sys.exit(1)
    url = sys.argv[1]
    out = os.path.join(proj_root, 'extractor', 'vedantu_demo_chunks.json')
    chunks = page_to_chunks(url)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    print(f'Wrote {len(chunks)} chunks to {out}')
