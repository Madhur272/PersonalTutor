import urllib.parse
from typing import List

def byjus_search(query: str, num_results: int = 5) -> List[str]:
    """Search Byju's for `query` and return a list of result URLs.

    Note: This is a lightweight site-specific adapter that queries the
    public Byju's search URL pattern and scrapes result links. It is a
    best-effort implementation and should be replaced with an official
    API if one is available.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception as e:
        raise ImportError("byjus_search requires requests and beautifulsoup4. Install them with: pip install requests beautifulsoup4") from e

    encoded = urllib.parse.quote_plus(query)
    # Best-effort search endpoint; if Byju's layout changes this may need updates
    search_url = f"https://byjus.com/search/?q={encoded}"
    resp = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0 (compatible)"}, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # heuristics: collect anchors that look like article links
    anchors = soup.find_all('a', href=True)
    urls = []
    for a in anchors:
        href = a['href']
        # skip javascript and mailto
        if href.startswith('javascript:') or href.startswith('mailto:'):
            continue
        # make absolute if relative
        if href.startswith('/'):
            href = urllib.parse.urljoin('https://byjus.com', href)
        # only include byjus domain links for safety
        if 'byjus.com' in urllib.parse.urlparse(href).netloc or href.startswith('https://byjus.com'):
            if href not in urls:
                urls.append(href)
        if len(urls) >= num_results:
            break

    return urls
