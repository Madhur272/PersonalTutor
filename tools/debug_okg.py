import sys, os
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj not in sys.path:
    sys.path.insert(0, proj)

from retrieval.kg_query import OnlineKGQuery

class FakeResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

sample_html = '''<html><head><title>Byju's Sample</title></head>
<body>
<article>
<h1>Understanding Newton's First Law</h1>
<p>Newton's first law states that an object in motion stays in motion unless acted upon by a net external force.</p>
<p>This is a key concept in classical mechanics.</p>
</article>
</body></html>'''

import requests
requests.get = lambda url, timeout=10, headers=None: FakeResp(sample_html)

okg = OnlineKGQuery()
res = okg.query_with_search("what is newton's first law", search_func=lambda q,n=5: ['https://byjus.example.com/sample-lesson'])
print('OKG returned', len(res), 'contexts')
for i,c in enumerate(res[:5]):
    print(i, c)
