import sys, os
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj not in sys.path:
    sys.path.insert(0, proj)
from pipeline.segment_pipeline import run_pipeline
import preprocessing.build_retriever as br
# monkeypatch hybrid_search
br.hybrid_search = lambda q, k=5: []
# prepare fake requests
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
# monkeypatch generate_answer to inspect retrieved_chunks
import generation.generate_answers as ga_pkg
import sys

def fake_generate_answer(query, retrieved_chunks):
    print('--- FAKE_GENERATE_CALLED with retrieved_chunks ---')
    for i,c in enumerate(retrieved_chunks):
        print(i, 'TEXT:', repr(c.get('text',''))[:200])
    return '\n'.join([c.get('text','') for c in retrieved_chunks])

# Patch both the packaged and top-level module objects so pipeline picks up the stub
ga_pkg.generate_answer = fake_generate_answer
if 'generate_answers' in sys.modules:
    sys.modules['generate_answers'].generate_answer = fake_generate_answer
# run pipeline
resp = run_pipeline("what is newton's first law", search_func=lambda q,n=5: ['https://byjus.example.com/sample-lesson'])
print('\n---- RESPONSE ----\n', resp)
