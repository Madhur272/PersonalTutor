from preprocessing.build_retriever import hybrid_search

print('Calling hybrid_search...')
res = hybrid_search('Who was Ashoka?', k=3)
print('Returned', len(res), 'results')
for i, r in enumerate(res):
    print(i+1, r.get('doc_id') if isinstance(r, dict) else 'chunk', type(r))
