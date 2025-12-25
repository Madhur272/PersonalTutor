import sys, os
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj not in sys.path:
    sys.path.insert(0, proj)

from pipeline import segment_pipeline as sp
import generation.generate_answers as ga_pkg
import sys


def fake_generate_answer(query, retrieved_chunks, chat_history=None):
    print('==== FAKE_GENERATE_CALLED ====')
    print('query:', query)
    print('chat_history:', chat_history)
    print('retrieved_chunks:')
    for i,c in enumerate(retrieved_chunks):
        print(i, c)
    return 'OK'

# patch both module references
ga_pkg.generate_answer = fake_generate_answer
if 'generate_answers' in sys.modules:
    sys.modules['generate_answers'].generate_answer = fake_generate_answer

# simulate conversation
mem = []
q1 = 'Who was Margie?'
mem.append({'user': q1})
print('\n-- Running Q1 --')
print(sp.run_pipeline(q1, chat_history=mem))
q2 = 'How old was she?'
mem.append({'user': q2})
print('\n-- Running Q2 --')
print(sp.run_pipeline(q2, chat_history=mem))
