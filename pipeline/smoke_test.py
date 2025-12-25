import os
import sys

# ensure project root is on sys.path so 'pipeline' package can be resolved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.segment_pipeline import run_pipeline


if __name__ == '__main__':
    print(run_pipeline('Please correct the grammar of: She go to school yesterday.'))
