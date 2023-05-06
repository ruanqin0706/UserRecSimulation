from transformers import pipeline, set_seed
import gc
import sys

set_seed(42)

pipe = pipeline("summarization", model=sys.argv[3], device=0)

with open(sys.argv[1], 'r') as f, open(sys.argv[2], 'w') as f_write:
    for line in f:
        try:
            nid, article = line.rstrip("\n").split("***sep***")
            pipe_out = pipe(article[:1020])
            abstract = pipe_out[0]["summary_text"].replace(" .<n>", ". ")
            print(nid, abstract, sep='***sep***', file=f_write)

        except Exception as e:
            print(f'ignore {line.split("***sep***")[0]}')

        gc.collect()
