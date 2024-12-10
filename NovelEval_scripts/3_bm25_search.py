
import subprocess

command = f"python -m pyserini.search.lucene \
  --threads 16 \
  --batch-size 128 \
  --index indexes/NovelEval-index \
  --topics queries.tsv \
  --output run.noveleval.bm25.txt \
  --bm25 --k1 0.9 --b 0.4"

out_path = "3_bm25_search.txt"
with open(out_path, "w") as f:
    process = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")