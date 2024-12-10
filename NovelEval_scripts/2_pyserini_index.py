import subprocess
import os

command = f"python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input corpus_formatted \
  --index indexes/NovelEval-index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw"

out_path = "2_pyserini_index_output.txt"
with open(out_path, "w") as f:
    process = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")