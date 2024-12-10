import subprocess

command = f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 qrels.txt run.noveleval.bm25.txt"

out_path = "4_bm25_eval.txt"
with open(out_path, "w") as f:
    process = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")