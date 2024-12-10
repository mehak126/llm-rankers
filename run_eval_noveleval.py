import os
import subprocess

root = "./output_files/noveleval"
log_file = "./output_files/noveleval/eval_output_allpair.log"


model_names = ["google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]
# ranking_method_names = ["listwise", "pairwise", "setwise", "pointwise"]
ranking_method_names = ["pairwise"]
sorting_method_names = {
    # 'pairwise': ['heapsort', 'bubblesort', 'allpair'],
    "pairwise": ["allpair"],
    "setwise": ["heapsort", "bubblesort"],
    "pointwise": ["yes_no", "qlm"],
    "listwise": ["likelihood", "generation"],
}


with open(log_file, "w") as log:
    for model_name in model_names:
        print(f"MODEL: {model_name}")
        model_savename = model_name.split("/")[-1]
        for ranking_method in ranking_method_names:
            print(f"RANKING METHOD: {ranking_method}")
            for sorting_method in sorting_method_names[ranking_method]:
                print(f"SORTING METHOD: {sorting_method}")
                fname = f"run.noveleval.{model_savename}.{ranking_method}.{sorting_method}.txt"

                fpath = os.path.join(root, fname)
                if not os.path.isfile(fpath):
                    print(f"{fpath} doesn't exist. Continuing..")
                    continue

                command = f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 qrels.txt {fpath}"
                print(f"Running command: {command}..")

                process = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )
                log.write(f"MODEL: {model_name}\n")
                log.write(f"RANKING METHOD: {ranking_method}\n")
                log.write(f"SORTING METHOD: {sorting_method}\n")
                log.write(f"COMMAND: {command}\n")
                log.write(f"RETURN CODE: {process.returncode}\n")
                log.write(f"STDOUT:\n{process.stdout}\n")
                log.write(f"STDERR:\n{process.stderr}\n")
                log.write("=" * 50 + "\n")
                if process.returncode != 0:
                    print(f"Command failed with return code {process.returncode}")

print("FIN.")
