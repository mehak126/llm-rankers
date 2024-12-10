import os
import subprocess

root = "./output_files/table4"
log_file = "./output_files/table4/eval_output_vicuna.log"


model_names = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "lmsys/vicuna-13b-v1.5",
    "meta-llama/Llama-2-13b-hf",
    "lmsys/vicuna-7b-v1.5",
    "meta-llama/Llama-2-7b-hf",
]
dataset_names = ["dl19", "dl20"]
ranking_method_names = ["setwise", "pairwise", "listwise"]
sorting_method_names = {
    "pairwise": ["heapsort", "bubblesort"],
    "setwise": ["heapsort", "bubblesort"],
    "pointwise": ["yes_no", "qlm"],
    "listwise": ["generation", "likelihood"],
}

with open(log_file, "w") as log:
    for dataset_name in dataset_names:
        print(f"DATA: {dataset_name}")
        log.write(f"DATASET: {dataset_name}\n")
        for model_name in model_names:
            print(f"MODEL: {model_name}")
            log.write(f"MODEL: {model_name}\n")
            model_savename = model_name.split("/")[-1]
            for ranking_method in ranking_method_names:
                print(f"RANKING METHOD: {ranking_method}")
                for sorting_method in sorting_method_names[ranking_method]:
                    print(f"SORTING METHOD: {sorting_method}")
                    fname = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.txt"
                    fpath = os.path.join(root, fname)

                    command = f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 {dataset_name}-passage {fpath}"
                    print(f"Running command: {command}..")

                    process = subprocess.run(
                        command, shell=True, capture_output=True, text=True
                    )
                    output = process.stdout
                    # log.write(f"{output.split(' ')[-1]}")
                    log.write(f"{output}")
                    if process.returncode != 0:
                        print(f"Command failed with return code {process.returncode}")
