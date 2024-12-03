"""
Script for running experiments needed to generate the results in Figure 3.
Please specify the subfigure ('a' or 'b') in the variable 'subfigure'.
"""

import os
import subprocess

model_names = ["google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]
sorting_method_names = {
    "setwise": ["heapsort", "bubblesort"],
    "listwise": ["likelihood", "generation"],
}
dataset_names = ["dl19", "dl20"]

subfigure = "a"  # a or b

save_dir = f"output_files/figure3{subfigure}"
device = 0  # cuda device
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if subfigure == "a":
    num_children = [3, 5, 7, 9]
    doc_lens = [128, 85, 60, 45]
    ranking_method = "setwise"
else:
    num_children = [1, 3, 5, 7, 9]
    doc_lens = [100, 100, 100, 100]
    ranking_method = "listwise"


for dataset_name in dataset_names:
    if dataset_name == "dl20":
        ir_dataset_name = "msmarco-passage/trec-dl-2020"
    else:
        ir_dataset_name = "msmarco-passage/trec-dl-2019"
    for model_name in model_names:
        model_savename = model_name.split("/")[-1]
        for sorting_method in sorting_method_names:
            for num_child, doc_len in zip(num_children, doc_lens):
                if ranking_method == "a":
                    save_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.c={num_child}.txt"
                    out_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.c={num_child}.out"
                else:
                    save_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.r={num_child}.txt"
                    out_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.r={num_child}.out"

                save_path = os.path.join(save_dir, save_file)
                out_path = os.path.join(save_dir, out_file)

                if os.path.exists(save_path):
                    print(f"File exists.. continuing")
                    continue

                if ranking_method == "setwise":
                    command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                run --model_name_or_path {model_name} \
                    --tokenizer_name_or_path {model_name} \
                    --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                    --save_path {save_path} \
                    --ir_dataset_name {ir_dataset_name} \
                    --hits 100 \
                    --query_length 32 \
                    --passage_length {doc_len} \
                    --scoring generation \
                    --device cuda \
                {ranking_method} --method {sorting_method} \
                        --num_child {num_child} \
                        --k 10"
                elif ranking_method == "listwise":
                    command = f"CUDA_VISIBLE_DEVICES=0 python3 run.py \
                run --model_name_or_path {model_name} \
                    --tokenizer_name_or_path {model_name} \
                    --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                    --save_path {save_path} \
                    --ir_dataset_name {ir_dataset_name} \
                    --hits 100 \
                    --query_length 32 \
                    --passage_length {doc_len} \
                    --scoring {sorting_method} \
                    --device cuda \
                listwise --window_size 4 \
                        --step_size 2 \
                        --num_repeat {num_child}"
                else:
                    print(f"INVALID RANKING METHOD: {ranking_method}. CONTINUING..")

                print(f"Running command: {command}..")

                with open(out_path, "w") as f:
                    process = subprocess.run(
                        command, shell=True, stdout=f, stderr=subprocess.STDOUT
                    )
                    if process.returncode != 0:
                        print(f"Command failed with return code {process.returncode}")
