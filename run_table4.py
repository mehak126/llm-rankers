import os
import subprocess

model_names = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-2-7b-hf",
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
    "lmsys/vicuna-13b-v1.5",
]
ranking_method_names = ["setwise", "listwise", "pairwise"]
sorting_method_names = {
    "pairwise": ["heapsort", "bubblesort"],
    "setwise": ["heapsort", "bubblesort"],
    "pointwise": ["yes_no", "qlm"],
    "listwise": ["likelihood", "generation"],
}
dataset_names = ["dl19", "dl20"]
save_dir = "output_files/table4"
device = 0  # cuda device

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for dataset_name in dataset_names:
    if dataset_name == "dl20":
        ir_dataset_name = "msmarco-passage/trec-dl-2020"
    else:
        ir_dataset_name = "msmarco-passage/trec-dl-2019"
    for model_name in model_names:
        for ranking_method in ranking_method_names:
            for sorting_method in sorting_method_names[ranking_method]:
                model_savename = model_name.split("/")[-1]
                save_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.txt"
                out_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.out"

                save_path = os.path.join(save_dir, save_file)
                out_path = os.path.join(save_dir, out_file)

                if os.path.exists(save_path):
                    print(f"File exists.. continuing")
                    continue

                if ranking_method == "pairwise" or ranking_method == "setwise":
                    command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path  run.msmarco-v1-passage.bm25-default.{dataset_name}.txt\
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 128 \
                        --scoring generation \
                        --device cuda \
                    {ranking_method} --method {sorting_method} \
                            --k 10"
                elif ranking_method == "listwise":
                    command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt\
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 100 \
                        --scoring {sorting_method} \
                        --device cuda \
                    listwise --window_size 4 \
                            --step_size 2 \
                            --num_repeat 5"
                elif ranking_method == "pointwise":
                    command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 128 \
                        --device cuda \
                    pointwise --method {sorting_method} \
                                --batch_size 32"
                else:
                    print(f"INVALID RANKING METHOD: {ranking_method}. CONTINUING..")

                print(f"Running command: {command}..")

                with open(out_path, "w") as f:
                    process = subprocess.run(
                        command, shell=True, stdout=f, stderr=subprocess.STDOUT
                    )
                    if process.returncode != 0:
                        print(f"Command failed with return code {process.returncode}")
