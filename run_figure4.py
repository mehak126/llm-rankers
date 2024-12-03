import os
import subprocess

model_name = "google/flan-t5-large"
dataset_names = ["dl19", "dl20"]
ranking_method_names = ["listwise", "setwise", "pairwise"]
sorting_method_names = {
    "pairwise": ["heapsort", "bubblesort"],
    "setwise": ["heapsort", "bubblesort"],
    "listwise": ["likelihood", "generation"],
}
shuffle_ranking_methods = ["none", "random", "inverse"]
save_dir = f"output_files/figure4_temp"
device = 0

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for dataset_name in dataset_names:
    if dataset_name == "dl20":
        ir_dataset_name = "msmarco-passage/trec-dl-2020"
    else:
        ir_dataset_name = "msmarco-passage/trec-dl-2019"
    model_savename = model_name.split("/")[-1]
    for ranking_method in ranking_method_names:
        print(f"RANKING METHOD: {ranking_method}")
        for sorting_method in sorting_method_names[ranking_method]:
            print(f"SORTING METHOD: {sorting_method}")
            for shuffle_ranking in shuffle_ranking_methods:
                save_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.shuffle={shuffle_ranking}.txt"
                out_file = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.shuffle={shuffle_ranking}.out"

                save_path = os.path.join(save_dir, save_file)
                out_path = os.path.join(save_dir, out_file)

                if os.path.exists(save_path):
                    print(f"File exists.. continuing")
                    continue

                if ranking_method == "setwise":
                    if shuffle_ranking == "none":
                        command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 128 \
                        --scoring generation \
                        --device cuda \
                    {ranking_method} --method {sorting_method} \
                            --num_child 4 \
                            --k 10"
                    else:
                        command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 128 \
                        --scoring generation \
                            --shuffle_ranking {shuffle_ranking} \
                        --device cuda \
                    {ranking_method} --method {sorting_method} \
                            --num_child 4 \
                            --k 10"
                elif ranking_method == "listwise":
                    if shuffle_ranking == "none":
                        command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 128 \
                        --scoring {sorting_method} \
                        --device cuda \
                    listwise --window_size 4 \
                            --step_size 2 \
                            --num_repeat 5"
                    else:
                        command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 128 \
                        --scoring {sorting_method} \
                        --shuffle_ranking {shuffle_ranking} \
                        --device cuda \
                    listwise --window_size 4 \
                            --step_size 2 \
                            --num_repeat 5"
                elif ranking_method == "pairwise":
                    if shuffle_ranking == "none":
                        command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                run --model_name_or_path {model_name} \
                    --tokenizer_name_or_path {model_name} \
                    --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                    --save_path {save_path} \
                    --ir_dataset_name {ir_dataset_name} \
                    --hits 100 \
                    --query_length 32 \
                    --passage_length 128 \
                    --scoring generation \
                    --device cuda \
                    {ranking_method} --method {sorting_method} \
                --k 10"
                    else:
                        command = f"CUDA_VISIBLE_DEVICES={device} python3 run.py \
                    run --model_name_or_path {model_name} \
                        --tokenizer_name_or_path {model_name} \
                        --run_path run.msmarco-v1-passage.bm25-default.{dataset_name}.txt \
                        --save_path {save_path} \
                        --ir_dataset_name {ir_dataset_name} \
                        --hits 100 \
                        --query_length 32 \
                        --passage_length 128 \
                        --shuffle_ranking {shuffle_ranking} \
                        --scoring generation \
                        --device cuda \
                        {ranking_method} --method {sorting_method} \
                    --k 10"
                else:
                    print(f"INVALID RANKING METHOD: {ranking_method}. CONTINUING..")

                print(f"Running command: {command}..")

                with open(out_path, "w") as f:
                    process = subprocess.run(
                        command, shell=True, stdout=f, stderr=subprocess.STDOUT
                    )
                    if process.returncode != 0:
                        print(f"Command failed with return code {process.returncode}")
