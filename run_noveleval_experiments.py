import os
import subprocess

model_names = ["google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]

ranking_method_names = ["setwise", "pointwise", "listwise", "pairwise"]
sorting_method_names = {
    "pairwise": ["heapsort", "bubblesort", "allpair"],
    "setwise": ["heapsort", "bubblesort"],
    "pointwise": ["yes_no", "qlm"],
    "listwise": ["likelihood", "generation"],
}
device = 0

run_path = "run.noveleval.bm25.txt"
pyserini_index = "./NovelEval"


for model_name in model_names:
    for ranking_method in ranking_method_names:
        for sorting_method in sorting_method_names[ranking_method]:
            model_savename = model_name.split("/")[-1]
            save_path = (
                f"run.noveleval.{model_savename}.{ranking_method}.{sorting_method}.txt"
            )
            out_path = (
                f"run.noveleval.{model_savename}.{ranking_method}.{sorting_method}.out"
            )

            print(f"{save_path}")

            if os.path.exists(save_path):
                print(f"File exists.. continuing")
                continue

            if ranking_method == "pairwise" or ranking_method == "setwise":
                command = f"CUDA_VISIBLE_DEVICES={device} python3 noveleval_run.py \
                run --model_name_or_path {model_name} \
                    --tokenizer_name_or_path {model_name} \
                    --run_path  {run_path}\
                    --save_path {save_path} \
                    --pyserini_index {pyserini_index} \
                    --hits 100 \
                    --query_length 32 \
                    --passage_length 128 \
                    --scoring generation \
                    --device cuda \
                {ranking_method} --method {sorting_method} \
                        --k 10"
            elif ranking_method == "listwise":
                command = f"CUDA_VISIBLE_DEVICES={device} python3 noveleval_run.py \
              run --model_name_or_path {model_name} \
                  --tokenizer_name_or_path {model_name} \
                  --run_path {run_path} \
                  --save_path {save_path} \
                  --pyserini_index {pyserini_index} \
                  --hits 100 \
                  --query_length 32 \
                  --passage_length 100 \
                  --scoring {sorting_method} \
                  --device cuda \
              listwise --window_size 4 \
                       --step_size 2 \
                       --num_repeat 5"
            elif ranking_method == "pointwise":
                command = f"CUDA_VISIBLE_DEVICES={device} python3 noveleval_run.py \
              run --model_name_or_path {model_name} \
                  --tokenizer_name_or_path {model_name} \
                  --run_path {run_path} \
                  --save_path {save_path} \
                  --pyserini_index {pyserini_index} \
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

print("FIN.")
