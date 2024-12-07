import subprocess
import os

root = './output_files/table1'
log_file = './output_files/table1/eval_output_allpair.log'


model_names = ['google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl']
dataset_names = ['dl19', 'dl20']
# ranking_method_names = ['pairwise', 'setwise', 'pointwise', 'listwise']
ranking_method_names = ['pairwise']
sorting_method_names = {
    'pairwise': ['allpair'], #['heapsort', 'bubblesort'],
    'setwise': ['heapsort', 'bubblesort'],
    'pointwise': ['yes_no', 'qlm'],
    'listwise': ['likelihood', 'generation']
}

with open(log_file, 'w') as log:
    for dataset_name in dataset_names:
        print(f"DATA: {dataset_name}")
        for model_name in model_names:
            print(f"MODEL: {model_name}")
            model_savename = model_name.split('/')[-1]
            for ranking_method in ranking_method_names:
                print(f"RANKING METHOD: {ranking_method}")
                for sorting_method in sorting_method_names[ranking_method]:
                    print(f"SORTING METHOD: {sorting_method}")
                    if dataset_name == 'dl20':
                        fname = f"run.{model_savename}.{ranking_method}.{sorting_method}.{dataset_name}.txt"
                    else:
                        fname = f"run.{model_savename}.{ranking_method}.{sorting_method}.txt"
                    fpath = os.path.join(root, fname)

                    command = f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 {dataset_name}-passage {fpath}"
                    print(f"Running command: {command}..")

                    # process = subprocess.run(command, shell=True, stderr=subprocess.STDOUT)
                    process = subprocess.run(command, shell=True, capture_output=True, text=True)

                    log.write(f"DATASET: {dataset_name}\n")
                    log.write(f"MODEL: {model_name}\n")
                    log.write(f"RANKING METHOD: {ranking_method}\n")
                    log.write(f"SORTING METHOD: {sorting_method}\n")
                    log.write(f"COMMAND: {command}\n")
                    log.write(f"RETURN CODE: {process.returncode}\n")
                    log.write(f"STDOUT:\n{process.stdout}\n")
                    log.write(f"STDERR:\n{process.stderr}\n")
                    log.write("="*50 + "\n")
                    if process.returncode != 0:
                        print(f"Command failed with return code {process.returncode}")
print("FIN.")


