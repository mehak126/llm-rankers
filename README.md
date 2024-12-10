# CMPSC 291A: Final Project
## Jasmine Lesner and Mehak Dhaliwal

This repository contains the scripts needed to reproduce the results from the paper [A Setwise Approach for Effective and Highly Efficient Zero-shot
Ranking with Large Language Models](https://arxiv.org/pdf/2310.09497), published at SIGIR 2024. Additional experiments include (1) Evaluation on [NovelEval](https://arxiv.org/abs/2304.09542), a new ranking benchmark consisting of questions beyond the evaluated LLMs' knowledge cutoff date, and (2) Measuring the impact of instructional and conversational fine-tuning on zero-shot ranking.

### Evaluation on TREC DL 2019 and 2020
First, run the pyserini command for generating BM25 run files on TREC DL 2019 and 2020 as follows:

```bash
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl{year}-passage \
  --output run.msmarco-v1-passage.bm25-default.dl{year}.txt \
  --bm25 --k1 0.9 --b 0.4
```

Where {year} is 19 or 20 for the 2019 and 2020 datasets respectively.

We provide the output files in this directory.

Next, run the script `run_table2.py` to generate the output logs and ranking order files for the different methods in the directory [output_files/table2/](output_files/table2/) in the format `run.{model_name}.{ranking_method}.{dataset}.out` and `run.{model_name}.{ranking_method}.{dataset}.txt` respectively.

To obtain the NDCG@10 scores, run the script `run_eval_table2.py`.

### Evaluation on BEIR


### Evaluation on NovelEval


### Efficiency and Effectiveness Tradeoffs

The script `run_figure3.py` runs the experiments for different values of hyperparameters c (number of documents compared simultaneously in setwise prompting), and r (number of sliding window repetitions in listwise prompting).

For running the experiment with hyperparameter c, please set the variable `subfigure` to `a`. The output files will be saved in the directory [output_files/figure3a/](output_files/figure3a/). 

For running the experiment with hyperparameter c, please set the variable `subfigure` to `b`. The output files will be saved in the directory [output_files/figure3b/](output_files/figure3b/). 

Run the script `run_figure3.py` to generate the output logs and ranking order files for the different models and methods.  in the directory [output_files/figure4/](output_files/figure4/) in the format `run.{model_name}.{ranking_method}.{dataset}.shuffle={shuffle_method}.out` and `run.{model_name}.{ranking_method}.{dataset}shuffle={shuffle_method}.txt` respectively.

To obtain the NDCG@10 scores, run the script `run_eval_figure3.py` after setting the `subfigure` variable to `a` or `b`.

We provide the compiled results in `output_files/figure3a/3a_results.csv` and `output_files/figure3b/3b_results.csv`.


### Initial Ranking Sensitivity

Run the script `run_figure4.py` to generate the output logs and ranking order files for the different models and methods in the directory [output_files/figure4/](output_files/figure4/) in the format `run.{model_name}.{ranking_method}.{dataset}.shuffle={shuffle_method}.out` and `run.{model_name}.{ranking_method}.{dataset}shuffle={shuffle_method}.txt` respectively.

To obtain the NDCG@10 scores, run the script `run_eval_figure4.py`.

We provide the compiled results in `output_files/figure4/figure4_results.csv`.


### Impact of Fine-Tuning

Run the script `run_table4.py` to generate the output logs and ranking order files for the different models and methods in the directory [output_files/table4/](output_files/table4/) in the format `run.{model_name}.{ranking_method}.{dataset}.out` and `run.{model_name}.{ranking_method}.{dataset}.txt` respectively.

To obtain the NDCG@10 scores, run the script `run_eval_table4.py`.


### Plotting figures
