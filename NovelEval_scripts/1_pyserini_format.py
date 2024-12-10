import pandas as pd

if __name__ == '__main__':
    fpath = './corpus.tsv'
    df = pd.read_csv(fpath, sep='\t', header=None, names=['id', 'contents'])
    df.to_json("corpus_formatted.jsonl", orient="records", lines=True)
