import pandas as pd
import json

queries = pd.read_csv('queries.tsv', sep='\t', header=None, names=['query_id', 'query'])

# Convert it to a topics format
topics = {}
for _, row in queries.iterrows():
    topic_id = str(row['query_id'])
    query_text = row['query']
    topics[topic_id] = {'title': query_text}

query_map = {}
for topic_id, topic in topics.items():
    text = topic['title']
    query_map[str(topic_id)] = text

print(query_map)

# save
with open('NovelEval-test', 'w') as f:
    for topic_id, query_text in query_map.items():
        json.dump({"id": topic_id, "query": query_text}, f)
        f.write('\n')
