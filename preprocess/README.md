# Preprocess
- coha-preprocess/: preprocess COHA dataset
- pseudowords/: generate pseudowords with semantic change

## Example
### 1. Count word frequency in each corpus
```
$python3 counter_from_doc.py --file_path {corpus_1}

$python3 counter_from_doc.py --file_path {corpus_2}
```
### 2. Obtain shared vocab and id2word
```
id2word_from_counter.py --count_dic {counter_1} {counter_2} \
  --threshold 100
```
