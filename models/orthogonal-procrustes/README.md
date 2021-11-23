# Orthogonal Procrustes ([Hamilton2016](https://aclanthology.org/P16-1141/))

- train.py: train word2vec
- rotate.py: compute rotate matrix and align word embeddings
 
## Example: Word2Vec with alignment (Hamilton2016)

### 1. train word2vec models
```
$pwd
> ~/semantic-change-embed/
$python3 train.py --file_path {doc_file_1} \
  --dim 100 --min_count 100 --window_size 4 \
  --negative 5 --seed 1 

$python3 train.py --file_path {doc_file_2} \
  --dim 100 --min_count 100 --window_size 4 \
  --negative 5 --seed 1 
```

### 2. alignment
```
$python3 rotate.py --model_path {model_1} {model_2} \
  --id_to_word {id2word.pkl}
```
