# generate pseudowords with semantic change

## obtain pair of words
- obtain\_wordpair.py

## replace 
- replace\_words.py
- replace\_with\_rate.py
- convert\_freq\_replaced.py

## others
- ioutils.py
- matrixutils.py

## Example
### 0. train cooccur/word vectors using target documents
- Cooccur matrix
- PPMI matrix
- PMI-SVDjoint, c
- Word2Vec with alignment
- Dynamic Word Embeddings

### 1. obtain pair of words
```
$python3 obtain_wordpair.py --word_matrices {matrix_1} {matrix_2} \
  --id2word {id2word.pkl} \
  > wordpairs.txt
```

### 2. replace corpus with 50 pairs of words
#### 2.a replace all
```
$python3 replace_words.py --file_path {target_corpus} \
  --wordpair_list {wordpairs.txt}

$python3 convert_freq_replaced.py --word2freq {word2freq_target} \
  --wordpair_list {wordpairs.txt}
```

#### 2.b with rate (0.1, 0.2, 0.3, 0.4, 0.5)

```
$python3 replace_with_rate.py --file_path {target_corpus} \
  --word_to_freq {word2freq_target.pkl} \
  --wordpair_list {wordpairs.txt} 
```


