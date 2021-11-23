# Dynamic Word Embeddings ([Yao2018](https://arxiv.org/abs/1605.09096))

- model.py: Yao's model
- search.py: grid search best parameters (lambda, tau, gamma)
- train.py: train Yao's model directly without searching parameters
- util.py: util of Yao's model

## 1. obtain (shifted)ppmi matrix from [pmi-svd](https://github.com/a1da4/sppmi-svd/tree/d1648f59a650caafec2f3de7ac30c9aed2a87e75)
```
$python3 main.py --file_path {doc_file_1} \
  --pickle_id2word {id2word.pkl} \
  --threshold 0 --has_cds --window_size 4 --shift 1

$python3 main.py --file_path {doc_file_2} \
  --pickle_id2word {id2word.pkl} \
  --threshold 0 --has_cds --window_size 4 --shift 1
```

## 2. search best param or train directly

### 2.1 search best params {lambda, tau, gamma}
```
$python3 search.py --ppmi_pathes {ppmi_1} {ppmi_2} \
  --seed 1 --dim 100 --n_iter 50 \
  --es 3 --id_to_word {id2word.pkl} \
  --dev_list {dev_word_list}
```

### 2.2 train model directly
```
$python3 train.py --ppmi_pathes {ppmi_1} {ppmi_2} \
  --seed 1 --dim 100 --n_iter 50 \
  --lam 0.001 --tau 100 --gam 0.001 \
  --es 3 --id_to_word {id2word.pkl} \
  --dev_list {dev_word_list}
```
