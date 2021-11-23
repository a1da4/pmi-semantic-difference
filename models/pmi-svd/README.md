# PMI-SVD joint, constrained (Proposal)

- sppmi-svd/: PMI-SVD
- constrained/: PMI-SVDc
- save\_joint\_pmi.py: PMI-SVDjoint

## 1. obtain (shifted)ppmi matrix from [pmi-svd](https://github.com/a1da4/sppmi-svd/tree/d1648f59a650caafec2f3de7ac30c9aed2a87e75)
```
$cd sppmi-svd/

$python3 main.py --file_path {doc_file_1} \
  --pickle_id2word {id2word.pkl} \
  --threshold 0 --has_cds --window_size 4 --shift 1

$python3 main.py --file_path {doc_file_2} \
  --pickle_id2word {id2word.pkl} \
  --threshold 0 --has_cds --window_size 4 --shift 1
```

## 2a (PMI-SVDjoint) decompose vertically stacked pmi matrices
```
$python3 save_joint_pmi.py --dic_id2word {id2word.pkl} \
  --path_models {ppmi_1} {ppmi_2} \
  --dim 100
```

## 2b (PMI-SVDconstrained) search best param or train directly

### 2b-1 search best param {tau}
```
$cd constrained/

$python3 search.py --ppmi_pathes {ppmi_1}, {ppmi_2} \
  --seed 1 --dim 100 --n_iter 50 \
  --es 3 --id_to_word {id2word.pkl} \
  --dev_list {dev_word_list} 
```

### 2b-2 train model directly
```
$cd constrained/

$python3 train.py --ppmi_pathes {ppmi_1} {ppmi_2} \
  --seed 1 --dim 100 --n_iter 50 \
  --tau 1 --es 3 --id_to_word {id2word.pkl} \
  --dev_list {dev_word_list}
```
