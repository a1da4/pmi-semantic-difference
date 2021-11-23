# coha-preprocess
use WLP(word, lemma, pos) lists.

## Create target word list from WLP lists
- counter\_nvjr\_from\_wlp.py: create dict contains word frequency. (only noun, verb, adj, adv. proper-noun is excluded.)
- id2word\_from\_counter.py: create id2word dict from counters. words are selected from matching words from counter dictionaries.

## Create document from WLP lists
- docs\_from\_wlp.py: create document from wlp lists. each sentence is each file from a decade. 
- sentences\_from\_wlp.py: create sentences from wlp lists. 

## Others
- ioutils.py
- utils.py

## Example
### 1. Count frequency in each document
```
$python3 counter_nvjr_from_wlp.py --data_dir {dir_1900s}

$python3 counter_nvjr_from_wlp.py --data_dir {dir_1990s}
```

### 2. Obtain shared vocab and id2word dict
```
$python3 id2word_from_counter.py --count_dic {counter_1900s} {counter_1990s} \
  --threshold 100
```

### 3. Create document/sentences from wlp lists
#### 3.1 Documents
```
$python3 docs_from_wlp.py --data_dir {dir_1900s}

$python3 docs_from_wlp.py --data_dir {dir_1990s} 
```

#### 3.2 Sentences
```
$python3 sentences_from_wlp.py --data_dir {dir_1900s}

$python3 sentences_from_wlp.py --data_dir {dir_1990s}
```
