# coha-preprocess
use WLP(word, lemma, pos) lists.

## Create target word list from WLP lists
- counter\_nvjr\_from\_wlp.py: create dict contains word frequency. (only noun, verb, adj, adv. proper-noun is excluded.)
- id2word\_from\_counter.py: create id2word dict from counters. words are selected from matching words from counter dictionaries.

## Create document from WLP lists
- docs\_from\_wlp.py: create document from wlp lists. each sentence is each file from a decade. (for SPPMI)
- sentences\_from\_wlp.py: create sentences from wlp lists. (for BERT)

## Others
- ioutils.py
- utils.py
