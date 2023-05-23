# Defined in Section 4.6.4

import torch
from vocab import Vocab
import sys

# 词表映射
def load_sentence_polarity():
    from nltk.corpus import sentence_polarity
    # import nltk 
    # nltk.download('sentence_polarity') # windows C:\nltk_data\corpora\sentence_polarity

    vocab = Vocab.build(sentence_polarity.sents())

    # ([23, 2444, 61, 9851, 76, 308, 23, 1664, 14509, 496, 219, 14510, 219, 4, 27, 175, 363, 76, 29, 32, 5884, 201, 7984, 73, 5354, 4219, 2, 14511, 1204, 2701, 25, 2184, 14512, 6], 0)
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[:4000]]
    

    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data, test_data, vocab

def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask

def load_treebank():
    from nltk.corpus import treebank
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocab.build(sents, reserved_tokens=["<pad>"])

    tag_vocab = Vocab.build(postags)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab

