# 본 문서는 아래 링크 문서를 코드화한 문서입니다.
# https://bab2min.tistory.com/585

import re
import numpy as np
import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer


class DTMGenerator:

    def __init__(self, df, stopwords, stemmer, is_lower=True):
        self.df = df
        self.n_d = len(df)
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.is_lower = is_lower

    def forward(self, sub_func, tokenizer, is_df=False):
        self._build_vocab(sub_func, tokenizer)
        self._build_dtm(sub_func, tokenizer)
        if is_df:
            return pd.DataFrame(
                data=self.DTM,
                index=range(self.n_d),
                columns=self.vocab,
            )
        return self.DTM

    def _build_vocab(self, sub_func, tokenizer):
        self.vocab = []
        for data in self.df:
            if self.is_lower:
                data = data.lower()
            data = tokenizer(sub_func(data))
            for word in data:
                word = self.stemmer.stem(word)
                if word in self.vocab:
                    continue
                elif word in self.stopwords:
                    continue
                else:
                    self.vocab.append(word)
        self.n_v = len(self.vocab)

    def _build_dtm(self, sub_func, tokenizer):
        DTM = np.zeros((self.n_d, self.n_v), dtype=np.int8)
        for r, doc in enumerate(self.df):
            if self.is_lower:
                doc = doc.lower()
            data = tokenizer(sub_func(doc))
            data = [self.stemmer.stem(word) for word in data]
            for c, voc in enumerate(self.vocab):
                if voc in data:
                    DTM[r, c] += 1
        self.DTM = DTM


if __name__ == '__main__':
    dataset = [
        'Cute kitty',
        'Eat rice or cake',
        'Kitty and hamster',
        'Eat bread',
        'Rice, bread and cake',
        'Cute hamster eats bread and cake',
    ]

    sub_func = lambda s: re.sub(',', '', s)
    tokenizer = lambda s: s.split(' ')
    stopwords = ['or', 'and']
    stemmer = PorterStemmer()

    dtmgen = DTMGenerator(dataset, stopwords, stemmer)
    DTM = dtmgen.forward(sub_func, tokenizer, True)
    print(DTM)

    #    cute  kitti  eat  rice  cake  hamster  bread
    # 0     1      1    0     0     0        0      0
    # 1     0      0    1     1     1        0      0
    # 2     0      1    0     0     0        1      0
    # 3     0      0    1     0     0        0      1
    # 4     0      0    0     1     1        0      1
    # 5     1      0    1     0     1        1      1
    