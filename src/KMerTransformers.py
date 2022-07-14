from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from tqdm.auto import tqdm
from itertools import permutations


class KMerTransformer(TransformerMixin, BaseEstimator):
    """
    Sklearn Pipeline Transformer to create k-mer frequency columns.
    
    Parameters:
        k (int) : length of characters in each substring of the input DNA sequence.
        normalize (bool): True will scale each frequency by the total length of the input DNA sequence.
    
    Output:
        4^k number of columns for the frequency of each k-mer in the input DNA sequence.
    
    """
    
    def __init__(self, k, normalize=False, verbose=True):
        self.k = k
        self.normalize = normalize
        self.verbose = verbose
        
    def fit(self, X):
        if self.verbose:
            print(f'Finding all kmer permutations for k={self.k}')
        self.kmer_perms = self.__create_kmer_permutations(self.k)
        
        return self
        
    def transform(self, X):
        tqdm.pandas()
        seqs = X['seq']
        if self.verbose:
            X_trans = pd.DataFrame(seqs.progress_apply(lambda x: self.__kmer_transform_seq(x, self.k, self.kmer_perms)).tolist(), index=seqs.index)
        else:
            X_trans = pd.DataFrame(seqs.apply(lambda x: self.__kmer_transform_seq(x, self.k, self.kmer_perms)).tolist(), index=seqs.index)

        return X_trans
    
    def fit_transform(self, X, y=None):
        
        return self.fit(X).transform(X)
    
    def __create_kmer_permutations(self, k):
        
        superstring = 'G' * k + 'A' * k + 'T' * k + 'C' * k
        kmer_perms = sorted([''.join(kmer) for kmer in set(permutations(superstring, k))])
        
        return kmer_perms
    
    def __kmer_transform_seq(self, seq, k, kmer_perms):
        
        kmer_freq = dict.fromkeys(kmer_perms, 0)
        
        for i in range(len(seq) - k):
            if self.normalize:
                kmer_freq[seq[i:i+k]] += 1/len(seq)
            else:
                kmer_freq[seq[i:i+k]] += 1

        return kmer_freq

class KGroupKMerTransformer(TransformerMixin, BaseEstimator):
    """
    Sklearn Pipeline Transformer to create k-sized groups of k-mer frequency columns.
    
    Parameters:
        k (int) : length of characters in each substring of the input DNA sequence.
        normalize (bool): True will scale each frequency by the total length of the input DNA sequence.
    
    """
    
    def __init__(self, k, normalize=False, verbose=True):
        self.k = k
        self.normalize = normalize
        self.verbose = verbose
        
    def fit(self, X):
        if self.verbose:
            print(f'Finding all k-grouped k-mer permutations for k={self.k}')
        self.kmer_perms = self.__create_kgroup_kmer_permutations(self.k)
        if self.verbose:
            print(f'Total columns: {len(self.kmer_perms)}')
        return self
        
    def transform(self, X):
        tqdm.pandas()
        seqs = X['seq']
        if self.verbose:
            X_trans = pd.DataFrame(seqs.progress_apply(lambda x: self.__kmer_transform_seq(x, self.k, self.kmer_perms)).tolist(), index=seqs.index)
        else:
            X_trans = pd.DataFrame(seqs.apply(lambda x: self.__kmer_transform_seq(x, self.k, self.kmer_perms)).tolist(), index=seqs.index)

        return X_trans
    
    def fit_transform(self, X, y=None):
        
        return self.fit(X).transform(X)
    
    def __create_kgroup_kmer_permutations(self, k):
        
        kmer_perms =[]
        for a in range(k+1):
            for c in range(k+1):
                for g in range(k+1):
                    for t in range(k+1):
                        if a + c + g + t == k:
                            kmer_perms.append(self.__make_key_string(a,c,t,g))
        
        return kmer_perms
    
    def __kmer_transform_seq(self, seq, k, kmer_perms):
        
        kmer_freq = dict.fromkeys(kmer_perms, 0)
        
        for i in range(len(seq) - k):
            subseq = seq[i:i+k]
            a = subseq.count('A')
            c = subseq.count('C')
            g = subseq.count('G')
            t = subseq.count('T')
            key_string = self.__make_key_string(a,c,t,g)
            if self.normalize:
                kmer_freq[key_string] += 1/len(seq)
            else:
                kmer_freq[key_string] += 1

        return kmer_freq
    
    def __make_key_string(self, a,c,t,g):
        return (a,c,t,g)
    
class KGroupColumnPruner(TransformerMixin, BaseEstimator):
    
    def __init__(self, verbose=True):
        
        self.verbose = verbose
       
    def fit(self, X):
        
        self.columns_to_keep = X.loc[:,(X>0).any(axis=0)].columns
        
        if self.verbose:
            print(f"""
            Original Column Length: {len(X.columns)}
            Pruned Column Length: {len(self.columns_to_keep)}
            """)
        
        return self
        
    def transform(self, X):
        
        X_trans = X[self.columns_to_keep]
        
        return X_trans
    
    def fit_transform(self, X, y=None):
        
        return self.fit(X).transform(X)