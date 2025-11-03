# hmm_model.py
import numpy as np
from collections import defaultdict, Counter
from utils import ALPHABET, LETTER_TO_IDX

class HMMOracle:
    """
    Simple character-level n-gram oracle:
    - collects unigram and bigram counts and position-wise letter counts.
    - can compute probability distribution for letters for each blank position
      given the masked word and guessed letters.
    """
    def __init__(self, ngram=2, smoothing=1e-3):
        self.ngram = ngram
        self.unigram = Counter()
        self.bigram = Counter()
        self.position_counts = defaultdict(Counter)  # position -> letter counts
        self.total_unigrams = 0
        self.smoothing = smoothing
        self.max_word_len = 0
    
    def train_on_corpus(self, words):
        for w in words:
            word = w.strip().lower()
            if not word: 
                continue
            self.max_word_len = max(self.max_word_len, len(word))
            for i, ch in enumerate(word):
                if ch.isalpha():
                    self.unigram[ch] += 1
                    self.position_counts[i][ch] += 1
                    self.total_unigrams += 1
                    if i > 0:
                        pair = (word[i-1], ch)
                        self.bigram[pair] += 1

    def letter_position_probs(self, masked, guessed=set()):
        """
        For each position that is blank, compute probability distribution over letters.
        We'll combine:
          - position frequency P(letter | position)
          - unigram P(letter)
          - bigram P(letter | left_char) if left char known
        Combine via simple weighted sum and normalize.
        Returns: list of length len(masked), each item is np.array(26,)
        """
        L = len(masked)
        out = []
        for i in range(L):
            if masked[i] != '_':
                # fixed letter: probability 1 on that letter
                vec = np.zeros(26, dtype=np.float32)
                ch = masked[i]
                vec[LETTER_TO_IDX[ch]] = 1.0
                out.append(vec)
                continue

            # compute scores for each letter
            scores = np.zeros(26, dtype=np.float64)
            for j, letter in enumerate(ALPHABET):
                if letter in guessed:
                    # Encourage zero prob for already guessed letters (or small smoothing)
                    base = 1e-8
                else:
                    # components
                    # p_pos
                    pos_count = self.position_counts[i][letter]
                    pos_total = sum(self.position_counts[i].values()) + 1e-9
                    p_pos = pos_count / pos_total

                    # p_uni
                    p_uni = (self.unigram[letter] + 1e-9) / (self.total_unigrams + 1e-9)

                    # p_left_bigram
                    p_big = 0.0
                    if i > 0 and masked[i-1] != '_':
                        left = masked[i-1]
                        pair_count = self.bigram[(left, letter)]
                        left_count_total = sum(v for (a,b), v in self.bigram.items() if a == left) + 1e-9
                        p_big = pair_count / left_count_total

                    # weights - heuristics
                    scores[j] = 0.5 * p_pos + 0.35 * p_uni + 0.15 * p_big + self.smoothing
            # normalize
            scores = scores / (scores.sum() + 1e-12)
            out.append(scores.astype(np.float32))
        return out

    def letter_probs_for_mask(self, masked, guessed=set()):
        """Return aggregated letter probability vector (26,) for the whole masked word.
           We aggregate by summing position-wise probabilities for blank positions.
        """
        p_list = self.letter_position_probs(masked, guessed)
        agg = np.zeros(26, dtype=np.float32)
        for i, p in enumerate(p_list):
            if masked[i] == '_':
                agg += p
        # normalize
        if agg.sum() == 0:
            agg += 1e-8
        agg = agg / agg.sum()
        return agg



#P(letter)=0.7×Ppositional​+0.2×Punigram​+0.1×Pbigram​
