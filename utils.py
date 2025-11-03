# utils.py
import string
import random
import numpy as np

ALPHABET = list(string.ascii_lowercase)
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_LETTER = {i: c for c, i in LETTER_TO_IDX.items()}

def mask_word(word, guessed):
    return ''.join([ch if ch in guessed else '_' for ch in word])

def random_word_from_list(words):
    return random.choice(words).strip().lower()

def one_hot_letter(letter):
    vec = np.zeros(26, dtype=np.float32)
    if letter in LETTER_TO_IDX:
        vec[LETTER_TO_IDX[letter]] = 1.0
    return vec

def letters_to_binary_vector(guessed):
    vec = np.zeros(26, dtype=np.float32)
    for ch in guessed:
        if ch in LETTER_TO_IDX:
            vec[LETTER_TO_IDX[ch]] = 1.0
    return vec

def word_to_onehot_positions(word, max_len):
    # returns shape (max_len, 27) - 26 letters + blank
    blanks = np.zeros((max_len, 27), dtype=np.float32)
    for i in range(max_len):
        if i >= len(word):
            blanks[i, 26] = 1.0
        else:
            ch = word[i]
            blanks[i, LETTER_TO_IDX[ch]] = 1.0
    return blanks

def masked_to_onehot_positions(masked, max_len):
    # '_' as blank token index 26
    arr = np.zeros((max_len, 27), dtype=np.float32)
    for i in range(max_len):
        if i >= len(masked):
            arr[i, 26] = 1.0
        else:
            ch = masked[i]
            if ch == '_':
                arr[i, 26] = 1.0
            else:
                arr[i, LETTER_TO_IDX[ch]] = 1.0
    return arr
