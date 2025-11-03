# hangman_env.py
import random
from utils import mask_word, LETTER_TO_IDX, IDX_TO_LETTER

class HangmanEnv:
    def __init__(self, words, max_wrong=6, seed=None):
        self.words = [w.strip().lower() for w in words if w.strip()]
        self.max_wrong = max_wrong
        self.word = None
        self.guessed = set()
        self.wrong = 0
        if seed is not None:
            random.seed(seed)

    def reset(self, word=None, partial_reveal=True):
        self.word = (word or random.choice(self.words)).lower()
        self.done = False
        self.wrong = 0

        # reveal some random letters initially
        if partial_reveal:
            num_reveal = max(1, int(len(self.word) * random.uniform(0.2, 0.5)))  # reveal 20–50%
            revealed_indices = random.sample(range(len(self.word)), num_reveal)
            self.guessed = {self.word[i] for i in revealed_indices}
        else:
            self.guessed = set()

        self.revealed = mask_word(self.word, self.guessed)
        return self._get_obs()


    def _get_obs(self):
        return {
            'word': self.word,
            'masked': mask_word(self.word, self.guessed),
            'guessed': set(self.guessed),
            'wrong': self.wrong,
            'allowed_wrong': self.max_wrong
        }

    def step(self, letter):
        """
        letter: 'a'..'z'
        returns: obs, reward, done, info
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        info = {}
        reward = 0.0
        if letter in self.guessed:
            # repeated guess
            reward = -2.0
        else:
            self.guessed.add(letter)
            if letter in self.word:
                # reveal all occurrences
                count = self.word.count(letter)
                reward = 5.0 * count
            else:
                self.wrong += 1
                reward = -3.0

        masked = mask_word(self.word, self.guessed)
        self.revealed = masked
        if '_' not in masked:
            self.done = True
            reward += 100.0  # solved bonus
        elif self.wrong >= self.max_wrong:
            self.done = True
            reward -= 10.0  # extra penalty for losing

        return self._get_obs(), reward, self.done, info
