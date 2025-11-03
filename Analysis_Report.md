# Analysis Report — Hackman

## 1. Summary
(short summary of approach)

## 2. HMM Design Choices
- Type: character-level n-gram with position conditioning.
- Training: counts of unigrams, bigrams, and position-wise letter counts.
- Rationale: fast, simple, generalizes over words of different lengths.

## 3. RL Agent Design
- State: masked word (one-hot positions, padded to MAX_WORD_LEN), guessed letters binary vector, remaining lives, HMM probability vector.
- Action: choose letter 0..25.
- Reward: +5 per revealed letter, -3 on wrong guess, -2 for repeated guess, +100 solve bonus.
- Algorithm: DQN, experience replay, target network.

## 4. Training Setup
- Episodes: ...
- Hyperparameters: ...

## 5. Results
- Learning curves (insert training reward / loss plots)
- Final Evaluation metrics:
  - Success Rate: ...
  - Avg Wrong Guesses: ...
  - Avg Repeated Guesses: ...
  - Final Score: ...

## 6. Key Observations & Challenges
- HMM limitations
- State space size
- Exploration vs exploitation tuning
- Sample efficiency

## 7. Future work
- Use more powerful sequence model (char-LSTM HMM or CRF)
- Curriculum learning (train on short words first)
- Better reward shaping
