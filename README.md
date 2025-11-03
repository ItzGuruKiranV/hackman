# Hackman - Hangman ML agent

## Requirements
- Python 3.8+
- pip install -r requirements.txt
  (torch, numpy, tqdm, pandas)

## Files
- corpus.txt: provided word list (50k)
- train.py: trains HMM and DQN
- evaluate.py: evaluate saved model on test_words.txt
- models/: saved artifacts after training

## To run:
1. Place corpus.txt in project root.
2. (Optional) prepare test_words.txt with evaluation words.
3. Train: python train.py
4. Evaluate: python evaluate.py
