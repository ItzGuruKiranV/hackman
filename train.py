# train.py
import os
import random
import numpy as np
from tqdm import trange, tqdm

from utils import ALPHABET, LETTER_TO_IDX, masked_to_onehot_positions, letters_to_binary_vector
from hmm_model import HMMOracle
from hangman_env import HangmanEnv
from dqn_agent import DQNAgent

MAX_WORD_LEN = 15  # pad / truncate to this many positions

def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = [w.strip().lower() for w in f if w.strip() and w.strip().isalpha()]
    return words

def build_state_vector(masked, guessed, lives, hmm_oracle):
    # masked_to_onehot_positions -> shape (MAX_WORD_LEN, 27)
    pos_oh = masked_to_onehot_positions(masked, MAX_WORD_LEN)  # float32
    guessed_vec = letters_to_binary_vector(guessed)  # 26
    hh = hmm_oracle.letter_probs_for_mask(masked, guessed)  # 26
    # flatten and concat: pos_oh flatten + guessed + lives + hh
    flat = pos_oh.flatten()
    vec = np.concatenate([flat, guessed_vec, np.array([lives/6.0], dtype=np.float32), hh])
    return vec.astype(np.float32)

def train():
    corpus = load_corpus("Data/corpus.txt")

    random.shuffle(corpus)
    # split
    split = int(0.8 * len(corpus))
    train_words = corpus[:split]
    val_words = corpus[split:]

    # train HMM oracle on full train set
    hmm = HMMOracle()
    hmm.train_on_corpus(train_words)

    # env and agent
    env = HangmanEnv(train_words, max_wrong=6)
    # compute input dim
    dummy_mask = '_' * 6
    dummy_state = build_state_vector(dummy_mask, set(), 6, hmm)
    input_shape = dummy_state.shape
    agent = DQNAgent(input_shape=input_shape, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_final=0.05, eps_decay=100000)

    n_episodes = 40000
    target_sync_every = 1000
    batch_size = 128
    losses = []
    rewards = []

    for ep in trange(n_episodes, desc="Episodes"):
        # sample a random word length to encourage diversity
        word = random.choice(train_words)
        obs = env.reset(word=word, partial_reveal=True)

        total_reward = 0.0
        done = False
        while not done:
            masked = obs['masked']
            guessed = obs['guessed']
            lives_left = env.max_wrong - obs['wrong']
            state_vec = build_state_vector(masked, guessed, lives_left, hmm)
            # prepare mask
            mask = np.array([ (ch not in guessed) for ch in ALPHABET ], dtype=bool)
            act_idx = agent.act(state_vec, mask)
            act_letter = ALPHABET[act_idx]
            next_obs, reward, done, _ = env.step(act_letter)
            next_masked = next_obs['masked']
            next_guessed = next_obs['guessed']
            next_lives = env.max_wrong - next_obs['wrong']
            next_state_vec = build_state_vector(next_masked, next_guessed, next_lives, hmm)
            next_mask = np.array([ (ch not in next_guessed) for ch in ALPHABET ], dtype=bool)

            agent.push(state_vec, act_idx, reward, next_state_vec, done, next_mask)
            loss = agent.update(batch_size=batch_size)
            if loss:
                losses.append(loss)
            total_reward += reward
            obs = next_obs

        rewards.append(total_reward)
        if (ep+1) % target_sync_every == 0:
            agent.sync_target()
        if (ep+1) % 2000 == 0:
            print(f"Episode {ep+1} avg reward (last 2000): {np.mean(rewards[-2000:])}")

    # final save
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_agent.pth")
    # save the HMM counts (pickle)
    import pickle
    with open("models/hmm_oracle.pkl", "wb") as f:
        pickle.dump(hmm, f)

    # Save simple logs
    np.save("models/losses.npy", np.array(losses))
    np.save("models/rewards.npy", np.array(rewards))
    print("Training complete. Models saved to /models.")

if __name__ == "__main__":
    train()
