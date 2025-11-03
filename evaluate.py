import os
import pickle
import random
import numpy as np
from utils import ALPHABET, mask_word
from hangman_env import HangmanEnv
from dqn_agent import DQNAgent
from hmm_model import HMMOracle
from train import load_corpus, build_state_vector


def random_partial_mask(word, min_ratio=0.2, max_ratio=0.5):
    """
    Randomly hides 50–80% of the letters in a word with underscores.
    Returns (masked_word, revealed_letters_set)
    """
    L = len(word)
    num_reveal = max(1, int(L * random.uniform(min_ratio, max_ratio)))
    reveal_indices = random.sample(range(L), num_reveal)
    revealed_letters = {word[i] for i in reveal_indices}
    masked_word = "".join(ch if ch in revealed_letters else "_" for ch in word)
    return masked_word, revealed_letters


def evaluate(
    model_path="models/dqn_agent.pth",
    hmm_path="models/hmm_oracle.pkl",
    test_words_path="Data/test.txt",
    n_games=2000
):
    # ==== Step 1: Safety checks ====
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    if not os.path.exists(hmm_path):
        raise FileNotFoundError(f"❌ HMM file not found: {hmm_path}")
    if not os.path.exists(test_words_path):
        raise FileNotFoundError(f"❌ Test words file not found: {test_words_path}")

    # ==== Step 2: Load HMM Oracle ====
    with open(hmm_path, "rb") as f:
        hmm = pickle.load(f)

    # ==== Step 3: Create dummy state to rebuild agent ====
    dummy_mask = "_" * 6
    dummy_state = build_state_vector(dummy_mask, set(), 6, hmm)
    agent = DQNAgent(input_shape=dummy_state.shape)
    agent.load(model_path)

    # disable exploration (greedy)
    agent.epsilon = 0.0
    agent.q_net.eval()

    # ==== Step 4: Load test words ====
    with open(test_words_path, "r", encoding="utf-8") as f:
        test_words = [w.strip().lower() for w in f if w.strip() and w.strip().isalpha()]

    if len(test_words) == 0:
        raise ValueError("❌ No valid test words found in Data/test.txt.")
    if len(test_words) < n_games:
        n_games = len(test_words)
    else:
        test_words = test_words[:n_games]

    env = HangmanEnv(test_words, max_wrong=6)

    wins = 0
    wrong_guesses_total = 0
    repeated_total = 0

    print(f"\n🧠 Starting Evaluation on {n_games} partially masked games...\n")

    # ==== Step 5: Run evaluation ====
    for i, w in enumerate(test_words, start=1):
        # random partial masking
        masked_word, revealed_letters = random_partial_mask(w)

        # initialize environment manually with revealed letters
        obs = env.reset(word=w, partial_reveal=False)
        env.guessed = revealed_letters.copy()  # manually set revealed letters
        obs = env._get_obs()  # refresh observation

        done = False
        while not done:
            masked = obs["masked"]
            guessed = obs["guessed"]
            lives_left = env.max_wrong - obs["wrong"]

            state_vec = build_state_vector(masked, guessed, lives_left, hmm)
            mask = np.array([(ch not in guessed) for ch in ALPHABET], dtype=bool)

            # predict best letter
            aidx = agent.act(state_vec, mask)
            letter = ALPHABET[aidx]

            if letter in guessed:
                repeated_total += 1

            prev_wrong = obs["wrong"]
            obs, reward, done, _ = env.step(letter)
            if obs["wrong"] > prev_wrong:
                wrong_guesses_total += 1

        if "_" not in obs["masked"]:
            wins += 1

        if i % 100 == 0 or i == n_games:
            print(f"Progress: {i}/{n_games} | Current Wins: {wins}")

    # ==== Step 6: Metrics ====
    success_rate = wins / n_games
    final_score = (success_rate * 2000) - (wrong_guesses_total * 5) - (repeated_total * 2)

    print("\n===============================")
    print("📊  Evaluation Results")
    print("===============================")
    print(f"Total Games           : {n_games}")
    print(f"Total Wins            : {wins}")
    print(f"Success Rate          : {success_rate:.4f}")
    print(f"Total Wrong Guesses   : {wrong_guesses_total}")
    print(f"Total Repeated Guesses: {repeated_total}")
    print(f"🏁 Final Score         : {final_score:.2f}")
    print("===============================\n")

    return {
        "games": n_games,
        "wins": wins,
        "success_rate": success_rate,
        "wrong_guesses": wrong_guesses_total,
        "repeated_guesses": repeated_total,
        "final_score": final_score,
    }


if __name__ == "__main__":
    evaluate()
