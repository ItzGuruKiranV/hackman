import pickle
import random
from utils import ALPHABET
from hmm_model import HMMOracle

def mask_one_letter(word):
    """Hide exactly one random letter in the given word."""
    if len(word) < 2:
        return word, set(), None
    hide_index = random.randint(0, len(word) - 1)
    masked = ''.join('_' if i == hide_index else word[i] for i in range(len(word)))
    guessed = set(word[i] for i in range(len(word)) if i != hide_index)
    return masked, guessed, hide_index

def predict_missing_letter(hmm, masked, guessed, hide_index):
    """
    Predict the most probable letter for the single hidden position.
    Uses hmm.letter_position_probs() to get per-position probability distributions.
    """
    p_list = hmm.letter_position_probs(masked, guessed)
    probs = p_list[hide_index]
    ranked = sorted([(ALPHABET[j], float(probs[j])) for j in range(26)],
                    key=lambda x: x[1], reverse=True)
    top_letter, top_prob = ranked[0]
    return top_letter, top_prob, ranked[:5]  # also return top 5 for display

def show_predictions(hmm_path="models/hmm_oracle.pkl", test_path="Data/test.txt", num_words=10):
    # Load trained HMM model
    with open(hmm_path, "rb") as f:
        hmm = pickle.load(f)

    # Load test words
    with open(test_path, "r", encoding="utf-8") as f:
        words = [w.strip().lower() for w in f if w.strip().isalpha()]

    print("\n📘 HMM Prediction for Single Missing Letter\n")
    for word in words[:num_words]:
        masked, guessed, hide_index = mask_one_letter(word)
        if hide_index is None:
            continue

        top_letter, top_prob, top5 = predict_missing_letter(hmm, masked, guessed, hide_index)

        print("===============================================")
        print(f"Original Word     : {word}")
        print(f"Masked Word       : {masked}")
        print(f"Known Letters     : {sorted(guessed)}")
        print(f"Missing Position  : {hide_index + 1} (1-based index)")
        print(f"🔹 Predicted Letter : '{top_letter}' (Probability = {top_prob:.4f})")

        # Show top 5 ranked letters for clarity
        print("\nTop 5 Candidate Letters:")
        for rank, (letter, prob) in enumerate(top5, start=1):
            print(f"  {rank}. {letter} — {prob:.4f}")

        # Show the completed prediction
        filled = list(masked)
        filled[hide_index] = top_letter
        print(f"\nPredicted Complete Word: {''.join(filled)}")
        print("===============================================\n")

if __name__ == "__main__":
    # Run the predictor on a few test words
    show_predictions(hmm_path="models/hmm_oracle.pkl", test_path="Data/test.txt", num_words=10)
