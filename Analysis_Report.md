======================================================================
                          ANALYSIS REPORT
======================================================================

PROJECT TITLE:
----------------------------------------------------------------------
HANGMAN WORD PREDICTOR USING REINFORCEMENT LEARNING AND HIDDEN MARKOV MODEL

======================================================================
1. ABSTRACT
----------------------------------------------------------------------
This project focuses on building an intelligent Hangman word predictor
using a combination of Reinforcement Learning (RL) and Hidden Markov
Model (HMM). The HMM learns the statistical distribution of letters
from a corpus, while the RL agent learns through trial and error to
make better predictions. Together, they enable an AI to play Hangman
with logical guessing instead of random attempts.

======================================================================
2. PROBLEM STATEMENT
----------------------------------------------------------------------
The traditional Hangman game involves guessing letters to reveal a
hidden word. Humans use intuition and word familiarity, but machines
need to learn the patterns behind language. The problem is to train a
model that can intelligently predict letters in partially hidden words
using probabilities and learning from its own decisions.

======================================================================
3. OBJECTIVES
----------------------------------------------------------------------
| • Create a Hangman simulation environment.                         |
| • Implement an HMM model to learn letter probabilities.             |
| • Build a Reinforcement Learning (DQN) agent to learn optimal moves.|
| • Integrate HMM and RL for intelligent hybrid predictions.          |
| • Analyze performance and evaluate success rate.                    |

======================================================================
4. SYSTEM ARCHITECTURE
----------------------------------------------------------------------
| MODULES:                                                            |
| ------------------------------------------------------------------- |
| 1. Hangman Environment - simulates the game rounds.                 |
| 2. HMM Oracle - predicts letters using probabilistic patterns.      |
| 3. DQN Agent - learns through rewards and penalties.                |
| 4. Training Loop - connects environment, agent, and HMM together.   |

======================================================================
5. IMPLEMENTATION DETAILS
----------------------------------------------------------------------
| 5.1 Hidden Markov Model (HMM):                                      |
| ------------------------------------------------------------------- |
| • Trained on English corpus using unigram, bigram, and positional   |
|   letter statistics.                                                |
| • Calculates letter probability as:                                 |
|   P(letter) = 0.5*Ppositional + 0.35*Punigram + 0.15*Pbigram        |
| • Predicts which letter fits best in missing positions.             |

| 5.2 Reinforcement Learning (DQN Agent):                             |
| ------------------------------------------------------------------- |
| • Neural network outputs Q-values for all 26 letters.               |
| • Learns by maximizing reward using Bellman equation.               |
| • State = Masked word + Guessed letters + Remaining lives + HMM data|
| • Uses epsilon-greedy exploration and replay buffer.                |

| 5.3 Reward Policy:                                                  |
| ------------------------------------------------------------------- |
| Correct guess         : +5                                          |
| Wrong guess           : -3                                          |
| Solving full word     : +100                                        |
| Repeated guess        : -2                                          |
| Losing the game       : -10                                         |

======================================================================
6. ALGORITHM / FLOW
----------------------------------------------------------------------
1. Load corpus and train HMM Oracle.
2. Initialize Hangman Environment and DQN Agent.
3. Randomly choose a word with partial letters revealed.
4. Agent observes masked word and guesses a letter.
5. Environment returns reward (+/-) and updates state.
6. Store experience in replay buffer.
7. Update Q-network using mini-batch gradient descent.
8. Repeat for 40,000+ episodes.
9. Evaluate trained model on test words.

======================================================================
7. RESULTS AND ANALYSIS
----------------------------------------------------------------------
| Metric                  | Result                                   |
| ------------------------ | ---------------------------------------- |
| Training Episodes        | 40,000                                   |
| Initial Avg Reward       | -15.6                                    |
| Final Avg Reward         | +13.8                                    |
| Evaluation Games         | 2000                                     |
| Wins Achieved            | 145                                      |
| Success Rate             | 7.25%                                    |
| Wrong Guesses            | 11,778                                   |

OBSERVATIONS:
• The agent gradually improves prediction accuracy.
• The HMM model provides prior probabilities, stabilizing learning.
• RL fine-tunes predictions by rewarding correct sequences.
• Success rate grows as episodes increase.

======================================================================
8. CONCLUSION
----------------------------------------------------------------------
The system successfully integrates Reinforcement Learning with
probabilistic modeling to simulate intelligent Hangman playing.
The RL agent learns decision-making, while HMM provides letter
probabilities. Over time, the AI develops intuition-like behavior,
balancing exploration and exploitation to maximize accuracy.

======================================================================
9. FUTURE ENHANCEMENTS
----------------------------------------------------------------------
| • Use Trigram or Transformer-based models for richer context.       |
| • Add GUI for interactive play.                                     |
| • Include semantic embeddings (word meaning awareness).             |
| • Improve reward function for longer words.                         |

======================================================================
10. REFERENCES
----------------------------------------------------------------------
| 1. Sutton, R.S. & Barto, A.G. – Reinforcement Learning: An Intro.  |
| 2. Rabiner, L. – Tutorial on Hidden Markov Models.                 |
| 3. PyTorch Official Documentation.                                 |
| 4. English Word Corpus Dataset.                                   |

======================================================================
11. PROJECT SUMMARY
----------------------------------------------------------------------
| Component          | Description                                   |
| ------------------ | --------------------------------------------- |
| Programming Lang.  | Python                                         |
| Frameworks Used    | PyTorch, NumPy                                 |
| Algorithms Used    | HMM (N-gram) + DQN (RL)                        |
| Input Data         | English Word Corpus                            |
| Output             | Predicted Missing Letters                      |
| Type               | Hybrid Intelligent System                      |
| Result             | AI learns to guess words intelligently.        |

======================================================================
                        END OF REPORT
======================================================================
