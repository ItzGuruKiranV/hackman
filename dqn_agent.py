# dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from utils import ALPHABET, LETTER_TO_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden=256, output_dim=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d,m = zip(*batch)
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(ns), np.array(d, dtype=np.float32), np.stack(m)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_shape, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.05, eps_decay=50000):
        input_dim = 1
        for d in input_shape:
            input_dim *= d
        self.q_net = QNetwork(input_dim, hidden=256).to(DEVICE)
        self.target_net = QNetwork(input_dim, hidden=256).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(200000)
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.eps_decay = eps_decay
        self.steps_done = 0

    def act(self, state, mask):
        """
        state: np array input for the network shape=input_shape
        mask: boolean array shape (26,) True if allowed
        returns index (0..25)
        """
        self.steps_done += 1
        # epsilon decay
        self.epsilon = max(self.epsilon_final, self.epsilon - (1.0 - self.epsilon_final) / self.eps_decay)
        if random.random() < self.epsilon:
            choices = [i for i in range(26) if mask[i]]
            return random.choice(choices)
        else:
            self.q_net.eval()
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                qvals = self.q_net(x).cpu().numpy().squeeze()
                # mask out invalid
                qvals_masked = qvals.copy()
                qvals_masked[~mask] = -1e9
                return int(qvals_masked.argmax())

    def push(self, *args):
        self.replay.push(*args)

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return 0.0
        s, a, r, ns, d, m = self.replay.sample(batch_size)
        s_t = torch.tensor(s, dtype=torch.float32).to(DEVICE)
        ns_t = torch.tensor(ns, dtype=torch.float32).to(DEVICE)
        a_t = torch.tensor(a, dtype=torch.int64).to(DEVICE)
        r_t = torch.tensor(r, dtype=torch.float32).to(DEVICE)
        d_t = torch.tensor(d, dtype=torch.float32).to(DEVICE)

        q_vals = self.q_net(s_t)
        q_val = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(ns_t)
            # mask invalid next actions using m (mask for next_state, shape batch x 26)
            mask_next = torch.tensor(m, dtype=torch.bool).to(DEVICE)
            next_q[~mask_next] = -1e9
            max_next_q = next_q.max(dim=1)[0]
            target = r_t + (1 - d_t) * self.gamma * max_next_q

        loss = nn.functional.mse_loss(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.sync_target()
