# Class 3: Training a Ms. Pac-Man DQN Agent

This repository is my Class 3 submission: a Deep Q-Network (DQN) trained on `ALE/MsPacman-v5` using the
class-provided [`pacman_dqn.ipynb`](pacman_dqn.ipynb) notebook. The executed notebook (with all outputs
visible) is committed as-is, and the evidence it produced is copied into [`results/`](results/) so it can
be inspected without rerunning anything.

## How to open and run it

1. Clone this repository.
2. Local Jupyter/VS Code: create a Python 3.11–3.13 environment, `pip install -r requirements.txt`, open
   `pacman_dqn.ipynb`, and select that kernel.
3. Or use Colab: [Open in Google Colab](https://colab.research.google.com/github/pepealonso95/pacman-dqn/blob/main/pacman_dqn.ipynb)
   (select a GPU under Runtime → Change runtime type if available).
4. Edit the three values in section 1 (already set below for this run) and choose **Run All**.

## My three hyperparameters

| Setting | Value | Why I chose it |
|---|---|---|
| Exploration | **0.05** | I wanted the agent to mostly follow its learned policy after warm-up rather than move randomly, since this was a short run and I wanted whatever signal existed in 5 episodes to be visible instead of drowned out by random moves. This also matches the notebook's fixed evaluation exploration (5%), so training and evaluation behave consistently. |
| Episodes | **5** | The assignment explicitly frames 5 episodes as a setup check, not a promise of useful play. I used it to verify the whole pipeline (environment, MPS device, saving, evaluation, GIFs) end-to-end before considering a longer run, and to report the results honestly as a short run rather than overstating them. |
| Learning rate | **0.0001** | This is the assignment's suggested reference value and a standard, stable Adam learning rate for DQN. With only 5 episodes there was no reason to risk instability from a higher rate. |

All other hyperparameters (replay size, batch size, warm-up steps, target-network sync, gamma, frame skip,
evaluation seeds/exploration, etc.) were left at the notebook's fixed classroom defaults — see
[`results/config.json`](results/config.json) for the exact values used.

## What I expected vs. what I observed

**Expected:** With only 5 training episodes (536 learning updates total), I did not expect the network to
learn a meaningfully better Ms. Pac-Man policy. I expected training scores to be noisy and expected the
before/after evaluation means to be close to each other, with any difference dominated by randomness
(sticky actions, no-op resets) rather than real skill improvement.

**Observed:** That is what happened. The five before/after evaluation scores moved in both directions
(two trained scores over 1000, but also two trained scores at 220, both lower than every baseline score).
The mean score went from 492.0 (untrained) to 550.0 (trained) — a small increase that is well within the
run-to-run noise you'd expect from 5 evaluation games, not evidence of a converged, capable agent. The
training-loss curve (see the dashboard below) is still rising over the 5 episodes, which is expected: the
target network and Q-values are still moving quickly right after warm-up, before loss has had a chance to
settle down. This run was a successful **setup check** — the pipeline works correctly end-to-end — not a
demonstration of learned Pac-Man skill.

## Actual run stats (from `results/training_summary.json` and `results/config.json`)

- **Completed episodes:** 5 / 5 requested (not interrupted)
- **Total decisions:** 3,140
- **Learning updates:** 536
- **Elapsed time:** ~7.0 seconds (`6.96` seconds including periodic-demo overhead)
- **Hardware:** Apple Silicon (Apple M5), device `mps` (PyTorch's Metal backend), macOS 26.6.2 arm64
- **Software:** Python 3.13.15, PyTorch 2.14.0, Gymnasium 1.3.0, ale-py 0.11.2, NumPy 2.5.3

Because this run only reached 5 completed episodes (under the 25-episode threshold), the notebook did not
produce intermediate checkpoints or progress GIFs — only the untrained baseline and final-best gameplay
samples, both included below. No episodes were interrupted early; every learning update shown is real
(the run did pass the 1,000-decision warm-up, so `mean_loss` is populated from episode 2 onward).

## What the agent observes, does, and is rewarded for

- **Observations:** The agent does not see game "state" directly — it sees pixels. Each of its screens
  is a stack of the last 4 game frames, converted to grayscale and resized to 84×84 pixels. Stacking 4
  frames lets the network infer motion (e.g., which way a ghost is moving) from a single snapshot.
- **Actions:** The agent chooses one of 9 discrete joystick actions (no-op plus the 8 directions Ms.
  Pac-Man's joystick supports). One chosen action is held for 4 emulator frames ("frame skip") before the
  agent decides again.
- **Rewards:** The reward at each step is the change in the game's own score — eating a dot, a power
  pellet, a fruit, or a frightened ghost all add points; losing is not directly penalized beyond the game
  ending. During training only, rewards are clipped to [-1, 1] to keep learning stable; all scores reported
  here (training curve and evaluation) are the raw, unclipped game score.
- **Learning signal:** A convolutional Q-network predicts a value for each of the 9 actions given the
  current 4-frame stack. It's trained with experience replay (sampling past transitions instead of only
  the most recent one) and a separate, periodically-synced target network, using a Huber loss against a
  discounted (`gamma = 0.99`) reward target — standard DQN (Mnih et al., 2015).

## One limitation and one next experiment

**Limitation:** 5 episodes (536 learning updates, one replay buffer never even close to full at 5,000
capacity) is nowhere near enough experience for a from-scratch DQN to learn a meaningfully better Ms.
Pac-Man policy. The before/after score difference here is statistical noise, not learned skill — this is
visible directly in the evaluation table below, where the trained agent's scores are actually more
variable (220 to 1060) than the untrained baseline's (320 to 800).

**Next experiment:** The single setting I'd change next is **episodes**, raising it from 5 to at least 100
(the notebook's own starting point), keeping exploration and learning rate fixed. That would cross the
25-episode threshold that triggers periodic checkpoints/GIFs every 25 games, letting me watch the policy
change over time instead of only comparing two endpoints, and it would give the replay buffer and target
network enough updates to plausibly show a real (not noise-level) improvement in mean evaluation score.

## Evidence

### Training dashboard (score, loss, exploration)

![Training dashboard](results/training_dashboard.png)

### Gameplay: untrained vs. best trained (first 20 seconds, 4× speed)

| Untrained (before training) | Best trained (after training) |
|---|---|
| ![Untrained gameplay](results/untrained.gif) | ![Best trained gameplay](results/trained_best.gif) |

This was a 5-episode run, which is below the notebook's 25-episode threshold for intermediate progress
GIFs and checkpoints, so no intermediate samples were produced.

### All five before/after evaluation scores

Same 5 seeds, same 5% evaluation exploration, same step cap, before and after training. Baseline is an
**untrained network**, not a random-action agent. Full data: [`results/comparison.json`](results/comparison.json).

| Seed | Untrained score | Trained score |
|---|---|---|
| 101 | 350.0 | 230.0 |
| 202 | 500.0 | 1020.0 |
| 303 | 320.0 | 220.0 |
| 404 | 800.0 | 1060.0 |
| 505 | 490.0 | 220.0 |
| **Mean** | **492.0** | **550.0** |

### Files

- Executed notebook with outputs: [`pacman_dqn.ipynb`](pacman_dqn.ipynb)
- Run config (hyperparameters, hardware, package versions): [`results/config.json`](results/config.json)
- Per-episode training log: [`results/training.csv`](results/training.csv)
- Training run totals: [`results/training_summary.json`](results/training_summary.json)
- Before/after evaluation scores: [`results/comparison.json`](results/comparison.json)
- Untrained baseline evaluation only: [`results/baseline.json`](results/baseline.json)

Model checkpoints (`untrained.pt`, `trained.pt`, ~6.5 MB each) and the full run folder are kept locally
under `pacman_runs/20260915_225840_432331/` (git-ignored per the notebook's default `.gitignore`) rather
than committed to this repository, per the assignment's guidance to keep large checkpoints out of the repo.

## Credits

Notebook and DQN implementation from the class-provided [pacman-dqn](https://github.com/pepealonso95/pacman-dqn)
starter repository.
