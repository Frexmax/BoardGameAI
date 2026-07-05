# BoardGameAI

BoardGameAI trains reinforcement learning agents to play three board games of increasing difficulty:

1. Tic-Tac-Toe
2. Connect 4
3. Checkers

Each agent learns purely through self-play, using an **Actor-Critic neural network** guided by **Monte-Carlo-Tree Search (MCTS)** — the same general approach popularized by AlphaZero. No human game data is used; the agent starts from random play and improves by repeatedly playing against itself.

## How it works

For a given game, the training loop (`trainer/rl_trainer.py`) repeats the following cycle:

1. **Self-play data generation** — the current best model, combined with MCTS, plays full games against itself across multiple worker processes (`trainer/data_generator/`). Each move records the board state, the MCTS-refined action probabilities, and the eventual game outcome.
2. **Training** — the recorded (state, action probabilities, outcome) triples are sampled from a replay buffer (`trainer/buffers/`) to train the Actor-Critic network (`trainer/algorithms/actor_critic/`), which predicts both a move distribution (policy) and an expected outcome (value) for a given board state.
3. **Testing** — the newly trained network plays a batch of games against a random-move agent to sanity-check basic progress (`trainer/test/`).
4. **Tournament** — the newly trained network plays a batch of games against the previous best network (`trainer/tournament/`). It only replaces the current best model if it wins a large enough fraction of games, so training never regresses.
5. **Logging** — win rate, loss and step counts are recorded after every iteration (`trainer/Logger/`) and can be plotted later.

The three games share this same training/MCTS/Actor-Critic infrastructure; each only supplies its own environment (rules, board representation and rendering) implementing a common `reset` / `step` / `render` interface, so the same agent code trains all three.

## Games

The games themselves, which also function as the reinforcement learning training environments, were implemented from scratch using [PyGame](https://www.pygame.org/) for rendering and NumPy for board state.

### Tic-Tac-Toe
<p align="center">
  <img src="https://github.com/user-attachments/assets/e3921e88-6758-4168-9b58-ec05cd691f6d" width=50% height=50%>
</p>

### Connect 4
<p align="center">
  <img src="https://github.com/user-attachments/assets/31d67904-94c9-4ad4-b3d2-2285ae135b44" width=50% height=50%>
</p>

### Checkers
<p align="center">
  <img src="https://github.com/user-attachments/assets/1051e6d9-888d-4f1f-96f2-276a55b21595" width=50% height=50%>
</p>

## Project structure

```
BoardGameAI/
├── checkers/                       # Checkers environment, parameters and saved models
├── simple_games/
│   ├── connect4/                   # Connect 4 environment, parameters and saved models
│   └── tic_tac_toe/                # Tic-Tac-Toe environment, parameters and saved models
├── trainer/                        # Game-agnostic training infrastructure
│   ├── rl_trainer.py                #   Main training loop
│   ├── algorithms/actor_critic/      #   Actor-Critic network and training step
│   ├── monte_carlo_tree_search/      #   MCTS used during self-play and testing
│   ├── data_generator/                #   Parallel self-play data generation
│   ├── buffers/                       #   Replay buffer for training data
│   ├── tournament/                    #   Best-model-so-far comparison
│   ├── self_play_model/               #   Lightweight inference-only model wrapper
│   └── Logger/                        #   Training metrics logging/plotting
├── run.py                          # CLI entry point for training/testing (see Makefile)
├── view_log.py                     # CLI entry point for plotting a saved training log
├── Makefile                        # Convenience wrapper around run.py / view_log.py
└── requirements.txt                # Python dependencies
```

Each game directory (`checkers/`, `simple_games/connect4/`, `simple_games/tic_tac_toe/`) follows the same layout: an `*_env/` package with the game rules, board and drawer, a `*_parameters/` module with training/model hyperparameters, `train_<game>.py` / `test_<game>.py` entry points, and a `saved_models/` folder where trained checkpoints and logs are written.

## Installation

**Requirements:** Python 3.10+ and `pip`.

1. Clone the repository and move into it:
   ```bash
   git clone <repository-url>
   cd BoardGameAI
   ```

2. (Recommended) create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This installs TensorFlow/Keras (the Actor-Critic model), NumPy (board state), Numba (JIT-compiled move generation), PyGame (rendering) and Matplotlib (log plotting).

The repository already ships with a handful of pretrained checkpoints under each game's `saved_models/` folder, so you can jump straight to testing/playing without training anything yourself.

## Usage

The `Makefile` is the easiest way to run things; each command has an equivalent direct `python3` invocation below it in case `make` isn't available (e.g. on Windows without WSL/Git Bash) or you want to inspect what it's doing.

### Training a model

```bash
make train game:=<game>
# equivalent to:
python3 run.py train <game>
```

Where `<game>` is one of `checkers`, `connect4`, `tic_tac_toe`. This runs the full self-play/train/test/tournament loop described above for the number of iterations configured for that game (see [Configuring training](#configuring-training)), saving checkpoints and logs to that game's `saved_models/` directory after every iteration. Each iteration spawns multiple self-play worker processes and trains a neural network, so it's CPU/GPU-intensive — expect it to take a while, especially for checkers.

### Testing / playing against a model

```bash
make test game:=<game> human:=<human> eps:=<episodes> num_sim:=<simulations> \
          ai_player:=<ai_player> heuristic_w:=<heuristic_weight> model_name:=<model_name>
# equivalent to:
python3 run.py test <game> <human> <episodes> <simulations> <ai_player> <heuristic_weight> <model_name>
```

| Argument | Meaning |
|---|---|
| `game` | `checkers`, `connect4` or `tic_tac_toe` |
| `human` | `1` to control the non-AI side yourself with mouse clicks in the PyGame window. **Note:** this is parsed with Python's `bool(...)`, so any non-empty value (including `0`) is treated as `True`; to have the opponent move randomly, call `run.py` directly with an empty string in this position (`make` can't pass an empty positional argument), e.g. `python3 run.py test checkers "" 1 100 -1 0 <model_name>` |
| `eps` | number of games to play |
| `num_sim` | number of MCTS simulations run per move (higher = stronger but slower play) |
| `ai_player` | which side (`1` or `-1`) is controlled by the model |
| `heuristic_w` | weight (`0`-`1`) given to the game's hand-crafted heuristic during MCTS, on top of the network's own evaluation |
| `model_name` | filename of a trained checkpoint inside that game's `saved_models/` folder, e.g. `actor-critic-CHECKERS-month-6-day-8-12-62%.h5` |

Example — play checkers yourself against a pretrained model, as player 1, with 100 MCTS simulations per AI move (checkpoint filenames contain `%`, so quote the value):

```bash
python3 run.py test checkers 1 1 100 -1 0 "actor-critic-CHECKERS-month-6-day-8-12-62%.h5"
```

### Viewing training logs

```bash
make view-log game:=<game_path> log_name:=<log_name>
# equivalent to:
python3 view_log.py <game_path> <log_name>
```

This loads a saved `Logger` (pickled during training) from `<game_path>/saved_models/saved_logs/<log_name>` and plots its recorded metrics. Note that `game_path` here is the *relative path to the game's folder from the repository root* rather than the short game name used above: `checkers`, `simple_games/connect4` or `simple_games/tic_tac_toe`.

## Configuring training

Each game's hyperparameters live in its own `*_parameters.py` module (e.g. `checkers/checkers_parameters/checkers_parameters.py`), split into:

- `trainer_parameters` — training loop settings: number of iterations, self-play episodes and workers, replay buffer size, MCTS budget per phase, tournament/test game counts, and whether to resume from an existing checkpoint (`LOAD`, `MODEL_PATH`, `LOGGER_PATH`).
- `actor_critic_parameters` — network settings: learning rate, batch size, training iterations per cycle, and convolutional network size/regularization.

Board and rendering settings (window size, colors, piece sizes) live separately in each game's `env_parameters` module, e.g. `checkers/checkers_env/env_parameters/checkers_env_parameters.py`.

## License

Released under the [MIT License](LICENSE).
