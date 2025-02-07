# BoardGameAI
Project developing AI agents for 3 board games of increasing difficulty:
1. Tic-Tac-Toe
2. Connect 4
3. Checkers
   
Each agent is trained via reinforcement learning using the Actor-Critic algorithm, combined with Monte-Carlo-Tree Search.

## Games 
The games themselves, which also function as reinforcement learning training environments, were implemented using [PyGame](https://github.com/pygame)

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

## Running the Program

### Installing dependencies
```cmd
pip install -r requirements.txt
```

### Train models
```cmd
make train game:=<game name>
```

### Test models in a game
```cmd
make test game:=<game name> human:=<is human> eps:=<ep count> 
                num_sim:=<MCTS sim num> ai_player:=<which player ai> heuristic_w:=<heuristic weight (0-1)>
                model_name:=<model name>
```

### View training logs
```cmd
make make view-log game:=<game name> log_name:=<log name>
```

## Explanation of Actor-Critic Learning

## Explanation of Monte-Carlo-Tree Search

## Technical Documentation
If you are interested in more detail about how the code works, please visit the docs section of the repository
