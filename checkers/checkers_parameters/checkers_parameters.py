# Iteration training parameters
trainer_parameters = {"LOAD": True, "MODEL_PATH": "saved_models/actor-critic-CHECKERS-month-6-day-8-12-62%.h5",
                      "LOGGER_PATH": "saved_models/saved_logs/log-CHECKERS-month-6-day-8-ep-125551-62.5%.pkl",
                      "TEST_GAMES": 16, "TEST_BUDGET": 250, "DECAY": 1, "TOURNAMENT_GAMES": 16,
                      "TOURNAMENT_BUDGET": 100, "MEMORY_SIZE": 40_000, "MCTS_BUDGET": 600,
                      "NUM_WORKERS": 8, "ITERATIONS": 40, "DATA_GENERATION_EPISODES": 120,
                      "THRESHOLD": 0.55, "HEURISTIC_START_WEIGHT": 0, "HEURISTIC_END_WEIGHT": 0,
                      "HEURISTIC_STEPS": 10}

actor_critic_parameters = {"LEARNING_RATE": 0.005, "BATCH_SIZE": 128, "TRAINING_ITERATIONS": 60, "STEP_SIZE": 1000,
                           "NUM_KERNELS": 128, "CONV_REGULARIZATION": 0.001, "DENSE_REGULARIZATION": 0.001}
