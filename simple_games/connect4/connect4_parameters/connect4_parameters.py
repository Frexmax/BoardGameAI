# STEP TRAINING
actor_critic_trainer_parameters = {"LOAD": False, "MODEL_PATH": None, "EPISODES": 10_000, "TEST_EVERY": 250,
                                   "TEST_GAMES": 10, "TEST_BUDGET": 100, "DECAY": 0.9, "TOURNAMENT_EVERY": 250,
                                   "TOURNAMENT_GAMES": 50, "TOURNAMENT_BUDGET": 100, "MEMORY_SIZE": 200_000,
                                   "MIN_REPLAY_SIZE": 1000, "MCTS_BUDGET": 100, "SAVE_EVERY": 500}

# ITERATION TRAINING
actor_critic_trainer_parametersV2 = {"LOAD": True, "MODEL_PATH": "saved_models/actor-critic--month-5-day-22-8-100%.h5",
                                     "LOGGER_PATH": "saved_models/saved_logs/log-CONNECT4-month-5-day-20-ep-24949-100.0%.pkl",
                                     "TEST_GAMES": 6, "TEST_BUDGET": 100, "DECAY": 1, "TOURNAMENT_GAMES": 18,
                                     "TOURNAMENT_BUDGET": 100, "MEMORY_SIZE": 40_000, "MCTS_BUDGET": 800,
                                     "NUM_WORKERS": 6, "ITERATIONS": 20, "DATA_GENERATION_EPISODES": 240,
                                     "THRESHOLD": 0.55, "HEURISTIC_START_WEIGHT": 0, "HEURISTIC_END_WEIGHT": 0,
                                     "HEURISTIC_STEPS": 1}

actor_critic_parameters = {"LEARNING_RATE": 0.001, "BATCH_SIZE": 128, "TRAINING_ITERATIONS": 40, "STEP_SIZE": 1000,
                           "NUM_KERNELS": 128, "CONV_REGULARIZATION": 0.001, "DENSE_REGULARIZATION": 0.001}
