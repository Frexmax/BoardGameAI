ppo_trainer_parameters = {"LOAD": False, "MODEL_PATH": None, "EPOCHS": 10, "EPOCH_STEPS": 4000, "TEST_EVERY": 5,
                          "TEST_GAMES": 10, "TEST_SIMULATIONS": 50, "GAMMA": 0.99, "LAM": 0.97,
                          "MCTS_SIMULATIONS": 150,  "MCTS_ON": False, "TOURNAMENT_EVERY": 5,
                          "TOURNAMENT_GAMES": 5, "TOURNAMENT_SIMULATIONS": 50}

ppo_parameters = {"ACTOR_LRN": 0.0003, "CRITIC_LRN": 0.0001, "BATCH_SIZE": 256, "CLIP_RATIO": 0.2, "TARGET_KL": 0.01,
                  "ACTOR_ITERATIONS": 80, "CRITIC_ITERATIONS": 80}

# STEP TRAINING
actor_critic_trainer_parameters = {"LOAD": False, "MODEL_PATH": None, "EPISODES": 10_000, "TEST_EVERY": 400,
                                   "TEST_GAMES": 10, "TEST_BUDGET": 100, "DECAY": 0.9, "TOURNAMENT_EVERY": 200,
                                   "TOURNAMENT_GAMES": 30, "TOURNAMENT_BUDGET": 100, "MEMORY_SIZE": 200_000,
                                   "MIN_REPLAY_SIZE": 1000, "MCTS_BUDGET": 100, "SAVE_EVERY": 400}

# ITERATION TRAINING
actor_critic_trainer_parametersV2 = {"LOAD": True, "MODEL_PATH": "saved_models/actor-critic-CHECKERS-month-6-day-8-12-62%.h5",
                                     "LOGGER_PATH": "saved_models/SavedLogs/log-CHECKERS-month-6-day-8-ep-125551-62.5%.pkl",
                                     "TEST_GAMES": 16, "TEST_BUDGET": 250, "DECAY": 1, "TOURNAMENT_GAMES": 16,
                                     "TOURNAMENT_BUDGET": 100, "MEMORY_SIZE": 40_000, "MCTS_BUDGET": 600,
                                     "NUM_WORKERS": 8, "ITERATIONS": 40, "DATA_GENERATION_EPISODES": 120,
                                     "THRESHOLD": 0.55, "HEURISTIC_START_WEIGHT": 0, "HEURISTIC_END_WEIGHT": 0,
                                     "HEURISTIC_STEPS": 10}

actor_critic_parameters = {"LEARNING_RATE": 0.005, "BATCH_SIZE": 128, "TRAINING_ITERATIONS": 60, "STEP_SIZE": 1000,
                           "NUM_KERNELS": 128, "CONV_REGULARIZATION": 0.001, "DENSE_REGULARIZATION": 0.001}
