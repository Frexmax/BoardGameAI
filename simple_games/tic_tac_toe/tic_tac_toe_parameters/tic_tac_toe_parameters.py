# Iteration training parameters
actor_critic_trainer_parametersV2 = {"LOAD": False, "MODEL_PATH": None,
                                     "LOGGER_PATH": None,
                                     "TEST_GAMES": 16, "TEST_BUDGET": 20, "DECAY": 0.9, "TOURNAMENT_GAMES": 16,
                                     "TOURNAMENT_BUDGET": 100, "MEMORY_SIZE": 40_000, "MCTS_BUDGET": 200,
                                     "NUM_WORKERS": 8, "ITERATIONS": 20, "DATA_GENERATION_EPISODES": 640,
                                     "THRESHOLD": 0.55, "HEURISTIC_START_WEIGHT": 0, "HEURISTIC_END_WEIGHT": 0,
                                     "HEURISTIC_STEPS": 10}

# Parameters for the Actor-Critic model
actor_critic_parameters = {"LEARNING_RATE": 0.001, "BATCH_SIZE": 64, "TRAINING_ITERATIONS": 40, "STEP_SIZE": 1000,
                           "NUM_KERNELS": 128, "CONV_REGULARIZATION": 0.001, "DENSE_REGULARIZATION": 0.001}
