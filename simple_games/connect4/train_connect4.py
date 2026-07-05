from simple_games.connect4.connect4_env.env_parameters.connect4_env_parameters import board_parameters, draw_parameters
from simple_games.connect4.connect4_parameters.connect4_parameters import trainer_parameters, actor_critic_parameters
from trainer.rl_trainer import TrainerActorCriticV2


def train_connect4():
    """
    Entry point for training a connect4 agent.
    Initializes the Trainer with the connect4 environment and parameters, then runs the training loop
    """

    trainer = TrainerActorCriticV2(trainer_parameters, actor_critic_parameters, env_name="CONNECT4",
                                   draw_parameters=draw_parameters, board_parameters=board_parameters)
    trainer.run()
