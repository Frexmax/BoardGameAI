from simple_games.connect4.connect4_env.env_parameters.connect4_env_parameters import board_parameters, draw_parameters
from simple_games.connect4.connect4_parameters.connect4_parameters import trainer_parameters, actor_critic_parameters
from trainer.rl_trainer import TrainerActorCriticV2


def train_connect4():
    trainer = TrainerActorCriticV2(trainer_parameters, actor_critic_parameters, env_name="CONNECT4",
                                   draw_parameters=draw_parameters, board_parameters=board_parameters)
    trainer.run()
