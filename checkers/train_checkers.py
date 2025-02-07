from checkers.checkers_env.env_parameters.checkers_env_parameters import board_parameters, draw_parameters
from checkers.checkers_parameters.checkers_parameters import trainer_parameters, actor_critic_parameters
from trainer.rl_trainer import TrainerActorCriticV2


def train_checkers():
    trainer = TrainerActorCriticV2(trainer_parameters, actor_critic_parameters, env_name="CHECKERS",
                                   draw_parameters=draw_parameters, board_parameters=board_parameters)
    trainer.run()
