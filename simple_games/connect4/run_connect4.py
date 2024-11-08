from connect4_env.env_parameters.connect4_env_parameters import board_parameters, draw_parameters
from connect4_parameters.connect4_parameters import (actor_critic_trainer_parameters,
                                                     actor_critic_parameters, actor_critic_trainer_parametersV2)

from trainer.trainer_actor_critic import TrainerActorCritic
from trainer.trainer_actor_critic_v2 import TrainerActorCriticV2


TRAINING_MODE = "ITERATION"
if __name__ == '__main__':
    # CREATE TRAINER
    if TRAINING_MODE == "STEP":
        trainer = TrainerActorCritic(actor_critic_trainer_parameters, actor_critic_parameters, env_name="CONNECT4",
                                     draw_parameters=draw_parameters, board_parameters=board_parameters)
    else:
        trainer = TrainerActorCriticV2(actor_critic_trainer_parametersV2, actor_critic_parameters, env_name="CONNECT4",
                                       draw_parameters=draw_parameters, board_parameters=board_parameters)
    trainer.run()

