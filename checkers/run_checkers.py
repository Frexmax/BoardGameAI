from checkers.checkers_env.env_parameters.checkers_env_parameters import board_parameters, draw_parameters

from checkers_parameters.checkers_parameters import (actor_critic_trainer_parameters,
                                                     actor_critic_trainer_parametersV2, actor_critic_parameters)

from trainer.trainer_actor_critic import TrainerActorCritic
from trainer.trainer_actor_critic_v2 import TrainerActorCriticV2


TRAINING_MODE = "ITERATION"
if __name__ == '__main__':
    # CREATE TRAINER
    if TRAINING_MODE == "STEP":
        trainer = TrainerActorCritic(actor_critic_trainer_parameters, actor_critic_parameters, env_name="CHECKERS",
                                     draw_parameters=draw_parameters, board_parameters=board_parameters)
    else:
        trainer = TrainerActorCriticV2(actor_critic_trainer_parametersV2, actor_critic_parameters, env_name="CHECKERS",
                                       draw_parameters=draw_parameters, board_parameters=board_parameters)
    trainer.play_tournament()
    trainer.run()
