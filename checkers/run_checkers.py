from TrainerActorCritic import TrainerActorCritic
from TrainerActorCriticV2 import TrainerActorCriticV2
from CheckersEnvParameters import board_parameters, draw_parameters
from checkers_parameters import actor_critic_trainer_parameters, actor_critic_trainer_parametersV2, \
    actor_critic_parameters

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
