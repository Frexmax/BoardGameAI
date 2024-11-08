from TrainerActorCritic import TrainerActorCritic
from TrainerActorCriticV2 import TrainerActorCriticV2
from TicTacToeEnvParameters import board_parameters, draw_parameters
from tic_tac_toe_parameters import actor_critic_trainer_parameters, actor_critic_parameters, \
    actor_critic_trainer_parametersV2

TRAINING_MODE = "ITERATION"
if __name__ == '__main__':
    # CREATE TRAINER
    if TRAINING_MODE == "STEP":
        trainer = TrainerActorCritic(actor_critic_trainer_parameters, actor_critic_parameters, env_name="TIC_TAC_TOE",
                                     draw_parameters=draw_parameters, board_parameters=board_parameters)
    else:
        trainer = TrainerActorCriticV2(actor_critic_trainer_parametersV2, actor_critic_parameters,
                                       env_name="TIC_TAC_TOE", draw_parameters=draw_parameters,
                                       board_parameters=board_parameters)
    trainer.run()


