from trainer.rl_trainer import TrainerActorCriticV2
from tic_tac_toe_env.env_parameters.tic_tac_toe_env_parameters import board_parameters, draw_parameters
from tic_tac_toe_parameters.tic_tac_toe_parameters import trainer_parameters, actor_critic_parameters

if __name__ == '__main__':
    trainer = TrainerActorCriticV2(trainer_parameters, actor_critic_parameters,
                                   env_name="TIC_TAC_TOE", draw_parameters=draw_parameters,
                                   board_parameters=board_parameters)
    trainer.run()


