from trainer.rl_trainer import TrainerActorCriticV2
from simple_games.tic_tac_toe.tic_tac_toe_env.env_parameters.tic_tac_toe_env_parameters import board_parameters, draw_parameters
from simple_games.tic_tac_toe.tic_tac_toe_parameters.tic_tac_toe_parameters import trainer_parameters, actor_critic_parameters


def train_tic_tac_toe():
    """
    Entry point for training a tic-tac-toe agent.
    Initializes the Trainer with the tic-tac-toe environment and parameters, then runs the training loop
    """

    trainer = TrainerActorCriticV2(trainer_parameters, actor_critic_parameters,
                                   env_name="TIC_TAC_TOE", draw_parameters=draw_parameters,
                                   board_parameters=board_parameters)
    trainer.run()


