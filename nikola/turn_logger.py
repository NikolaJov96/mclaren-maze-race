import os

from imports import *


class TurnLogger:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.race = None
        self.race_id = 0
        self.turn_id = 0

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def new_race(self):

        self.race = []
        self.race_id += 1
        self.turn_id = 0

    def log_turn(self, correct_turns, position, action):

        race_save_dir = os.path.join(self.save_dir, 'race_{}'.format(self.race_id))
        if not os.path.exists(race_save_dir):
            os.mkdir(race_save_dir)

        self.turn_id += 1

        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.set_title('Race {} turn {}'.format(self.race_id, self.turn_id))
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])
        ax.invert_yaxis()

        for turn_position, correct_action in correct_turns.items():
            if correct_action == Action.TurnLeft:
                ax.plot(turn_position.column, turn_position.row, 'r+')
            else:
                ax.plot(turn_position.column, turn_position.row, 'g+')

        if action == Action.TurnLeft:
            ax.plot(position.column, position.row, 'ro')
        else:
            ax.plot(position.column, position.row, 'go')

        plt.savefig(os.path.join(race_save_dir, 'turn_{}.png'.format(self.turn_id)))
        plt.close(fig)
