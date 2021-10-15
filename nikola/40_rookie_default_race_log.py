import os
from random import choice
import os
import sys

from imports import *
from drivers.rookiedriver import RookieDriver
from nikola.race_logger import RaceLogger


class RookieDriverLooged(RookieDriver):

    def __init__(self, name, race_logger_dir, random_action_probability=0.5, random_action_decay=0.99,
            min_random_action_probability=0, *args, **kwargs):
        super().__init__(name, random_action_probability=random_action_probability,
            random_action_decay=random_action_decay,
            min_random_action_probability=min_random_action_probability, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.race_id = 0
        self.race_logger = None

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        return super().prepare_for_race()

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result)


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run one season with turn logger
    Season(Level.Rookie).race(RookieDriverLooged(
        'RD', race_logger_dir=os.path.join(path, 'race_logger')))
