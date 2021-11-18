import os
import os
import sys

from imports import *
from drivers.driver import Driver
from nikola.car_dynamics_tracker import RookieCarDynamicsTracker
from nikola.race_logger import RaceLogger
from nikola.straight_simulator import StraightSimulator
from nikola.turn_chooser import MultipleClosestTurnChooser
from nikola.turn_tracker import RealtimeTurnTracker


class MyRookieDriver(Driver):

    def __init__(self, name: str, race_logger_dir: str):
        super().__init__(name)
        self.race_logger_dir = race_logger_dir

        self.turn_tracker = RealtimeTurnTracker()
        self.turn_chooser = MultipleClosestTurnChooser(3, True)
        self.race_logger = None

        self.race_id = 0

    def prepare_for_race(self):
        # Next race id
        self.race_id += 1
        # Initialize this race logger
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        # Start new race in the turn tracker
        self.turn_tracker.new_race()

    def make_a_move(self, car_state: CarState, track_state: TrackState) -> Action:
        return ""

    def update_with_action_results(self,
            previous_car_state: CarState, previous_track_state: TrackState, action: Action,
            new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        # Log using the race logger
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        # Update turn tracker
        self.turn_tracker.new_track_state(previous_track_state, is_final=False)
        if result.finished:
            self.turn_tracker.new_track_state(new_track_state, is_final=True)

    def update_after_race(self, correct_turns: Dict[Position, Action]):
        # Check turn tracker validity
        self.turn_tracker.update_after_race(correct_turns)


def main():
    pass

if __name__ == '__main__':
    main()
