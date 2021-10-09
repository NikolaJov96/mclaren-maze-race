import os
from random import choice
import sys

from imports import *
from drivers.youngdriver import YoungDriver
from nikola.turn_tracker import TurnTracker


class TurnTrackerYoungDriver(YoungDriver):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.turn_tracker = TurnTracker()
        self.correct_turns = self.turn_tracker.correct_turns
        self.track_num = 0

    def prepare_for_race(self):
        self.turn_tracker.new_race()
        self.track_num += 1

    def make_a_move(self, car_state: CarState, track_state: TrackState):
        self.turn_tracker.new_track_state(track_state)
        return super().make_a_move(car_state, track_state)

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult):

        if result.finished:
            self.turn_tracker.new_track_state(new_track_state, is_final=True)

        return super().update_with_action_results(previous_car_state, previous_track_state, action, new_car_state, new_track_state, result)

    def update_after_race(self, correct_turns: Dict[Position, Action]):
        # May fail if the driver did not manage to finish the track in allowed time
        assert self.turn_tracker._this_race_correct_turns == correct_turns

    def __setattr__(self, name, value):
        if name == 'turn_tracker':
            super(TurnTrackerYoungDriver, self).__setattr__(name, value)
            super(TurnTrackerYoungDriver, self).__setattr__('correct_turns', value.correct_turns)
        elif name == 'correct_turns':
            pass
        else:
            super(TurnTrackerYoungDriver, self).__setattr__(name, value)


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Fixed young driver parameters
    speed_rounding = 40
    max_distance = 8
    learning_rate = 0.25

    # Chanpionship parameters
    num_repeats = 10

    # Compare drivers with different min distances
    drivers = [
        YoungDriver('Original', speed_rounding=speed_rounding, max_distance=max_distance, learning_rate=learning_rate),
        TurnTrackerYoungDriver('TurnTracker', speed_rounding=speed_rounding, max_distance=max_distance, learning_rate=learning_rate)
    ]

    championship = Championship(drivers, Level.Young, shuffle_tracks=False, verbose=True)
    # Skip track 21, as young driver is not capable of finishing it on time
    track_indices = list(range(TrackStore.get_number_of_tracks(level=Level.Young)))
    track_indices.remove(21 - 1)
    championship_results, race_results, race_times = championship.run_championship(num_repeats=num_repeats, track_indices=track_indices)

    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'original_vs_turn_tracker.png'))
