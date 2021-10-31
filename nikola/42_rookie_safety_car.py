import os
import os
import sys

from imports import *
from drivers.rookiedriver import RookieDriver
from nikola.race_logger import RaceLogger
from nikola.safety_car_tracker import SafetyCarTracker


class RookieDriverSafetyCar(RookieDriver):

    def __init__(self, name, race_logger_dir, safety_car_tracker: SafetyCarTracker, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.safety_car_tracker = safety_car_tracker
        self.race_id = 0
        self.race_logger = None

        self.safety_car_target_speed = 0

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        self.safety_car_tracker.new_race()
        return super().prepare_for_race()

    def make_a_move(self, car_state: CarState, track_state: TrackState, **kwargs):
        self.safety_car_target_speed = self.safety_car_tracker.new_track_state(track_state)
        return super().make_a_move(car_state, track_state, **kwargs)

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        self.safety_car_tracker.action_result(new_car_state, result)
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result)

    def _get_target_speed(self, distance_ahead, safety_car_active, target_speeds=None):
        if target_speeds is None:
            target_speeds = self.target_speeds

        if distance_ahead == 0:
            target_speed = 0
        else:
            target_speed = target_speeds[distance_ahead - 1]

        if safety_car_active:
            target_speed = min(target_speed, self.safety_car_target_speed)

        return target_speed


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run one season with turn logger
    print('Running season')
    Season(Level.Rookie).race(
        RookieDriverSafetyCar(
            'RFSC',
            os.path.join(path, 'season'),
            SafetyCarTracker()))

    # Run the championship with the original rookie
    print('Running championship')
    drivers = [
        RookieDriver('RD'),
        RookieDriverSafetyCar('RFSC', '', SafetyCarTracker())
    ]
    championship = Championship(drivers, Level.Rookie, shuffle_tracks=True, verbose=True)
    championship_results, _, _ = championship.run_championship(num_repeats=100)
    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'championship.png'))
    plt.close()
