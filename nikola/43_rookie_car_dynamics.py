import os
import os
import sys

from imports import *
from drivers.rookiedriver import RookieDriver
from nikola.race_logger import RaceLogger
from nikola.car_dynamics_tracker import RookieCarDynamicsTracker


class RookieDriverCarDynamics(RookieDriver):

    def __init__(self, name, race_logger_dir, car_dynamics_tracker: RookieCarDynamicsTracker, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.car_dynamics_tracker = car_dynamics_tracker
        self.race_id = 0
        self.race_logger = None

        self.safety_car_target_speed = 0

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        return super().prepare_for_race()

    def estimate_next_speed(self, action: Action, speed, drs_active: bool, **kwargs):
        if action == Action.Continue:
            return speed
        return self.car_dynamics_tracker.estimate_next_speed(speed, action, drs_active)

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        if self.car_dynamics_tracker.data[False][Action.FullThrottle] != self.sl_data[Action.FullThrottle]:
            print(self.car_dynamics_tracker.data[False][Action.FullThrottle])
            print(self.sl_data[Action.FullThrottle])
        assert self.car_dynamics_tracker.data[False][Action.FullThrottle] == self.sl_data[Action.FullThrottle]
        self.car_dynamics_tracker.add_data_point(action, previous_car_state, new_car_state)
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result)


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run one season with turn logger
    print('Running season')
    Season(Level.Rookie).race(
        RookieDriverCarDynamics(
            'RDCD',
            os.path.join(path, 'season'),
            RookieCarDynamicsTracker()))

    # Run the championship with the original rookie
    print('Running championship')
    drivers = [
        RookieDriver('RD'),
        RookieDriverCarDynamics('RDCD', '', RookieCarDynamicsTracker())
    ]
    championship = Championship(drivers, Level.Rookie, shuffle_tracks=True, verbose=True)
    championship_results, _, _ = championship.run_championship(num_repeats=10)
    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'championship.png'))
    plt.close()
