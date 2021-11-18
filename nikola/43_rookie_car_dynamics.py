import os
import os
import sys

from imports import *
from drivers.rookiedriver import RookieDriver
from nikola.race_logger import RaceLogger
from nikola.car_dynamics_tracker import RookieCarDynamicsTracker
from nikola.straight_simulator import StraightSimulator


class RookieDriverCarDynamics(RookieDriver):

    def __init__(self, name,
            race_logger_dir,
            car_dynamics_tracker: RookieCarDynamicsTracker,
            use_target_speeds: bool,
            *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.car_dynamics_tracker = car_dynamics_tracker
        self.use_target_speeds = use_target_speeds
        self.race_id = 0
        self.race_logger = None

        self.potential_stop = False

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        self.potential_stop = False
        return super().prepare_for_race()

    def _get_target_speed(self, distance_ahead, safety_car_active, target_speeds=None):
        if distance_ahead == 0:
            return 0
        if self.use_target_speeds:
            target_speed = StraightSimulator.get_target_speeds(self.car_dynamics_tracker, distance_ahead, self.potential_stop)[-1]
            if safety_car_active:
                target_speed = min(target_speed, self.safety_car_speed)
            target_speed = min(target_speed, 350.0)
            return target_speed
        else:
            return super()._get_target_speed(distance_ahead, safety_car_active, target_speeds=target_speeds)

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
        self.car_dynamics_tracker.add_data_point(action, previous_car_state, new_car_state, result)
        if previous_track_state.distance_ahead == 0 and new_track_state.distance_ahead > 0:
            if previous_track_state.distance_left > 0 and previous_track_state.distance_right > 0:
                self.potential_stop = True
            else:
                self.potential_stop = False
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result)


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run one season with turn logger
    for use_target_speeds in [False, True]:
        print('Running season with target speeds {}'.format(use_target_speeds))
        Season(Level.Rookie).race(
            RookieDriverCarDynamics(
                'RDCD',
                os.path.join(path, 'season_{}'.format(use_target_speeds)),
                RookieCarDynamicsTracker(),
                use_target_speeds))

    # Run the championship with the original rookie
    print('Running championship with no target speeds')
    drivers = [
        RookieDriver('RD'),
        RookieDriverCarDynamics('RDCD', '', RookieCarDynamicsTracker(), False)
    ]
    championship = Championship(drivers, Level.Rookie, shuffle_tracks=True, verbose=True)
    championship_results, finishing_positions, all_race_times = championship.run_championship(num_repeats=10)
    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'championship_no_target_speeds.png'))
    plt.close()

    # Run the championship with the original rookie
    print('Running championship with target speeds')
    drivers = [
        RookieDriver('RD'),
        RookieDriverCarDynamics('RDCD', '', RookieCarDynamicsTracker(), True)
    ]
    championship = Championship(drivers, Level.Rookie, shuffle_tracks=True, verbose=True)
    championship_results, finishing_positions, all_race_times = championship.run_championship(num_repeats=10)
    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'championship_target_speeds.png'))
    plt.close()

    won_races = finishing_positions['RDCD'] < finishing_positions['RD']
    print('won_races')
    print(won_races)
    won_time_diff = all_race_times['RD'][won_races] - all_race_times['RDCD'][won_races]
    print('won_time_diff')
    print(won_time_diff)
    print(sum(won_time_diff) / len(won_time_diff))

    lost_races = finishing_positions['RDCD'] > finishing_positions['RD']
    print('lost_races')
    print(lost_races)
    lost_time_diff = all_race_times['RDCD'][lost_races] - all_race_times['RD'][lost_races]
    print('lost_time_diff')
    print(lost_time_diff)
    print(sum(lost_time_diff) / len(lost_time_diff))
