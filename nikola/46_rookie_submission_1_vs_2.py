import os
import sys

from nikola.race_logger import RaceLogger

from imports import *
from drivers.my_rookie_driver import MyDriver as Submission1
from drivers.my_rookie_driver_2 import MyDriver as Submission2


class S1Logged(Submission1):
    """
    Logged version of submission 1
    """

    def __init__(self, name, race_logger_dir):
        super().__init__(name)

        self.race_logger_dir = race_logger_dir
        self.race_logger = None

    def prepare_for_race(self):
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id + 1)))
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


class S2Logged(Submission2):
    """
    Logged version of submission 2
    """

    def __init__(self, name, race_logger_dir):
        super().__init__(name)

        self.race_logger_dir = race_logger_dir
        self.race_logger = None

    def prepare_for_race(self):
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id + 1)))
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


class RookieDriverLooged(RookieDriver):
    """
    Taken from 41_rookie_turn_choosing.py
    """

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


def main():
    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run one championship with the original rookie
    print('Running championship')
    drivers = [
        RookieDriverLooged('RD', os.path.join(path, 'RD')),
        S1Logged('S1', os.path.join(path, 'S1')),
        # S2Logged('S2', os.path.join(path, 'S2'), [2, 11, 22])
        # S2Logged('S2', os.path.join(path, 'S2'), list(range(1, 25)))
        S2Logged('S2', os.path.join(path, 'S2'))
    ]
    championship = Championship(drivers, Level.Rookie, shuffle_tracks=False, verbose=True)
    championship_results, finishing_positions, all_race_times = championship.run_championship(num_repeats=1)
    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'championship.png'))
    plt.close()

    with open(os.path.join(path, 'championship.txt'), 'w') as out_file:
        out_file.write('all_race_times\n')
        out_file.write(str(all_race_times))
        out_file.write('\n')

        for driver in all_race_times:
            out_file.write(driver)
            out_file.write('\n')
            won_races = finishing_positions[driver] == 1
            if sum([sum(won_sub_races) for won_sub_races in won_races]) == 0:
                continue

            out_file.write('won_races\n')
            out_file.write(str(won_races))
            out_file.write('\n')
            for other_driver in all_race_times:
                if other_driver == driver:
                    continue
                won_time_diff = all_race_times[other_driver][won_races] - all_race_times[driver][won_races]
                out_file.write('won_time_diff\n')
                out_file.write(str(won_time_diff))
                out_file.write('\n')
                out_file.write(str(sum(won_time_diff) / len(won_time_diff)))
                out_file.write('\n')

            out_file.write('\n')


if __name__ == '__main__':
    main()
