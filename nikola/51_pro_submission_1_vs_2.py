import os
import sys

from nikola.race_logger import RaceLogger

from imports import *
from drivers.my_pro_driver import MyDriver as Submission1
from drivers.my_pro_driver_2 import MyDriver as Submission2


class S1Logged(Submission1):

    def __init__(self, name, race_logger_dir, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.race_id = 0
        self.race_logger = None

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        return super().prepare_for_race()

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult,
            previous_weather_state: WeatherState):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result, previous_weather_state)


class S2Logged(Submission2):

    def __init__(self, name, race_logger_dir, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.race_id = 0
        self.race_logger = None

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        return super().prepare_for_race()

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult,
            previous_weather_state: WeatherState):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result, previous_weather_state)


def main():
    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run races
    set_seed(0)
    _, s1_race_times, _ = Season(Level.Pro).race(S1Logged('S1', os.path.join(path, 'S1')))
    set_seed(0)
    _, s2_race_times, _ = Season(Level.Pro).race(S2Logged('S2', os.path.join(path, 'S2')))

    race_times = [s2_race_times]

    with open(os.path.join(path, 'championship.txt'), 'w') as out_file:

        out_file.write('all_race_times\n')
        out_file.write(str(s1_race_times))
        out_file.write('\n')

        for i in range(len(race_times)):
            print(i)
            out_file.write('\n')

            s2_race_times = race_times[i]

            s1_won = [max(i, 0) for i in s2_race_times - s1_race_times]
            s2_won = [max(i, 0) for i in s1_race_times - s2_race_times]
            s1_co = sum([i > 0 for i in s2_race_times - s1_race_times])
            s2_co = sum([i > 0 for i in s1_race_times - s2_race_times])

            out_file.write('all_race_times\n')
            out_file.write(str(s2_race_times))
            out_file.write('\n')

            out_file.write('s1_won\n')
            out_file.write(str(s1_won))
            out_file.write('\n')

            out_file.write('s2_won\n')
            out_file.write(str(s2_won))
            out_file.write('\n')

            out_file.write('victories\n')
            out_file.write(str(s1_co))
            out_file.write('\n')
            out_file.write(str(s2_co))
            out_file.write('\n')

            out_file.write('\n')


if __name__ == '__main__':
    main()
