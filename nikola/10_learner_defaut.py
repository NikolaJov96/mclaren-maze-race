import numpy as np

from drivers.learnerdriver import LearnerDriver
from imports import *


if __name__ == '__main__':

    driver_name = 'Learner Nikola'

    level = Level.Learner
    track_id = 1
    track = TrackStore.load_track(level=level, index=track_id)

    num_repeats = 500
    race_times = []
    for i in range(num_repeats):

        print('\rRepeat {}/{}'.format(i + 1, num_repeats), end='')

        set_seed(i)
        driver = LearnerDriver(driver_name)
        _, race_time, finished = race(driver, track=track, plot=False, max_number_of_steps=150)

        if finished:
            race_times.append(race_time)
    print()

    print('Finished races {}'.format(len(race_times)))
    print('Best race time: {}'.format(min(race_times)))
    print('Worst race time: {}'.format(max(race_times)))
    print('Average race time: {}'.format(np.mean(race_times)))
    print('Race time standard deviation: {}'.format(np.std(race_times)))
