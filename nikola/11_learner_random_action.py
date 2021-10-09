import numpy as np

from drivers.learnerdriver import LearnerDriver
from imports import *


if __name__ == '__main__':

    driver_name = 'Learner Nikola'

    level = Level.Learner
    track_id = 1
    track = TrackStore.load_track(level=level, index=track_id)

    num_races = 200
    num_repeats = 10
    params = [
        {
            'rnd_probability': 0,
            'rnd_decay': 1
        },
        {
            'rnd_probability': 0.5,
            'rnd_decay': 0.999
        }
    ]
    plot_legend = [list(param.values()) for param in params]
    race_times = np.zeros((num_races, num_repeats, len(params)))

    fig = plt.figure(figsize=(9, 6))
    fig.add_axes([0.08, 0.08, 0.92, 0.92])
    lines = []

    for pi in range(len(params)):

        np.random.seed(0)

        for i in range(num_repeats):

            driver = LearnerDriver(
                driver_name,
                random_action_probability=params[pi]['rnd_probability'],
                random_action_decay=params[pi]['rnd_decay'],
                min_random_action_probability=0,
                discount_factor=0.9)

            for n in range(num_races):
                _, race_times[n, i, pi], _ = race(driver, track=track, plot=False, max_number_of_steps=1000)

            print('\rCompleted repeat {}/{} for parameter setting {}/{}'.format(
                i + 1, num_repeats, pi + 1, len(params)), end='')
        print()

        # Plot the results
        lines += plt.plot(range(1, num_races + 1), np.mean(race_times[:, :, pi], axis=1))
        plt.fill_between(range(1, num_races + 1), np.min(race_times[:, :, pi], axis=1),
            np.max(race_times[:, :, pi], axis=1), color=lines[-1].get_color(), alpha=0.1)

        print('Average driver time between {} repeats, after {} races, for parameters {}: {}'.format(
            num_repeats, num_races, plot_legend[pi], np.mean(race_times[-1, :, pi])))

    plt.ylabel('Race Times', fontsize=16)
    plt.xlabel('Race Number', fontsize=16)
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(lines, plot_legend, fontsize=16)
    plt.show()
