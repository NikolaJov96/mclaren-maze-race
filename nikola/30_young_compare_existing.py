import os
import sys

from imports import *
from drivers.youngdriver import *


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    num_championship_repeats = 50

    for shuffle_tracks in [True, False]:
        # Run championships with and without track shuffling

        drivers = [
            YoungDriver('YoungDriver', speed_rounding=40, max_distance=8, learning_rate=0.25),
            YoungDriver1('YoungDriver1', speed_rounding=40, max_distance=8, learning_rate=0.25),
            YoungDriverNearestState('NearestState', speed_rounding=40, max_distance=8, learning_rate=0.25)
        ]

        championship = Championship(drivers, Level.Young, shuffle_tracks=shuffle_tracks)
        championship_results, race_results, race_times = \
            championship.run_championship(num_repeats=num_championship_repeats)

        plot_multiple_championship_results(championship_results)
        plt.savefig(os.path.join(path, 'existing_young_drivers_shuffle_{}.png'.format(shuffle_tracks)))
