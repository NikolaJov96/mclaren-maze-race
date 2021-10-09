import os
import sys

from imports import *


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Display rookie tracks
    level = Level.Rookie
    for track_id in range(TrackStore.get_number_of_tracks(level=level)):
        track = TrackStore.load_track(level=level, index=track_id + 1)
        track.plot_track()
        plt.savefig(os.path.join(path, '{}.png'.format(track_id + 1)))
        plt.close()
