from random import randint, sample

from imports import *


if __name__ == '__main__':

    # Number of tracks per driver level
    valid_levels = set()
    for level in Level:
        try:
            print('Level {}, num of tracks: {}'.format(level, TrackStore.get_number_of_tracks(level=level)))
            valid_levels.add(level)
        except:
            print('Error: No tracks for level {}'.format(level))

    # Get random level and track
    level = sample(valid_levels, 1)[0]
    track_id = randint(1, TrackStore.get_number_of_tracks(level=level))
    print('Level: {} Track id: {}'.format(level, track_id))

    # Display the first track
    track = TrackStore.load_track(level=level, index=track_id)
    track.plot_track()
    plt.show()
