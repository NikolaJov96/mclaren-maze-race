import os
import sys

from imports import *


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Display number of left and right turns for each track
    total_left_turns = 0
    total_right_turns = 0
    for i, track in enumerate(TrackStore.load_all_tracks(level=Level.Young)):
        correct_turns = list(track.correct_turns.values())
        left_turns = correct_turns.count(Action.TurnLeft)
        total_left_turns += left_turns
        right_turns = correct_turns.count(Action.TurnRight)
        total_right_turns += right_turns
        print('Track: {:2}  Left turns {:2}  Right turns: {:2}  Total left turns {:3}  Total right turns: {:3}'.format(
            i + 1, left_turns, right_turns, total_left_turns, total_right_turns))

    # Display left and right turn positions per track
    per_track_path = os.path.join(path, 'per_track')
    if not os.path.exists(per_track_path):
        os.mkdir(per_track_path)
    for i, track in enumerate(TrackStore.load_all_tracks(level=Level.Young)):

        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])
        ax.invert_yaxis()

        for j, (position, turn_action) in enumerate(track.correct_turns.items()):
            if Action.TurnLeft == turn_action:
                ax.plot(position.column, position.row, 'r+')
            else:
                ax.plot(position.column, position.row, 'g+')
            ax.text(position.column, position.row, str(j + 1))

        plt.savefig(os.path.join(per_track_path, '{}.png'.format(i + 1)))
        plt.close(fig)

    # Display left and right turn positions combined
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.invert_yaxis()

    for track in TrackStore.load_all_tracks(level=Level.Young):
        for position, turn_action in track.correct_turns.items():
            if Action.TurnLeft == turn_action:
                ax.plot(position.column, position.row, 'r+')
            else:
                ax.plot(position.column, position.row, 'g+')

    plt.grid()
    plt.savefig(os.path.join(path, 'all_left_right_turns.png'))
    plt.close(fig)

    # Display all turns including only left and only right turns
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.invert_yaxis()

    for track in TrackStore.load_all_tracks(level=Level.Young):

        position = track.start_position
        heading = track.start_heading

        while not track.is_finished(position):

            state = track.get_state_for_position(position, heading)

            if state.distance_ahead > 0:
                position, _, _ = track.get_new_position(position, 1, heading)
            else:
                # Assert not a dead end
                assert state.distance_left != 0 or state.distance_right != 0

                # Decide on the correct turn
                correct_action = None
                if state.distance_left == 0:
                    correct_action = Action.TurnRight
                    ax.plot(position.column, position.row, 'g+')
                elif state.distance_right == 0:
                    correct_action = Action.TurnLeft
                    ax.plot(position.column, position.row, 'r+')
                else:
                    assert position in track.correct_turns.keys()
                    correct_action = track.correct_turns[position]

                assert correct_action in [Action.TurnLeft, Action.TurnRight]
                if correct_action == Action.TurnLeft:
                    heading = heading.get_left_heading()
                    ax.plot(position.column, position.row, 'r+')
                elif correct_action == Action.TurnRight:
                    heading = heading.get_right_heading()
                    ax.plot(position.column, position.row, 'g+')

    plt.savefig(os.path.join(path, 'all_turns.png'))
    plt.close(fig)

    # Display absolute directions of turns
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.invert_yaxis()

    for track in TrackStore.load_all_tracks(level=Level.Young):

        position = track.start_position
        heading = track.start_heading

        while not track.is_finished(position):

            state = track.get_state_for_position(position, heading)

            if state.distance_ahead > 0:
                position, _, _ = track.get_new_position(position, 1, heading)
            else:
                # Assert not a dead end
                assert state.distance_left != 0 or state.distance_right != 0

                # Decide on the correct turn
                correct_action = None
                if state.distance_left == 0:
                    correct_action = Action.TurnRight
                elif state.distance_right == 0:
                    correct_action = Action.TurnLeft
                else:
                    assert position in track.correct_turns.keys()
                    correct_action = track.correct_turns[position]

                assert correct_action in [Action.TurnLeft, Action.TurnRight]
                if correct_action == Action.TurnLeft:
                    heading = heading.get_left_heading()

                elif correct_action == Action.TurnRight:
                    heading = heading.get_right_heading()

                if heading == Heading(1, 0):
                    ax.plot(position.column, position.row, 'r+')
                elif heading == Heading(-1, 0):
                    ax.plot(position.column, position.row, 'g+')
                elif heading == Heading(0, 1):
                    ax.plot(position.column, position.row, 'b+')
                elif heading == Heading(0, -1):
                    ax.plot(position.column, position.row, 'y+')
                else:
                    assert False

    plt.savefig(os.path.join(path, 'absolute_turns.png'))
    plt.close(fig)
