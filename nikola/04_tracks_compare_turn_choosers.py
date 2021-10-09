import sys
from random import shuffle

from imports import *


class TurnChooser:

    def __init__(self):
        self.choices_per_track = [[]]

    def next_track(self):
        self.choices_per_track.append([])

    def store_choice(self, choice: bool):
        self.choices_per_track[-1].append(choice)

    def get_choice_ratio_per_track(self):
        if len(self.choices_per_track[-1]) == 0:
            self.choices_per_track = self.choices_per_track[:-1]
        return np.array(list(map(lambda t: sum(t) / len(t), self.choices_per_track)))

    def get_choice_ratio_per_track_cumulative(self):
        if len(self.choices_per_track[-1]) == 0:
            self.choices_per_track = self.choices_per_track[:-1]
        sum_per_track = list(map(lambda t: sum(t), self.choices_per_track))
        len_per_track = list(map(lambda t: len(t), self.choices_per_track))
        return np.array([sum(sum_per_track[:i + 1]) / sum(len_per_track[:i + 1]) for i in range(len(sum_per_track))])

    def next_turn(self, position: Position, correct_action: Action):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class RandomTurnChooser(TurnChooser):

    def next_turn(self, position: Position, correct_action: Action):
        action = driver_rng().choice([Action.TurnLeft, Action.TurnRight])

        self.store_choice(action == correct_action)

    def name(self):
        return "Random"


class SingleActionTurnChooser(TurnChooser):

    def __init__(self, action: Action, action_name):
        super().__init__()

        self.action = action
        self.action_name = action_name

    def next_turn(self, position: Position, correct_action: Action):
        self.store_choice(self.action == correct_action)

    def name(self):
        return "Only " + self.action_name


class DefaultTurnChooser(TurnChooser):

    def __init__(self):
        super().__init__()

        self.correct_turns = {}
        self.current_track_correct_turns = []

    def next_track(self):
        super().next_track()

        for position, action in self.current_track_correct_turns:
            self.correct_turns[position] = action

    def next_turn(self, position: Position, correct_action: Action):
        action = driver_rng().choice([Action.TurnLeft, Action.TurnRight])
        if len(self.correct_turns) > 0:
            distances = np.array([position.distance_to(turn_position) for turn_position in self.correct_turns])
            i_closest = np.argmin(distances)
            action = list(self.correct_turns.values())[i_closest]

        self.current_track_correct_turns.append((position, correct_action))

        self.store_choice(action == correct_action)

    def name(self):
        return "Default"


class RealTimeDefaultTurnChooser(TurnChooser):

    def __init__(self):
        super().__init__()

        self.correct_turns = {}

    def next_turn(self, position: Position, correct_action: Action):
        action = driver_rng().choice([Action.TurnLeft, Action.TurnRight])
        if len(self.correct_turns) > 0:
            distances = np.array([position.distance_to(turn_position) for turn_position in self.correct_turns])
            i_closest = np.argmin(distances)
            action = list(self.correct_turns.values())[i_closest]

        self.correct_turns[position] = correct_action

        self.store_choice(action == correct_action)

    def name(self):
        return "RealTimeDefault"


class MultipleClosestPositionsTurnChooser(TurnChooser):

    def __init__(self, num_positions):
        super().__init__()

        self.correct_turns = {}
        self.num_positions = num_positions

    def next_turn(self, position: Position, correct_action: Action):
        action = driver_rng().choice([Action.TurnLeft, Action.TurnRight])
        if len(self.correct_turns) > 0:
            distances = np.array([position.distance_to(turn_position) for turn_position in self.correct_turns])
            closest_ids = np.argsort(distances)[:min(self.num_positions, len(distances))]
            correct_actions = list(self.correct_turns.values())

            if distances[closest_ids[0]] == 0:
                return correct_actions[closest_ids[0]]

            left_ids = [i for i in closest_ids if correct_actions[i] == Action.TurnLeft]
            right_ids = [i for i in closest_ids if correct_actions[i] == Action.TurnRight]
            left_weight = sum(map(lambda d: 1.0 / d, distances[left_ids]))
            right_weight = sum(map(lambda d: 1.0 / d, distances[right_ids]))
            action = Action.TurnLeft if left_weight > right_weight else Action.TurnRight

        self.correct_turns[position] = correct_action

        self.store_choice(action == correct_action)

    def name(self):
        return "{} closest positions".format(self.num_positions)


class BalanceLeftRightTurnChooser(TurnChooser):

    def __init__(self, max_distance):
        super().__init__()

        self.correct_turns = {}
        self.turn_left_count = 0
        self.turn_right_count = 0
        self.max_distance = max_distance

    def next_turn(self, position: Position, correct_action: Action):
        action = driver_rng().choice([Action.TurnLeft, Action.TurnRight])
        if len(self.correct_turns) > 0:
            distances = np.array([position.distance_to(turn_position) for turn_position in self.correct_turns])
            i_closest = np.argmin(distances)
            if distances[i_closest] < self.max_distance:
                action = list(self.correct_turns.values())[i_closest]
            else:
                action = Action.TurnLeft if self.turn_left_count < self.turn_right_count else Action.TurnRight

        self.correct_turns[position] = correct_action
        if correct_action == Action.TurnLeft:
            self.turn_left_count += 1
        else:
            self.turn_right_count += 1

        self.store_choice(action == correct_action)

    def name(self):
        return "Left-right balance ({})".format(self.max_distance)


class FavorSameDirectionTurnChooser(TurnChooser):

    def __init__(self, max_distance):
        super().__init__()

        self.correct_turns = {}
        self.previous_correct_action = driver_rng().choice([Action.TurnLeft, Action.TurnRight])
        self.max_distance = max_distance

    def next_turn(self, position: Position, correct_action: Action):
        action = driver_rng().choice([Action.TurnLeft, Action.TurnRight])
        if len(self.correct_turns) > 0:
            distances = np.array([position.distance_to(turn_position) for turn_position in self.correct_turns])
            i_closest = np.argmin(distances)
            if distances[i_closest] < self.max_distance:
                action = list(self.correct_turns.values())[i_closest]
            else:
                action = self.previous_correct_action

        self.correct_turns[position] = correct_action
        self.previous_correct_action = correct_action

        self.store_choice(action == correct_action)

    def name(self):
        return "Favor same action ({})".format(self.max_distance)


def get_turn_choosers():
    return [
        RandomTurnChooser(),
        SingleActionTurnChooser(Action.TurnLeft, 'TurnLeft'),
        SingleActionTurnChooser(Action.TurnRight, 'TurnRight'),
        DefaultTurnChooser(),
        RealTimeDefaultTurnChooser(),
        MultipleClosestPositionsTurnChooser(3),
        BalanceLeftRightTurnChooser(8.0),
        FavorSameDirectionTurnChooser(8.0)
    ]


def run_test(all_tracks):

    turn_choosers = get_turn_choosers()

    for track in all_tracks:

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

                    # Step all turn choosers
                    for turn_chooser in turn_choosers:
                        turn_chooser.next_turn(position, correct_action)

                # Update the heading using the correct action
                assert correct_action in [Action.TurnLeft, Action.TurnRight]
                if correct_action == Action.TurnLeft:
                    heading = heading.get_left_heading()
                elif correct_action == Action.TurnRight:
                    heading = heading.get_right_heading()

        # Switch to the next track for all turn choosers
        for turn_chooser in turn_choosers:
            turn_chooser.next_track()

    return (
        np.array(list(map(lambda tc: tc.get_choice_ratio_per_track(), turn_choosers))),
        np.array(list(map(lambda tc: tc.get_choice_ratio_per_track_cumulative(), turn_choosers)))
    )


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    num_test_runs = 200

    all_tracks = TrackStore.load_all_tracks(level=Level.Young)
    turn_choosers = get_turn_choosers()

    per_track_data = []
    per_track_cumulative_data = []

    for i in range(num_test_runs):
        shuffle(all_tracks)
        per_track, per_track_cumulative = run_test(all_tracks)
        per_track_data.append(per_track)
        per_track_cumulative_data.append(per_track_cumulative)

    per_track_average = np.mean(per_track_data, axis=0)
    per_track_cumulative_average = np.mean(per_track_cumulative_data, axis=0)

    # Plot results
    num_tracks = len(all_tracks)
    tracks = [i + 1 for i in range(num_tracks)]
    ylim = (-0.05, 0.05)
    y_ticks = [i / 10.0 for i in range(11)]

    # Plot per track results
    fig, ax = plt.subplots()
    ax.set_title('Turn hit ratio per track')
    for i, per_track in enumerate(per_track_average):
        ax.plot(tracks, per_track, label=turn_choosers[i].name())
    ax.set_xticks(tracks)
    ax.set_ylim(ylim)
    ax.set_yticks(y_ticks)
    ax.legend()
    ax.grid()
    plt.savefig(os.path.join(path, 'per_track.png'))
    plt.close(fig)

    # Plot per track cumulative results
    fig, ax = plt.subplots()
    ax.set_title('Cumulative turn hit ratio')
    for i, per_track_cumulative in enumerate(per_track_cumulative_average):
        ax.plot(tracks, per_track_cumulative, label=turn_choosers[i].name())
    ax.set_xticks(tracks)
    ax.set_ylim(ylim)
    ax.set_yticks(y_ticks)
    ax.legend()
    ax.grid()
    plt.savefig(os.path.join(path, 'per_track_cumulative.png'))
    plt.close(fig)
