import os
from random import choice
import sys

from imports import *
from drivers.youngdriver import YoungDriver


class ShorterTurnYoungDriver(YoungDriver):

    def __init__(self, name,
        min_known_turn_distance,
        min_turn_length_difference,
        use_area_edge_detection,
        *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.min_known_turn_distance = min_known_turn_distance
        self.min_turn_length_difference = min_turn_length_difference
        self.use_area_edge_detection = use_area_edge_detection

    def _choose_turn_direction(self, track_state: TrackState):
        # Check if we need to make a decision about which way to turn
        if track_state.distance_left > 0 and track_state.distance_right > 0:  # both options available, need to decide
            if len(self.correct_turns) > 0:
                # Find the closest turn we have seen previously and turn in the same direction
                distances = np.array([track_state.position.distance_to(turn_position)
                                      for turn_position in self.correct_turns])
                i_closest = np.argmin(distances)

                if self._ignore_closest(distances):
                    if abs(track_state.distance_left - track_state.distance_right) >= self.min_turn_length_difference:
                        return Action.TurnLeft if track_state.distance_left < track_state.distance_right else Action.TurnRight
                    else:
                        return choice([Action.TurnLeft, Action.TurnRight])
                else:
                    return list(self.correct_turns.values())[i_closest]

            else:  # First race, no data yet so choose randomly
                return Action.TurnLeft if track_state.distance_left < track_state.distance_right else Action.TurnRight

        elif track_state.distance_left > 0:  # only left turn
            return Action.TurnLeft

        else:
            return Action.TurnRight  # only right or dead-end

    def _ignore_closest(self, distances):

        closest_ids = np.argsort(distances)

        if not self.use_area_edge_detection:
            i_closest = np.argmin(distances)
            return distances[i_closest] > self.min_known_turn_distance

        actions = list(self.correct_turns.values())

        right_found = False
        left_found = False

        i = 0
        while distances[closest_ids[i]] < self.min_known_turn_distance:
            if actions[closest_ids[i]] == Action.TurnLeft:
                left_found = True
            else:
                right_found = True
            i += 1

        return right_found == left_found


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Fixed young driver parameters
    speed_rounding = 40
    max_distance = 8
    learning_rate = 0.25

    # Chanpionship parameters
    num_repeats = 200

    # Compare drivers with different min distances
    for min_known_turn_distance in [2.0, 3.0, 4.0, 5.0, 6.0]:

        drivers = [
            YoungDriver('Original', speed_rounding=speed_rounding, max_distance=max_distance, learning_rate=learning_rate),
            ShorterTurnYoungDriver(
                'min_dist_{}'.format(int(min_known_turn_distance)),
                min_known_turn_distance=min_known_turn_distance,
                min_turn_length_difference=0,
                use_area_edge_detection=False,
                speed_rounding=speed_rounding,
                max_distance=max_distance,
                learning_rate=learning_rate)
        ]

        print('min_known_turn_distance {} championship'.format(min_known_turn_distance))
        championship = Championship(drivers, Level.Young, shuffle_tracks=True, verbose=True)
        championship_results, race_results, race_times = championship.run_championship(num_repeats=num_repeats)

        plot_multiple_championship_results(championship_results)
        plt.savefig(os.path.join(path, '0_min_known_turn_distance_{}.png'.format(int(min_known_turn_distance))))
