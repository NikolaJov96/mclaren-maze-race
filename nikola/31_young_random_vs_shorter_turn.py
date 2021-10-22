import os
from random import choice
import os
import sys

from imports import *
from drivers.youngdriver import YoungDriver
from nikola.race_logger import RaceLogger
from nikola.turn_logger import TurnLogger


class ShorterTurnYoungDriver(YoungDriver):

    def __init__(self, name,
        min_known_turn_distance,
        min_turn_length_difference,
        use_area_edge_detection,
        turn_logger_dir='',
        race_logger_dir='',
        *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.min_known_turn_distance = min_known_turn_distance
        self.min_turn_length_difference = min_turn_length_difference
        self.use_area_edge_detection = use_area_edge_detection
        self.race_logger_dir = race_logger_dir
        self.turn_logger = TurnLogger(turn_logger_dir) if turn_logger_dir != '' else None
        self.race_id = 0
        self.race_logger = None

    def prepare_for_race(self):
        if self.turn_logger is not None:
            self.turn_logger.new_race()
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        return super().prepare_for_race()

    def _choose_turn_direction(self, track_state: TrackState):
        action = None
        # Check if we need to make a decision about which way to turn
        if track_state.distance_left > 0 and track_state.distance_right > 0:  # both options available, need to decide
            if len(self.correct_turns) > 0:
                # Find the closest turn we have seen previously and turn in the same direction
                distances = np.array([track_state.position.distance_to(turn_position)
                                      for turn_position in self.correct_turns])
                # Multiple with the same distance???
                i_closest = np.argmin(distances)

                if self._ignore_closest(distances):
                    if abs(track_state.distance_left - track_state.distance_right) >= self.min_turn_length_difference:
                        action =  Action.TurnLeft if track_state.distance_left < track_state.distance_right else Action.TurnRight
                    else:
                        # action =  choice([Action.TurnLeft, Action.TurnRight])
                        action = list(self.correct_turns.values())[i_closest]
                else:
                    action = list(self.correct_turns.values())[i_closest]

            else:  # First race, no data yet so choose randomly
                action = Action.TurnLeft if track_state.distance_left < track_state.distance_right else Action.TurnRight

            if self.turn_logger is not None:
                self.turn_logger.log_turn(self.correct_turns, track_state.position, action)

        elif track_state.distance_left > 0:  # only left turn
            action = Action.TurnLeft
        else:
            action = Action.TurnRight  # only right or dead-end

        return action

    def _ignore_closest(self, distances):

        closest_ids = np.argsort(distances)

        if not self.use_area_edge_detection:
            # If not using edge detection, just check how far is the closest distance
            i_closest = np.argmin(distances)
            return distances[i_closest] > self.min_known_turn_distance

        actions = list(self.correct_turns.values())

        inf = float('Inf')
        closest_right = inf
        closest_left = inf

        i = 0
        while distances[closest_ids[i]] < self.min_known_turn_distance:
            if closest_left == inf and actions[closest_ids[i]] == Action.TurnLeft:
                closest_left = distances[closest_ids[i]]
            elif closest_right == inf and actions[closest_ids[i]] == Action.TurnRight:
                closest_right = distances[closest_ids[i]]
            i += 1

        if closest_left == inf and closest_right == inf:
            return True
        elif closest_left == inf or closest_right == inf or min(closest_left, closest_right) == 0:
            return False
        else:
            if min(closest_left, closest_right) / max(closest_left, closest_right) > 0.6:
                return True
            else:
                return False

    def update_with_action_results(self,
            previous_car_state: CarState, previous_track_state: TrackState, action: Action,
            new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        return super().update_with_action_results(previous_car_state, previous_track_state, action, new_car_state, new_track_state, result)


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Fixed young driver parameters
    speed_rounding = 40
    max_distance = 8
    learning_rate = 0.25

    # Run one season with turn logger
    Season(Level.Young).race(ShorterTurnYoungDriver(
        'YD',
        min_known_turn_distance=50.0,
        min_turn_length_difference=0,
        use_area_edge_detection=False,
        turn_logger_dir=os.path.join(path, 'turn_logger'),
        race_logger_dir=os.path.join(path, 'race_logger'),
        speed_rounding=speed_rounding,
        max_distance=max_distance,
        learning_rate=learning_rate))

    # Further experiments have no use because Young driver is just too bad











    # Chanpionship parameters
    num_repeats = 500

    # Compare drivers with different min distances
    for min_known_turn_distance in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        for use_area_edge_detection in [True, False]:

            drivers = [
                YoungDriver('Original', speed_rounding=speed_rounding, max_distance=max_distance, learning_rate=learning_rate),
                ShorterTurnYoungDriver(
                    'min_dist_{}'.format(int(min_known_turn_distance)),
                    min_known_turn_distance=min_known_turn_distance,
                    min_turn_length_difference=0,
                    use_area_edge_detection=use_area_edge_detection,
                    speed_rounding=speed_rounding,
                    max_distance=max_distance,
                    learning_rate=learning_rate)
            ]

            print('min_known_turn_distance {} use_area_edge_detection {} championship'.format(
                min_known_turn_distance,
                use_area_edge_detection))
            championship = Championship(drivers, Level.Young, shuffle_tracks=True, verbose=True)
            championship_results, race_results, race_times = championship.run_championship(num_repeats=num_repeats)

            plot_multiple_championship_results(championship_results)
            plt.savefig(os.path.join(path, '0_min_known_turn_distance_{}_use_area_edge_detection_{}.png'.format(
                int(min_known_turn_distance), use_area_edge_detection)))
