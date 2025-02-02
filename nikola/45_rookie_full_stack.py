import os
import sys

from imports import *
from drivers.driver import Driver
from drivers.my_rookie_driver import MyDriver as Submission
from nikola.car_dynamics_tracker import RookieCarDynamicsTracker
from nikola.race_logger import RaceLogger
from nikola.safety_car_tracker import SafetyCarTracker
from nikola.straight_simulator import StraightSimulator
from nikola.turn_chooser import MultipleClosestTurnChooser
from nikola.turn_tracker import RealtimeTurnTracker


class RookieDriverLooged(RookieDriver):
    """
    Taken from 41_rookie_turn_choosing.py
    """

    def __init__(self, name, race_logger_dir, random_action_probability=0.5, random_action_decay=0.99,
            min_random_action_probability=0, *args, **kwargs):
        super().__init__(name, random_action_probability=random_action_probability,
            random_action_decay=random_action_decay,
            min_random_action_probability=min_random_action_probability, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.race_id = 0
        self.race_logger = None

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        return super().prepare_for_race()

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result)


class MyRookieDriver(Driver):

    def __init__(self, name: str, race_logger_dir: str):
        super().__init__(name)
        self.race_logger_dir = race_logger_dir

        self.race_logger = None
        self.turn_tracker = RealtimeTurnTracker()
        self.turn_chooser = MultipleClosestTurnChooser(3, True)
        self.safety_car_tracker = SafetyCarTracker()
        self.car_dynamics_tracker = RookieCarDynamicsTracker()

        self.race_id = 0
        self.potential_stop = False
        self.drs_count = 0

    def prepare_for_race(self):
        # Next race id
        self.race_id += 1
        # Initialize this race logger
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        # Start new race in the turn tracker
        self.turn_tracker.new_race()
        # Start new race in the safety car tracker
        self.safety_car_tracker.new_race()
        # No way for an initial straight to be a dead end
        self.potential_stop = False

    def make_a_move(self, car_state: CarState, track_state: TrackState) -> Action:
        # Immediately update the safety car tracker and get the safety car speed if active
        safety_car_target_speed = self.safety_car_tracker.new_track_state(track_state)

        # Easy choice if distance_ahead is 0
        if track_state.distance_ahead == 0:
            if track_state.distance_left == 0 and track_state.distance_right == 0 and car_state.speed > 0:
                # Have to stop (in case of a crash we will be back in this state with the speed=0)
                return self.car_dynamics_tracker.choose_best_action(car_state.speed, 0, car_state.drs_active)
            else:
                # Have to turn (in case of a spin we will be back in this state with the speed=0)
                if track_state.distance_left > 0 and track_state.distance_right > 0:
                    return self.turn_chooser.t_junction_choose_turn(
                        self.race_id,
                        track_state.position,
                        self.turn_tracker.correct_turns,
                        [track_state.distance_left, track_state.distance_right])
                else:
                    return Action.TurnLeft if track_state.distance_left > 0 else Action.TurnRight

        # Get the max safe target speed
        target_speeds = StraightSimulator.get_target_speeds(
            self.car_dynamics_tracker,
            track_state.distance_ahead,
            self.potential_stop)
        target_speed = target_speeds[-1]

        # If safety car is active, make sure the target speed is under its
        if track_state.safety_car_active:
            target_speed = min(target_speed, safety_car_target_speed)

        # Chose the optimal action to reach the target speed
        action = self.car_dynamics_tracker.choose_best_action(car_state.speed, target_speed, car_state.drs_active)
        # Consider reasons to explore instead of to exploit
        # Lower the speed, experiment closer to the turn, rough heuristic
        if not track_state.safety_car_active and car_state.speed / 350.0 * 5.0 / track_state.distance_ahead < 1.0:
            # Find the action with the fardest data point and pick that one
            fardest_action, fardest_action_distance = \
                self.car_dynamics_tracker.get_fardest_action(car_state.speed, car_state.drs_active)
            if fardest_action_distance > 30:
                # Prevent from breaking the DRS too early
                if not (car_state.drs_active and fardest_action in [Action.LightBrake, Action.HeavyBrake] and track_state.distance_ahead > 7):
                    action = fardest_action
                else:
                    action, _ = self.car_dynamics_tracker.get_fardest_action(
                        car_state.speed, car_state.drs_active, [Action.LightThrottle, Action.FullThrottle])
        elif track_state.safety_car_active and car_state.speed > safety_car_target_speed:
            # If safety car is active and speed is exceeded make sure we brake
            if action not in [Action.HeavyBrake, Action.LightBrake]:
                action = action, _ = self.car_dynamics_tracker.get_fardest_action(
                    car_state.speed, car_state.drs_active, [Action.HeavyBrake, Action.LightBrake])

        # Determine should we start DRS
        if track_state.drs_available and not car_state.drs_active:
            # Never do it if distance ahead is less then 6 or safety car is active
            if track_state.distance_ahead > 5 and not track_state.safety_car_active:
                # Simulate straights
                time_no_drs, _, _ = StraightSimulator.simulate_straight(
                    self.car_dynamics_tracker,
                    car_state.speed,
                    track_state.distance_ahead,
                    self.potential_stop,
                    drs_active=False,
                    safety_car_speed=safety_car_target_speed if track_state.safety_car_active else 0)
                time_drs, targets_broken_drs, _ = StraightSimulator.simulate_straight(
                    self.car_dynamics_tracker,
                    car_state.speed,
                    track_state.distance_ahead - 1,
                    self.potential_stop,
                    drs_active=True,
                    safety_car_speed=safety_car_target_speed if track_state.safety_car_active else 0)
                time_drs = (1 / (car_state.speed + 1)) + time_drs
                # Make the decision
                if self.drs_count < 6 or (time_drs < time_no_drs and not targets_broken_drs):
                    self.drs_count += 1
                    action = Action.OpenDRS

        return action

    def update_with_action_results(self,
            previous_car_state: CarState, previous_track_state: TrackState, action: Action,
            new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        # Log using the race logger
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        # Update the turn tracker
        self.turn_tracker.new_track_state(previous_track_state, is_final=False)
        if result.finished:
            self.turn_tracker.new_track_state(new_track_state, is_final=True)
        # Update the safety car tracker
        self.safety_car_tracker.action_result(new_car_state, result)
        # Update car dynamics
        self.car_dynamics_tracker.add_data_point(action, previous_car_state, new_car_state, result)
        # Check the possibility of the next straight being a dead end
        if previous_track_state.distance_ahead == 0 and new_track_state.distance_ahead > 0:
            if previous_track_state.distance_left > 0 and previous_track_state.distance_right > 0:
                self.potential_stop = True
            else:
                self.potential_stop = False

    def update_after_race(self, correct_turns: Dict[Position, Action]):
        # Check turn tracker validity
        self.turn_tracker.update_after_race(correct_turns)


def main():
    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run one championship with the original rookie
    print('Running championship')
    drivers = [
        RookieDriverLooged('RD', os.path.join(path, 'original_rookie')),
        MyRookieDriver('MyRookieDriver', os.path.join(path, 'my_rookie')),
        Submission()
    ]
    championship = Championship(drivers, Level.Rookie, shuffle_tracks=True, verbose=True)
    championship_results, finishing_positions, all_race_times = championship.run_championship(num_repeats=1)
    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'championship.png'))
    plt.close()

    with open(os.path.join(path, 'championship.txt'), 'w') as out_file:
        out_file.write('all_race_times\n')
        out_file.write(str(all_race_times))
        out_file.write('\n')

        for driver in all_race_times:
            out_file.write(driver)
            out_file.write('\n')
            won_races = finishing_positions[driver] == 1
            if sum([sum(won_sub_races) for won_sub_races in won_races]) == 0:
                continue

            out_file.write('won_races\n')
            out_file.write(str(won_races))
            out_file.write('\n')
            for other_driver in all_race_times:
                if other_driver == driver:
                    continue
                won_time_diff = all_race_times[other_driver][won_races] - all_race_times[driver][won_races]
                out_file.write('won_time_diff\n')
                out_file.write(str(won_time_diff))
                out_file.write('\n')
                out_file.write(str(sum(won_time_diff) / len(won_time_diff)))
                out_file.write('\n')

            out_file.write('\n')

    # Run repeated championships with the original rookie
    print('Running championships')
    drivers = [
        RookieDriverLooged('RD', ''),
        MyRookieDriver('MyRookieDriver', ''),
        Submission()
    ]
    championship = Championship(drivers, Level.Rookie, shuffle_tracks=True, verbose=True)
    championship_results, finishing_positions, all_race_times = championship.run_championship(num_repeats=100)
    plot_multiple_championship_results(championship_results)
    plt.savefig(os.path.join(path, 'repeated_championship.png'))
    plt.close()


if __name__ == '__main__':
    main()
