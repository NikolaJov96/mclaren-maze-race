import scipy

from drivers.driver import Driver
from imports import *

from nikola.race_logger import RaceLogger


class RealtimeTurnTracker:
    """
    Tracks correct turns in real time, without waiting for the end of the race
    """

    class RealtimeTrackerState:
        """
        Base class for turn tracker states
        """

        def __init__(self, turn_tracker):
            self.turn_tracker = turn_tracker

        def new_track_state(self, track_state, is_final):
            raise NotImplementedError


    class DefaultRealtimeTrackerState(RealtimeTrackerState):
        """
        Turn tracker state when no turns are being examined
        """

        def new_track_state(self, track_state, is_final):
            if is_final:
                return self

            if track_state.distance_ahead == 0 and \
                    track_state.distance_left > 0 and \
                    track_state.distance_right > 0:
                return RealtimeTurnTracker.TurnJustTakenRealtimeTrackerState(self.turn_tracker)
            else:
                return self


    class TurnJustTakenRealtimeTrackerState(RealtimeTrackerState):
        """
        Turn tracker state when a turn has just been taken
        Used to determine which action (TurnLeft or TurnRight) was used
        Only active during one state
        """

        def new_track_state(self, track_state, is_final):

            def heading_from_pos(pos1, pos2):
                return Heading(pos2.row - pos1.row, pos2.column - pos1.column)

            current_heading = heading_from_pos(
                self.turn_tracker.current_position(),
                self.turn_tracker.last_position())
            last_heading = heading_from_pos(
                self.turn_tracker.last_position(),
                self.turn_tracker.before_last_position())

            if last_heading.get_left_heading() == current_heading:
                taken_action = Action.TurnLeft
            elif last_heading.get_right_heading() == current_heading:
                taken_action = Action.TurnRight
            else:
                raise ValueError

            return RealtimeTurnTracker.CheckingTakenTurnRealtimeTrackerState(
                self.turn_tracker, taken_action).new_track_state(track_state, is_final)


    class CheckingTakenTurnRealtimeTrackerState(RealtimeTrackerState):
        """
        Turn tracker state when waiting to reach the end of a straight
        to determine if the previous turn was correct
        """

        def __init__(self, turn_tracker, taken_action):
            super().__init__(turn_tracker)
            self.turn_position = self.turn_tracker.last_position()
            self.taken_action = taken_action

        def new_track_state(self, track_state, is_final):
            if track_state.distance_ahead > 0:
                return self

            if track_state.distance_left > 0 or track_state.distance_right > 0 or is_final:
                # No dead end, correct turn taken
                self.turn_tracker.put_correct_turn(self.turn_position, self.taken_action)
            else:
                # Dead end, wrong turn taken
                action = Action.TurnLeft if self.taken_action == Action.TurnRight else Action.TurnRight
                self.turn_tracker.put_correct_turn(self.turn_position, action)

            # Immediately check if new default state is finished
            return RealtimeTurnTracker.DefaultRealtimeTrackerState(self.turn_tracker).\
                new_track_state(track_state, False)

    def __init__(self):
        self.correct_turns = {}
        self._current_state = None
        self._recent_car_positions = []
        self._this_race_correct_turns = {}

    def new_race(self):
        # Reinitialize current race parameters
        self._current_state = RealtimeTurnTracker.DefaultRealtimeTrackerState(self)
        self._recent_car_positions = []
        self._this_race_correct_turns = {}

    def new_track_state(self, track_state, is_final=False):
        # Skip in case of the same position as was the last
        if len(self._recent_car_positions) > 0 and self._recent_car_positions[-1] == track_state.position:
            return

        # Record new position
        self._recent_car_positions.append(track_state.position)
        if len(self._recent_car_positions) > 3:
            self._recent_car_positions = self._recent_car_positions[-3:]

        # Update the tracker state
        self._current_state = self._current_state.new_track_state(track_state, is_final)
        assert self._current_state is not None

    def current_position(self):
        return self._recent_car_positions[-1]

    def last_position(self):
        return self._recent_car_positions[-2]

    def before_last_position(self):
        return self._recent_car_positions[-3]

    def put_correct_turn(self, position, action):
        self.correct_turns[position] = action
        self._this_race_correct_turns[position] = action

    def update_after_race(self, correct_turns: Dict[Position, Action]):
        assert correct_turns == self._this_race_correct_turns


class TurnChooser:
    """
    Weights multiple closest points to choose the next turn
    """

    def __init__(self, num_closest) -> None:
        self.num_closest = num_closest

    def t_junction_choose_turn(
            self,
            position: Position,
            correct_turns: Dict[Position, Action],
            left_right_distance: List[int]):
        if len(correct_turns) == 0:
            return Action.TurnLeft if left_right_distance[0] < left_right_distance[1] else Action.TurnRight

        # Find multiple closest data-points
        distances = np.array([position.distance_to(turn_position) for turn_position in correct_turns])
        closest_ids = np.argsort(distances)[:min(self.num_closest, len(distances))]
        correct_actions = list(correct_turns.values())

        # If one is exact, use it
        if distances[closest_ids[0]] == 0:
            return correct_actions[closest_ids[0]]
        # If the closest one is too far, pick the shorter straight
        if distances[closest_ids[0]] > 10.0:
            return Action.TurnLeft if left_right_distance[0] < left_right_distance[1] else Action.TurnRight

        # Else use the weighted sum
        left_ids = [i for i in closest_ids if correct_actions[i] == Action.TurnLeft]
        right_ids = [i for i in closest_ids if correct_actions[i] == Action.TurnRight]
        left_weight = sum(map(lambda d: 1.0 / d, distances[left_ids]))
        right_weight = sum(map(lambda d: 1.0 / d, distances[right_ids]))
        return Action.TurnLeft if left_weight > right_weight else Action.TurnRight


class SafetyCarTracker:
    """
    Tracks safety car speeds
    """

    MIN_SPEED = 30
    MAX_SPEED = 300

    def __init__(self):
        # List of different safety car instances with different speeds
        self.bounds = []
        # Safety car counter
        self.safety_car_id = -1
        # Penalty amount during a race
        self.race_penalties = 0
        # Penalty amount during a safety car event
        self.instance_penalties = 0
        # Current/temporary speed bounds
        self.temp_bounds = None
        # Tracks continuous safety car event
        self.was_safety_car_previous_turn = False
        # Force braking next turn
        self.force_braking = False

    def new_race(self):
        self.safety_car_id = -1
        self.race_penalties = 0
        self.instance_penalties = 0
        self.temp_bounds = None
        self.was_safety_car_previous_turn = False
        self.force_braking = False

    def new_track_state(self, track_state: TrackState):
        if track_state.safety_car_active:

            if not self.was_safety_car_previous_turn:
                # Commit temporary bounds from the previous safety car event in the race if one exists
                if self.temp_bounds is not None and self.safety_car_id >= 0:
                    self.bounds[self.safety_car_id] = self.temp_bounds

                # Safety car just activated
                self.safety_car_id += 1
                self.instance_penalties = 0
                self.was_safety_car_previous_turn = True

                if self.safety_car_id >= len(self.bounds):
                    # Safety car iteration not seen before
                    self.bounds.append([SafetyCarTracker.MIN_SPEED, SafetyCarTracker.MAX_SPEED])

                # Prepare temp bounds fot this event
                self.temp_bounds = self.bounds[self.safety_car_id].copy()

            # If under the penalty threshold experiment, else play safe
            if self.force_braking:
                self.force_braking = False
                return 0
            elif self.instance_penalties < 1 and self.race_penalties < 6:
                return sum(self.temp_bounds) / 2.0
            else:
                return self.temp_bounds[0]

        else:
            # Event ending
            self.was_safety_car_previous_turn = False
            self.force_braking = False
            return 1e+5

    def action_result(self, new_car_state: CarState, result: ActionResult):
        if self.was_safety_car_previous_turn:
            if result.safety_car_speed_exceeded:
                # Record the penalty and lover the speed bounds
                self.race_penalties += 1
                self.instance_penalties += 1
                self.temp_bounds[1] = min(self.temp_bounds[1], new_car_state.speed)
            else:
                # Increase the speed bounds
                self.temp_bounds[0] = max(self.temp_bounds[0], new_car_state.speed)

            if self.temp_bounds[0] > self.temp_bounds[1]:
                # Fingers crossed
                # Event was started immediately after the previous one finished so the change cannot be detected
                # Discard faulty temp bounds and roll to the next safety car id
                self.temp_bounds = None
                self.was_safety_car_previous_turn = False
                self.force_braking = True

        # Make sure what we learned from the last event is stored
        if result.finished and self.temp_bounds is not None and self.safety_car_id >= 0:
            self.bounds[self.safety_car_id] = self.temp_bounds


class RookieCarDynamicsTracker:
    """
    Tracks the car dynamics for querying and simulation
    """

    ACCEL_ACTIONS = [
        Action.FullThrottle, Action.LightThrottle, Action.HeavyBrake, Action.LightBrake
    ]

    def __init__(self):
        # Collected data for each acceleration action
        self.data = {
            drs_active: {
                action: [] for action in RookieCarDynamicsTracker.ACCEL_ACTIONS
            } for drs_active in [False, True]
        }

        self.cornering_speed_bounds = [10, 300]
        self.top_speed_recorded = { drs: 1 for drs in [True, False] }

    def add_data_point(self, action: Action, previous_car_state: CarState, new_car_state: CarState, result: ActionResult):

        if action in RookieCarDynamicsTracker.ACCEL_ACTIONS:
            # Handle acceleration actions
            current_data = self.data[previous_car_state.drs_active][action]
            if len(current_data) > 0:
                # Find the closest existing data point
                distances = np.abs(np.array(current_data)[:, 0] - previous_car_state.speed)
                closest_id = np.argmin(distances)
                if distances[closest_id] > 1.0:
                    # Add the new record if it is not too close
                    current_data.append([previous_car_state.speed, new_car_state.speed])
            else:
                current_data.append([previous_car_state.speed, new_car_state.speed])

        elif action in [Action.TurnLeft, Action.TurnRight]:
            # Handle turning actions
            if result.spun:
                # Spin
                self.cornering_speed_bounds[1] = min(
                    self.cornering_speed_bounds[1],
                    previous_car_state.speed)
            else:
                # Turned
                self.cornering_speed_bounds[0] = max(
                    self.cornering_speed_bounds[0],
                    previous_car_state.speed)

        elif action == Action.Continue:
            # Sanity check
            assert previous_car_state.speed == new_car_state.speed

        # Update max recorded speed
        self.top_speed_recorded[previous_car_state.drs_active] = max(
            self.top_speed_recorded[previous_car_state.drs_active],
            previous_car_state.speed)
        self.top_speed_recorded[new_car_state.drs_active] = max(
            self.top_speed_recorded[new_car_state.drs_active],
            new_car_state.speed)

    def max_cornering_speed(self):
        # Weighted average between known bounds
        return (self.cornering_speed_bounds[0] * 4.0 + self.cornering_speed_bounds[1]) / 5.0

    def estimate_next_speed(self, current_speed, action: Action, drs_active: bool):
        if action == Action.Continue:
            return current_speed
        assert action in RookieCarDynamicsTracker.ACCEL_ACTIONS
        current_data = np.array(self.data[drs_active][action])
        if len(current_data) < 2:
            return current_speed
        # Execute interpolation
        interpolation = scipy.interpolate.interp1d(
            current_data[:, 0], current_data[:, 1], fill_value='extrapolate', assume_sorted=False)
        return interpolation(current_speed).clip(min=0)

    def choose_best_action(self, speed: float, target_speed: float, drs_active: bool):
        # Choose a best action to get just under the target speed
        actions = Action.get_sl_actions()
        if 0 == speed:
            actions = [Action.LightThrottle, Action.FullThrottle]
        next_speeds = np.array([self.estimate_next_speed(speed, action, drs_active) for action in actions])
        errors = next_speeds - target_speed
        if np.any(errors <= 0):
            errors[errors > 0] = np.inf
        best_action_id = np.argmin(errors ** 2)
        return actions[best_action_id]

    def get_fardest_action(self, speed: float, drs_active: bool, actions: List[Action] = ACCEL_ACTIONS):
        # Find the action whose closest data points are the fardest among acceleration actions
        current_data = self.data[drs_active]
        for action in actions:
            if len(current_data[action]) == 0:
                return action, 1e+5
        distances = {
            action: min([abs(speed - data_point[0]) for data_point in current_data[action]])
            for action in actions
        }
        fardest_action = max(distances, key=distances.get)
        return fardest_action, distances[fardest_action]


class StraightSimulator:
    """
    Deals with predicting future
    """

    def __init__(self):
        pass

    @staticmethod
    def get_target_speeds(
            car_dynamics_tracker: RookieCarDynamicsTracker,
            distance_ahead: int,
            potential_stop: bool = False):
        if distance_ahead == 0:
            return np.zeros(1)

        target_speeds = np.zeros(distance_ahead + 1)
        # Final speed, for imaginary distance_ahead = -1
        target_speeds[0] = 0
        # Max conering speed for distance_ahead = 0
        target_speeds[1] = car_dynamics_tracker.max_cornering_speed()
        # In case that we may need to stop at the end of the straight, invalidate the max cornering speed
        for i in range(1 if potential_stop else 2, distance_ahead + 1):
            # Binary search for the optimal speed
            bounds = [target_speeds[i - 1], target_speeds[i - 1] + 100.0]
            while bounds[1] - bounds[0] > 1.0:
                target_speed = sum(bounds) / 2.0
                braking_result = min([
                    car_dynamics_tracker.estimate_next_speed(target_speed, action, False)
                    for action in [Action.LightBrake, Action.HeavyBrake]
                ])
                if braking_result <= target_speeds[i - 1]:
                    bounds[0] = target_speed
                elif braking_result > target_speeds[i - 1]:
                    bounds[1] = target_speed
            # Assign the lower bound to the target speed for this step
            target_speeds[i] = bounds[0]
        return target_speeds

    @staticmethod
    def simulate_straight(
            car_dynamics_tracker: RookieCarDynamicsTracker,
            speed: float,
            distance_ahead: int,
            potential_stop: bool,
            drs_active: bool,
            safety_car_speed: float = 0.0):
        # Find optimal actions to complete the straight
        speeds = np.zeros(distance_ahead)
        target_speeds = StraightSimulator.get_target_speeds(car_dynamics_tracker, distance_ahead, potential_stop)
        break_target_speed = False
        for d in range(distance_ahead):
            # Find step target speed
            target_speed = target_speeds[-d - 1]
            if safety_car_speed > 0.1:
                target_speed = min(target_speed, safety_car_speed)
            # Find the best action we can manage
            action = car_dynamics_tracker.choose_best_action(speed, target_speed, drs_active)
            # Simulate the action
            speeds[d] = car_dynamics_tracker.estimate_next_speed(speed, action, drs_active)
            speed = speeds[d]
            break_target_speed |= speed > target_speed
        time = np.sum(1 / (speeds + 1))
        return time, break_target_speed, speeds


class MyDriver(Driver):

    def __init__(self, race_logger_dir: str):
        super().__init__('McLando')
        self.race_logger_dir = race_logger_dir

        self.race_logger = None
        self.turn_tracker = RealtimeTurnTracker()
        self.turn_chooser = TurnChooser(3)
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
