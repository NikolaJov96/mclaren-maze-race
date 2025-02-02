import scipy

from imports import *


class CarDynamicsTracker:

    ACCEL_ACTIONS = [
        Action.FullThrottle, Action.LightThrottle, Action.HeavyBrake, Action.LightBrake
    ]

    def add_data_point(self, action: Action, previous_car_state: CarState, new_car_state: CarState, result: ActionResult):
        raise NotImplementedError

    def max_cornering_speed(self):
        raise NotImplementedError

    def estimate_next_speed(self, current_speed, action: Action):
        assert action in CarDynamicsTracker.ACCEL_ACTIONS
        return current_speed

    def can_turn(self, current_speed):
        return current_speed < self.max_cornering_speed()

    def choose_best_action(self, speed: float, target_speed: float, drs_active: bool):
        raise NotImplementedError

    def get_fardest_action(self, speed: float, drs_active: bool):
        raise NotImplementedError


class RookieCarDynamicsTracker(CarDynamicsTracker):
    """
    Tracks the car dynamics for querying and simulation
    """

    def __init__(self):
        # Collected data for each acceleration action
        self.data = {
            drs_active: {
                action: [] for action in CarDynamicsTracker.ACCEL_ACTIONS
            } for drs_active in [False, True]
        }

        self.cornering_speed_bounds = [10, 300]
        self.top_speed_recorded = { drs: 1 for drs in [True, False] }

    def add_data_point(self, action: Action, previous_car_state: CarState, new_car_state: CarState, result: ActionResult):

        if action in CarDynamicsTracker.ACCEL_ACTIONS:
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
        super().estimate_next_speed(current_speed, action)
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

    def get_fardest_action(self, speed: float, drs_active: bool, actions: List[Action] = CarDynamicsTracker.ACCEL_ACTIONS):
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


if __name__ == '__main__':

    class TestRookieDriver(RookieDriver):

        def __init__(self, name, random_action_probability=0.5, random_action_decay=0.99, min_random_action_probability=0, *args, **kwargs):
            super().__init__(name, random_action_probability=random_action_probability, random_action_decay=random_action_decay, min_random_action_probability=min_random_action_probability, *args, **kwargs)
            self.car_dynamics_tracker = RookieCarDynamicsTracker()

    # Check if deep copying works fine
    driver1 = TestRookieDriver('D1')
    driver2 = deepcopy(driver1)

    assert len(driver1.car_dynamics_tracker.data[False][Action.FullThrottle]) == 0
    assert len(driver2.car_dynamics_tracker.data[False][Action.FullThrottle]) == 0
    assert driver1.car_dynamics_tracker.cornering_speed_bounds[0] == 10
    assert driver2.car_dynamics_tracker.cornering_speed_bounds[0] == 10

    driver1.car_dynamics_tracker.add_data_point(
        action=Action.FullThrottle,
        previous_car_state=CarState(speed=100.0, heading=(1, 0)),
        new_car_state=CarState(speed=110.0, heading=(1, 0)),
        result=ActionResult(turned_ok=True, crashed=False, spun=False, finished=False))
    driver1.car_dynamics_tracker.add_data_point(
        action=Action.TurnRight,
        previous_car_state=CarState(speed=110.0, heading=(1, 0)),
        new_car_state=CarState(speed=110.0, heading=Heading(1, 0).get_right_heading()),
        result=ActionResult(turned_ok=True, crashed=False, spun=False, finished=False))

    assert len(driver1.car_dynamics_tracker.data[False][Action.FullThrottle]) == 1
    assert len(driver2.car_dynamics_tracker.data[False][Action.FullThrottle]) == 0
    assert driver1.car_dynamics_tracker.cornering_speed_bounds[0] == 110
    assert driver2.car_dynamics_tracker.cornering_speed_bounds[0] == 10

    print('All is good')
