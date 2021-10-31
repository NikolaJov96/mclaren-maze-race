import scipy

from imports import *


class CarDynamicsTracker:

    ACCEL_ACTIONS = [
        Action.FullThrottle, Action.LightThrottle, Action.HeavyBrake, Action.LightBrake
    ]

    def add_data_point(self, action: Action, previous_car_state: CarState, new_car_state: CarState):
        raise NotImplementedError

    def max_cornering_speed(self):
        raise NotImplementedError

    def accel_action(self, current_speed, action: Action):
        assert action in CarDynamicsTracker.ACCEL_ACTIONS
        return current_speed

    def can_turn(self, current_speed):
        return current_speed < self.max_cornering_speed()


class RookieCarDynamicsTracker(CarDynamicsTracker):

    INC_LOW =  [0.0, 1.0]
    INC_HIGH = [300.0, 301.0]
    DEC_LOW = [1.0, 0.0]
    DEC_HIGH = [301.0, 300.0]

    def __init__(self):
        self.data = {
            drs_active: {
                Action.FullThrottle: [RookieCarDynamicsTracker.INC_LOW, RookieCarDynamicsTracker.INC_HIGH],
                Action.LightThrottle: [RookieCarDynamicsTracker.INC_LOW, RookieCarDynamicsTracker.INC_HIGH],
                Action.HeavyBrake: [[RookieCarDynamicsTracker.DEC_LOW, RookieCarDynamicsTracker.DEC_HIGH]],
                Action.LightBrake: [RookieCarDynamicsTracker.DEC_LOW, RookieCarDynamicsTracker.DEC_HIGH]
            } for drs_active in [False, True]
        }

        self.cornering_speed_bounds = [10, 300]

    def add_data_point(self, action: Action, previous_car_state: CarState, new_car_state: CarState):

        if action in CarDynamicsTracker.ACCEL_ACTIONS:
            current_data = self.data[previous_car_state.drs_active][action]
            # Remove initial values
            if previous_car_state.speed < 5.0:
                current_data.remove(RookieCarDynamicsTracker.INC_LOW)
                current_data.remove(RookieCarDynamicsTracker.DEC_LOW)
            elif previous_car_state.speed > 200.0:
                current_data.remove(RookieCarDynamicsTracker.INC_HIGH)
                current_data.remove(RookieCarDynamicsTracker.DEC_HIGH)
            # Remove closest value if too close
            distances = np.abs(np.array(current_data)[:, 0] - previous_car_state.speed)
            closest_id = np.argmin(distances)
            if distances[closest_id] < 1.0:
                del current_data[closest_id]
            # Add the new record
            current_data.append([[previous_car_state.speed, new_car_state.speed]])

        elif action in [Action.TurnLeft, Action.TurnRight]:
            if previous_car_state.heading == new_car_state.heading:
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
            assert previous_car_state.speed == new_car_state.speed

    def max_cornering_speed(self):
        return sum(self.cornering_speed_bounds) / 2.0

    def accel_action(self, current_speed, action: Action, drs_active: bool):
        super().accel_action(current_speed, action)
        current_data = self.data[drs_active][action]
        assert len(current_data) > 2
        interpolation = scipy.interpolate.interp1d(
            current_data[:, 0], current_data[:, 1], fill_value='extrapolate', assume_sorted=False)
        return interpolation(current_speed)


if __name__ == '__main__':

    class TestRookieDriver(RookieDriver):

        def __init__(self, name, random_action_probability=0.5, random_action_decay=0.99, min_random_action_probability=0, *args, **kwargs):
            super().__init__(name, random_action_probability=random_action_probability, random_action_decay=random_action_decay, min_random_action_probability=min_random_action_probability, *args, **kwargs)
            self.car_dynamics_tracker = RookieCarDynamicsTracker()

    # Check if deep copying works fine
    driver1 = TestRookieDriver('D1')
    driver2 = deepcopy(driver1)

    assert len(driver1.car_dynamics_tracker.data[False][Action.FullThrottle]) == 2
    assert len(driver2.car_dynamics_tracker.data[False][Action.FullThrottle]) == 2
    assert driver1.car_dynamics_tracker.cornering_speed_bounds[0] == 10
    assert driver2.car_dynamics_tracker.cornering_speed_bounds[0] == 10

    driver1.car_dynamics_tracker.add_data_point(
        action=Action.FullThrottle,
        previous_car_state=CarState(speed=100.0, heading=(1, 0)),
        new_car_state=CarState(speed=110.0, heading=(1, 0)))
    driver1.car_dynamics_tracker.add_data_point(
        action=Action.TurnRight,
        previous_car_state=CarState(speed=110.0, heading=(1, 0)),
        new_car_state=CarState(speed=110.0, heading=Heading(1, 0).get_right_heading()))

    assert len(driver1.car_dynamics_tracker.data[False][Action.FullThrottle]) == 3
    assert len(driver2.car_dynamics_tracker.data[False][Action.FullThrottle]) == 2
    assert driver1.car_dynamics_tracker.cornering_speed_bounds[0] == 110
    assert driver2.car_dynamics_tracker.cornering_speed_bounds[0] == 10

    print('All is good')
