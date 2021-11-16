from imports import *

from nikola.car_dynamics_tracker import CarDynamicsTracker


class StraightSimulator:

    def __init__(self):
        pass

    @staticmethod
    def get_target_speeds(car_dynamics_tracker: CarDynamicsTracker, distance_ahead: int, potential_stop: bool = False):
        if distance_ahead == 0:
            return np.zeros(1)

        target_speeds = np.zeros(distance_ahead + 1)
        target_speeds[0] = 0
        target_speeds[1] = car_dynamics_tracker.max_cornering_speed()
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

    def simulate_straight(self, speed: float, distance_ahead: int, drs_active: bool, safety_car_active: bool):
        pass
