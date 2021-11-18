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

    @staticmethod
    def simulate_straight(
            car_dynamics_tracker: CarDynamicsTracker,
            speed: float,
            distance_ahead: int,
            potential_stop: bool,
            drs_active: bool,
            safety_car_speed: float = 0.0):
        speeds = np.zeros(distance_ahead)
        break_target_speed = False
        for d in range(distance_ahead):
            target_speed = StraightSimulator.get_target_speeds(car_dynamics_tracker, distance_ahead, potential_stop)[-1]
            if safety_car_speed > 0.1:
                target_speed = min(target_speed, safety_car_speed)
            action = car_dynamics_tracker.choose_best_action(speed, target_speed, drs_active)
            speeds[d] = car_dynamics_tracker.estimate_next_speed(speed, action, drs_active)
            speed = speeds[d]
            break_target_speed |= speed > target_speed
        time = np.sum(1 / (speeds + 1))
        return time, break_target_speed, speeds
