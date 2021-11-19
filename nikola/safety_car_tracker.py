from imports import *


class SafetyCarTracker:

    MIN_SPEED = 30
    MAX_SPEED = 300

    def __init__(self):
        self.bounds = []
        self.safety_car_id = -1
        self.race_penalties = 0
        self.instance_penalties = 0
        self.was_safety_car_previous_turn = False

    def new_race(self):
        self.safety_car_id = -1
        self.race_penalties = 0
        self.instance_penalties = 0
        self.was_safety_car_previous_turn = False

    def new_track_state(self, track_state: TrackState):
        if track_state.safety_car_active:

            if not self.was_safety_car_previous_turn:
                # Safety car just activated
                self.safety_car_id += 1
                self.instance_penalties = 0
                self.was_safety_car_previous_turn = True

                if self.safety_car_id >= len(self.bounds):
                    # Safety car iteration not seen before
                    self.bounds.append([SafetyCarTracker.MIN_SPEED, SafetyCarTracker.MAX_SPEED])

            if self.instance_penalties < 1 and self.race_penalties < 6:
                return sum(self.bounds[self.safety_car_id]) / 2.0
            else:
                return self.bounds[self.safety_car_id][0]

        else:
            self.was_safety_car_previous_turn = False
            return 1e+5

    def action_result(self, new_car_state: CarState, result: ActionResult):
        if self.was_safety_car_previous_turn:
            if result.safety_car_speed_exceeded:
                self.race_penalties += 1
                self.instance_penalties += 1
                self.bounds[self.safety_car_id][1] = min(
                    self.bounds[self.safety_car_id][1],
                    new_car_state.speed)
            else:
                self.bounds[self.safety_car_id][0] = max(
                    self.bounds[self.safety_car_id][0],
                    new_car_state.speed)
