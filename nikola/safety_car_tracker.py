from imports import *


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
