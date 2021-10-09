from imports import *


class TrackerState:
    """
    Base class for turn tracker states
    """

    def __init__(self, turn_tracker):
        self.turn_tracker = turn_tracker

    def new_track_state(self, track_state, is_final):
        raise NotImplementedError


class DefaultTrackerState(TrackerState):
    """
    Turn tracker state when no turns are being examined
    """

    def new_track_state(self, track_state, is_final):
        if is_final:
            return self

        if track_state.distance_ahead == 0 and \
                track_state.distance_left > 0 and \
                track_state.distance_right > 0:
            return TurnJustTakenTrackerState(self.turn_tracker)
        else:
            return self


class TurnJustTakenTrackerState(TrackerState):
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

        return CheckingTakenTurnTrackerState(self.turn_tracker, taken_action).new_track_state(track_state, is_final)


class CheckingTakenTurnTrackerState(TrackerState):
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
            self.turn_tracker.correct_turns[self.turn_position] = self.taken_action
        else:
            # Dead end, wrong turn taken
            self.turn_tracker.correct_turns[self.turn_position] = \
                Action.TurnLeft if self.taken_action == Action.TurnRight else Action.TurnRight

        # Immediately check if new default state is finished
        return DefaultTrackerState(self.turn_tracker).new_track_state(track_state, False)


class TurnTracker:
    """
    Class used to track correct turns in real time,
    without waiting for the end of the race
    """

    def __init__(self):
        self.correct_turns = {}
        self._current_state = None
        self._recent_car_positions = []

    def new_race(self):
        """
        Reset race-specific turn tracker parameters
        """
        self._current_state = DefaultTrackerState(self)
        self._recent_car_positions = []

    def new_track_state(self, track_state, is_final=False):
        """
        Handle a new track state
        """
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


if __name__ == '__main__':

    from random import choice

    def are_equivalent(turns1, turns2):
        # For double checking with ==
        if len(turns1) != len(turns2):
            return False
        for key in turns1:
            if key not in turns2 or turns1[key] != turns2[key]:
                return False
        return True

    # Test the turn tracker on all of the available tracks
    # Test both individually on all tracks and cumulative through all tracks
    global_turn_tracker = TurnTracker()

    all_tracks = TrackStore.load_all_tracks(level=Level.Young)
    for track in all_tracks:

        position = track.start_position
        heading = track.start_heading

        local_turn_tracker = TurnTracker()
        local_turn_tracker.new_race()
        global_turn_tracker.new_race()

        while not track.is_finished(position):

            track_state = track.get_state_for_position(position, heading)
            local_turn_tracker.new_track_state(track_state)
            global_turn_tracker.new_track_state(track_state)

            if track_state.distance_ahead > 0:
                position, _, _ = track.get_new_position(position, 1, heading)
            else:
                # Decide on the correct turn
                if track_state.distance_left == 0:
                    heading = heading.get_right_heading()
                elif track_state.distance_right == 0:
                    heading = heading.get_left_heading()
                else:
                    # Take a random turn (sometimes correct, sometime not correct)
                    heading = choice([heading.get_right_heading, heading.get_left_heading])()

        track_state = track.get_state_for_position(position, heading)
        local_turn_tracker.new_track_state(track_state, is_final=True)
        global_turn_tracker.new_track_state(track_state, is_final=True)

        assert are_equivalent(local_turn_tracker.correct_turns, track.correct_turns)
        assert local_turn_tracker.correct_turns == track.correct_turns

    accumulated_correct_turns = dict()
    for track in all_tracks:
        accumulated_correct_turns.update(track.correct_turns)

    assert are_equivalent(global_turn_tracker.correct_turns, accumulated_correct_turns)
    assert global_turn_tracker.correct_turns == accumulated_correct_turns

    print('All is good!')
