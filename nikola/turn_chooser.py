from imports import *


class TurnChooser:
    """
    Base class representing a way to choose a turn on a T junction
    Picks a random turn
    """

    def check(self, action: Action):
        # Check turn action validity
        assert action in [Action.TurnLeft, Action.TurnRight]
        return action

    def t_junction_choose_turn(
            self,
            race_id: int,
            position: Position,
            correct_turns: Dict[Position, Action],
            left_right_distance: List[int]):
        return self.check(driver_rng().choice([Action.TurnLeft, Action.TurnRight]))


class DefaultTurnChooser(TurnChooser):
    """
    Default way to choose a turn on a T junction
    Picks the same turn as the closest known correct turn,
    or a random turn if no correct turns are known
    """

    def t_junction_choose_turn(
            self,
            race_id: int,
            position: Position,
            correct_turns: Dict[Position, Action],
            left_right_distance: List[int]):
        if len(correct_turns) > 0:
            distances = np.array([
                position.distance_to(turn_position) for turn_position in correct_turns])
            i_closest = np.argmin(distances)
            return self.check(list(correct_turns.values())[i_closest])
        else:
            return super().t_junction_choose_turn(race_id, position, correct_turns, left_right_distance)


class BalanceLeftRightTurnChooser(DefaultTurnChooser):
    """
    Balances the number of total left and right turns in the campionship
    """

    def __init__(self, min_closest_distance, choose_shorter) -> None:
        super().__init__()
        self.min_closest_distance = min_closest_distance
        self.choose_shorter = choose_shorter

    def t_junction_choose_turn(
            self,
            race_id: int,
            position: Position,
            correct_turns: Dict[Position, Action],
            left_right_distance: List[int]):
        if len(correct_turns) == 0:
            if self.choose_shorter and left_right_distance[0] != left_right_distance[1]:
                return self.check(Action.TurnLeft if left_right_distance[0] < left_right_distance[1] else Action.TurnRight)
            else:
                return self.check(driver_rng().choice([Action.TurnLeft, Action.TurnRight]))

        closest_distance = min([position.distance_to(turn_position) for turn_position in correct_turns])
        correct_turns_list = list(correct_turns)
        num_left = correct_turns_list.count(Action.TurnLeft)
        num_right = correct_turns_list.count(Action.TurnRight)
        if closest_distance <= self.min_closest_distance:
            return super().t_junction_choose_turn(race_id, position, correct_turns, left_right_distance)
        elif num_left == num_right:
            if self.choose_shorter and left_right_distance[0] != left_right_distance[1]:
                return self.check(Action.TurnLeft if left_right_distance[0] < left_right_distance[1] else Action.TurnRight)
            else:
                return super().t_junction_choose_turn(race_id, position, correct_turns, left_right_distance)
        else:
            return self.check(Action.TurnLeft if num_left < num_right else Action.TurnRight)


class MultipleClosestTurnChooser(TurnChooser):
    """
    Weights multiple closest points to choose the next turn
    """

    def __init__(self, num_closest, choose_shorter) -> None:
        super().__init__()
        self.num_closest = num_closest
        self.choose_shorter = choose_shorter

    def t_junction_choose_turn(
            self,
            race_id: int,
            position: Position,
            correct_turns: Dict[Position, Action],
            left_right_distance: List[int]):
        if len(correct_turns) == 0:
            if self.choose_shorter and left_right_distance[0] != left_right_distance[1]:
                return self.check(Action.TurnLeft if left_right_distance[0] < left_right_distance[1] else Action.TurnRight)
            else:
                return self.check(driver_rng().choice([Action.TurnLeft, Action.TurnRight]))

        distances = np.array([position.distance_to(turn_position) for turn_position in correct_turns])
        closest_ids = np.argsort(distances)[:min(self.num_closest, len(distances))]
        correct_actions = list(correct_turns.values())

        if distances[closest_ids[0]] == 0:
            return self.check(correct_actions[closest_ids[0]])

        left_ids = [i for i in closest_ids if correct_actions[i] == Action.TurnLeft]
        right_ids = [i for i in closest_ids if correct_actions[i] == Action.TurnRight]
        left_weight = sum(map(lambda d: 1.0 / d, distances[left_ids]))
        right_weight = sum(map(lambda d: 1.0 / d, distances[right_ids]))
        return self.check(Action.TurnLeft if left_weight > right_weight else Action.TurnRight)


class CombinedTurnChooser(TurnChooser):
    """
    Uses BalanceLeftRightTurnChooser for the first half of the season and
    MultipleClosestTurnChooser for the second half
    """

    def __init__(self, min_closest_distance, num_closest, choose_shorter) -> None:
        super().__init__()
        self.balanced_tc = BalanceLeftRightTurnChooser(min_closest_distance, choose_shorter)
        self.multiple_tc = MultipleClosestTurnChooser(num_closest, choose_shorter)

    def t_junction_choose_turn(
            self,
            race_id: int,
            position: Position,
            correct_turns: Dict[Position, Action],
            left_right_distance: List[int]):
        if race_id < 13:
            return self.balanced_tc.t_junction_choose_turn(race_id, position, correct_turns, left_right_distance)
        else:
            return self.multiple_tc.t_junction_choose_turn(race_id, position, correct_turns, left_right_distance)
