from multiprocessing import Process, Queue
import os
import sys

from imports import *
from drivers.rookiedriver import RookieDriver
from nikola.race_logger import RaceLogger
from nikola.turn_chooser import *
from nikola.turn_tracker import *


class RookieDriverTurnChooser(RookieDriver):

    def __init__(self, name, race_logger_dir, turn_tracker: TurnTracker, turn_chooser: TurnChooser, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.race_logger_dir = race_logger_dir
        self.turn_tracker = turn_tracker
        self.turn_chooser = turn_chooser
        self.race_id = 0
        self.race_logger = None

    def prepare_for_race(self):
        self.race_id += 1
        if self.race_logger_dir != '':
            self.race_logger = RaceLogger(os.path.join(self.race_logger_dir, '{}.png'.format(self.race_id)))
        self.turn_tracker.new_race()
        return super().prepare_for_race()

    def _choose_turn_direction(self, track_state: TrackState):
        if track_state.distance_left > 0 and track_state.distance_right > 0:
            action = self.turn_chooser.t_junction_choose_turn(
                self.race_id,
                track_state.position,
                self.turn_tracker.correct_turns,
                [track_state.distance_left, track_state.distance_right])
            return action
        elif track_state.distance_left > 0:
            return Action.TurnLeft
        else:
            return Action.TurnRight

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
            action: Action, new_car_state: CarState, new_track_state: TrackState, result: ActionResult):
        if self.race_logger is not None:
            self.race_logger.log_race_step(
                previous_car_state, previous_track_state, action,
                new_car_state, new_track_state, result)
        self.turn_tracker.new_track_state(previous_track_state, is_final=False)
        if result.finished:
            self.turn_tracker.new_track_state(new_track_state, is_final=True)
        return super().update_with_action_results(
            previous_car_state, previous_track_state, action,
            new_car_state, new_track_state, result)

    def update_after_race(self, correct_turns: Dict[Position, Action]):
        self.turn_tracker.update_after_race(correct_turns)


class ChampionshipManager:
    def __init__(
            self,
            turn_tracker: TurnTracker,
            turn_chooser: TurnChooser,
            shuffle_tracks: bool,
            num_repeats: int,
            save_path: str):
        self.num_repeats = num_repeats
        self.save_path = save_path
        drivers = [
            RookieDriver('RD'),
            RookieDriverTurnChooser('RFTC', '', turn_tracker, turn_chooser)
        ]
        self.championship = Championship(drivers, Level.Rookie, shuffle_tracks=shuffle_tracks, verbose=True)
        self.championship_results = Queue()

    def run(self):
        championship_results, _, _ = self.championship.run_championship(num_repeats=self.num_repeats)
        self.championship_results.put(championship_results)

    def plot(self):
        championship_results = self.championship_results.get()
        plot_multiple_championship_results(championship_results)
        plt.savefig(self.save_path)
        plt.close()


def run_championship(championship):
    championship.run()


def join_and_plot(proccesses):
    for c, p in proccesses.items():
        print('Joining {}'.format(c.save_path))
        p.join()
        print('Joined {}'.format(c.save_path))
        c.plot()

    proccesses.clear()


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    # Run one season with turn logger
    Season(Level.Rookie).race(
        RookieDriverTurnChooser(
            'RFTC_DD',
            os.path.join(path, 'default_default'),
            DefaultTurnTracker(),
            DefaultTurnChooser()))

    shuffle_tracks = True
    verbose = True
    num_repeats = 1000

    turn_trackers = [
        (DefaultTurnTracker, 'default'),
        (RealtimeTurnTracker, 'realtime')
    ]

    turn_choosers = [(DefaultTurnChooser, [], 'default')]
    for use_shorter in [False, True]:
        turn_choosers.extend([
            (BalanceLeftRightTurnChooser, [8.0, use_shorter], 'balance_{}'.format(use_shorter)),
            (MultipleClosestTurnChooser, [3, use_shorter], 'multiple_{}'.format(use_shorter)),
            (CombinedTurnChooser, [8.0, 3, use_shorter], 'combined_{}'.format(use_shorter))
        ])

    i = 0
    proccesses = {}
    max_proccesses = 10
    for turn_chooser in turn_choosers:
        for turn_tracker in turn_trackers:
            championship = ChampionshipManager(
                turn_tracker[0](),
                turn_chooser[0](*turn_chooser[1]),
                shuffle_tracks,
                num_repeats,
                os.path.join(path, '{:02}_RD_vs_RFTC_{}_{}.png'.format(i, turn_tracker[1], turn_chooser[2]))
            )
            p = Process(target=run_championship, args=(championship,))
            proccesses[championship] = p
            p.start()
            if len(proccesses) == max_proccesses:
                join_and_plot(proccesses)
            i += 1

    join_and_plot(proccesses)

    print('All done')
