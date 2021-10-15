import os
from typing_extensions import runtime

from imports import *


class RaceLogger:

    def __init__(self, save_file):
        self.save_file = save_file

        self.end_of_straight_ids = []
        self.drs_available_ids = []
        self.safety_car_active_ids = []

        self.speed = []
        self.drs_active_ids = []

        self.actions = []

        self.turned_ok_ids = []
        self.crashed_ids = []
        self.spun_ids = []
        self.safety_car_penalty_level = {}

        self.step_id = 0

        save_file_dir = os.path.dirname(self.save_file)
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)

    def log_race_step(self,
            car_state: CarState, track_state: TrackState, action: Action,
            next_car_state: CarState, next_track_state: TrackState, result: ActionResult):
        # Car and track state
        self._add_car_track_state(car_state, track_state)
        if result.finished:
            self._add_car_track_state(next_car_state, next_track_state)

        # Action
        self.actions.append(action)

        # Result
        if result.turned_ok:
            self.turned_ok_ids.append(self.step_id + 0.5)
        if result.crashed:
            self.crashed_ids.append(self.step_id + 0.5)
        if result.spun:
            self.spun_ids.append(self.step_id + 0.5)
        if result.safety_car_speed_exceeded:
            self.safety_car_penalty_level[self.step_id + 0.5] = result.safety_car_penalty_level

        self.step_id += 1

        if result.finished:
            self._plot()

    def _add_car_track_state(self, car_state, track_state):
        # Car state
        self.speed.append(car_state.speed)
        if car_state.drs_active:
            self.drs_active_ids.append(self.step_id)

        # Track state
        if track_state.distance_ahead == 0:
            self.end_of_straight_ids.append(self.step_id)
        if track_state.drs_available:
            self.drs_available_ids.append(self.step_id)
        if track_state.safety_car_active:
            self.safety_car_active_ids.append(self.step_id)

    def _plot(self):

        def plot_scatter(X, y, c, labels=None):
            plt.scatter(X, [y] * len(X), c=[c], s=150)
            if labels is not None:
                for i in range(len(labels)):
                    plt.text(X[i], y, labels[i])

        def get_action_ids(actions, target_action):
            return [i for i, action in enumerate(actions) if action == target_action]


        xticks = range(self.step_id + 1)

        plt.subplots(2, 1, sharex=True,
            figsize=(self.step_id // 2, 8),
            gridspec_kw={ 'height_ratios':[3, 1] })

        # Plot speed graph
        plt.subplot(2, 1, 1)
        # Speed
        plt.plot(xticks, self.speed)
        plt.xticks(xticks)
        plt.ylabel('speed')
        plt.xlim(0, self.step_id)
        plt.grid()

        # Plot event marker graph
        plt.subplot(2, 1, 2)
        y_tick_labels = [
            '',
            'turn / spin / crash',
            'throttle / brake',
            'drs available',
            'safety car active',
            ''
        ]
        # End of straight positions
        plot_scatter(self.end_of_straight_ids, y=1, c=(0.7, 0.7, 0.7))
        plot_scatter(self.crashed_ids, y=1, c=(0.0, 0.0, 0.0))
        plot_scatter(self.spun_ids, y=1, c=(1.0, 0.0, 0.5))
        plot_scatter(self.turned_ok_ids, y=1, c=(0.0, 1.0, 0.0))
        # Throttle, braking and turning
        light_throttle_ids = get_action_ids(self.actions, Action.LightThrottle)
        plot_scatter(light_throttle_ids, y=2, c=(1.0, 0.5, 0.5))
        full_throttle_ids = get_action_ids(self.actions, Action.FullThrottle)
        plot_scatter(full_throttle_ids, y=2, c=(1.0, 0.0, 0.0))
        light_brake_ids = get_action_ids(self.actions, Action.LightBrake)
        plot_scatter(light_brake_ids, y=2, c=(0.5, 0.5, 1.0))
        heavy_brake_ids = get_action_ids(self.actions, Action.HeavyBrake)
        plot_scatter(heavy_brake_ids, y=2, c=(0.0, 0.0, 1.0))
        # DRS
        plot_scatter(self.drs_available_ids, y=3, c=(0.8, 0.8, 0.3))
        plot_scatter(self.drs_active_ids, y=3, c=(1.0, 1.0, 0.0))
        # Safety car
        plot_scatter(self.safety_car_active_ids, y=4, c=(0.8, 0.8, 0.3))
        plot_scatter(
            list(self.safety_car_penalty_level.keys()),
            y=4,
            c=(1.0, 1.0, 0.0),
            labels=list(self.safety_car_penalty_level.values()))
        plt.yticks(range(len(y_tick_labels)), y_tick_labels)
        plt.gca().invert_yaxis()
        plt.grid()

        plt.tight_layout()
        plt.savefig(self.save_file)
        plt.close()
