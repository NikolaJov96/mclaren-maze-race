from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from drivers.driver import *
from drivers.rookiedriver import RookieDriver


class TyreTracker:
    """
    Models tyre properties
    """

    def __init__(self):
        self.tyre_data = {tyre_choice: np.empty((1000, 0)) for tyre_choice in TyreChoice.get_choices()}
        self.current_tyre_choice = TyreChoice.Medium            # hard coded for now, you can improve this!
        self.current_tyre_age = 0
        self.current_base_tyre_model = None
        self.current_tyre_parameters = None
        self.pit_loss = 3.0

    def choose_tyres(self, track_info: TrackInfo):
        # This method is called at the start of the race and whenever the driver chooses to make a pitstop. It needs to
        # return a TyreChoice enum

        # TODO: make an informed choice here!
        # self.current_tyre_choice = ...

        self.fit_base_tyre_model()
        return self.current_tyre_choice

    def update_tyre_data(self, car_state: CarState):
        if 0 == car_state.tyre_age:         # new tyre, add a column of nans ready for data
            n = self.tyre_data[car_state.tyre_choice].shape[0]
            self.tyre_data[car_state.tyre_choice] = np.hstack([self.tyre_data[car_state.tyre_choice],
                                                               np.full((n, 1), np.nan)])
        self.tyre_data[car_state.tyre_choice][car_state.tyre_age, -1] = car_state.tyre_grip
        self.current_tyre_age = car_state.tyre_age
        if car_state.tyre_choice != self.current_tyre_choice:
            self.current_tyre_choice = car_state.tyre_choice
            self.fit_base_tyre_model()

    @property
    def current_tyre_grip(self):
        return self.get_measured_tyre_grip(self.current_tyre_age)

    def get_measured_tyre_grip(self, age, tyre_choice=None):
        if tyre_choice is None:
            tyre_choice = self.current_tyre_choice

        if self.tyre_data[tyre_choice].shape[1] == 0:
            return np.nan
        else:
            return self.tyre_data[tyre_choice][age, -1]

    def should_we_change_tyres(self, driver):
        if np.nanmin(self.tyre_data[self.current_tyre_choice]) > 0.6:
            return False

        time_current_tyres = driver.simulate_to_end_of_race(self.current_tyre_age)
        time_new_tyres = driver.simulate_to_end_of_race(0)
        start_grip = self.get_measured_tyre_grip(0)
        lost_grip_fraction = (start_grip - self.current_tyre_grip) / start_grip

        return time_new_tyres < time_current_tyres - self.pit_loss and lost_grip_fraction > 0.1

    @staticmethod
    def logistic(p, t):
        return p[0] + p[1] / (1 + np.exp(-p[2] + t / p[3]))

    def fit_base_tyre_model(self):
        # Fit initial model
        data = self.tyre_data[self.current_tyre_choice]
        t = np.arange(data.shape[0])
        if 0 == data.shape[1] or np.sum(~np.isnan(data[:, -1])) < 2:
            # Have no data so use a logistic as an initial guess. It will be offset and stretched to fit the observed
            # data as it comes in
            if self.current_tyre_choice == TyreChoice.Soft:
                p = 0.15, 1.05, 8, 16
            elif self.current_tyre_choice == TyreChoice.Medium:
                p = [0.1, 0.9, 15, 20]
            else:
                p = [0.15, 0.65, 8, 50]
            self.current_base_tyre_model = lambda t: TyreTracker.logistic(p, t)

        else:
            b_no_nan = np.any(~np.isnan(data), axis=1)
            m = np.nanmean(data[b_no_nan, :], axis=1)
            t = t[b_no_nan]
            self.current_base_tyre_model = interp1d(t, m, kind='linear', fill_value='extrapolate')

    def forecast_tyre_grip(self, tyre_ages, parameters=None):
        if parameters is None:
            if self.current_tyre_parameters is None:
                self.fit_tyre_model()
            parameters = self.current_tyre_parameters
        x_offset, x_scale, y_offset, y_scale = parameters
        return y_offset + y_scale * self.current_base_tyre_model(x_offset + x_scale*tyre_ages)

    def fit_tyre_model(self):
        # Fit the base tyre model to the new data coming in but translating it and scaling it
        ages = np.arange(self.tyre_data[self.current_tyre_choice].shape[0])
        latest_data = self.tyre_data[self.current_tyre_choice][:, -1]
        b_no_nan = ~np.isnan(latest_data)
        latest_data = latest_data[b_no_nan]
        ages = ages[b_no_nan]

        def obj_fun(p):
            error = latest_data - self.forecast_tyre_grip(ages, p)
            weight = np.ones_like(error)
            weight[error < 0] = 10
            return np.mean(weight * error**2)

        p0 = [0, 1, 0, 1] if self.current_tyre_parameters is None else self.current_tyre_parameters
        res = minimize(obj_fun, p0, bounds=[(-100, 100), (0.8, 1.5), (-0.2, 0.2), (0.8, 1.3)], method='Powell')
        self.current_tyre_parameters = res.x


class WeatherTracker:

    def __init__(self):
        self.weather_data = []
        self.track_grips = []
        self.last_raining_move = -1000
        self.current_weather_state = None
        self.track_grip_model_x = None
        self.track_grip_model_y = None
        self.num_previous_steps = None
        self.num_future_steps = None

    def prepare_for_race(self):
        self.weather_data.append([])
        self.track_grips.append([])
        self.track_grip_model_y = None  # will need to refit these as won't have historic data for this race
        self.track_grip_model_x = None

    def update_weather_data(self, weather_state: WeatherState, move_number: int):
        if weather_state.rain_intensity > 0:
            self.last_raining_move = move_number
        self.current_weather_state = weather_state

    def record_action_result(
            self,
            previous_weather_state: WeatherState,
            expected_delta: float,
            new_speed: float,
            previous_speed: float):

        estimated_track_grip = None
        if expected_delta > 0:
            estimated_track_grip = (new_speed - previous_speed) / expected_delta
            estimated_track_grip = max(min(estimated_track_grip, 1), 0.1)
        elif len(self.track_grips[-1]) > 0:         # don't start with nans
            estimated_track_grip = np.nan

        if estimated_track_grip is not None:
            self.track_grips[-1].append(estimated_track_grip)
            self.weather_data[-1].append(WeatherTracker.weather_state_to_list(previous_weather_state))

    @staticmethod
    def weather_state_to_list(weather_state: WeatherState):
        return [weather_state.air_temperature, weather_state.track_temperature,
                weather_state.humidity, weather_state.rain_intensity]

    def forecast_track_grip(self, current_weather_state: WeatherState, num_future_steps=0):
        # Predict current grip + num_future_steps into the future
        # Returns an array of length 1 + num_future_steps
        if self.track_grip_model_y is None:
            self.fit_track_grip()
        if self.track_grip_model_y is None or current_weather_state is None:
            return np.ones(num_future_steps + 1)

        historic_x = np.array(self.weather_data[-1])
        historic_y = self.interpolate_nans(np.array(self.track_grips[-1]))

        b_nan = np.isnan(historic_y)
        if np.all(b_nan):
            return np.ones(num_future_steps + 1)
        elif b_nan[-1]:
            # Go back in time until we find a non-nan value
            i_last_no_nan = np.where(~b_nan)[0][-1]
            num_extra_steps = historic_y.size - i_last_no_nan - 1
            current_x = historic_x[i_last_no_nan + 1, :]
            historic_x = historic_x[:i_last_no_nan + 1, :]
            historic_y = historic_y[:i_last_no_nan + 1]
            b_nan = b_nan[:i_last_no_nan+1]

        else:
            current_x = np.array(self.weather_state_to_list(current_weather_state))
            num_extra_steps = 0
        historic_x = historic_x[~b_nan, :]     # cut off any nans at the start
        historic_y = historic_y[~b_nan]        # cut off any nans at the start

        grips = self.autoregressive_forecast(model_y=self.track_grip_model_y, model_x=self.track_grip_model_x,
                                             historic_x=historic_x, historic_y=historic_y, current_x=current_x,
                                             num_forecast_steps=num_future_steps + 1 + num_extra_steps, bound_y=True)
        return grips[num_extra_steps:]

    @staticmethod
    def interpolate_nans(track_grips):
        # Deal with nans - trim beginning and end but we have to interpolate missing values in the middle as we
        # need a continuous time series
        I = np.arange(len(track_grips))
        nans = np.isnan(track_grips)
        if np.sum(~nans) > 1:
            track_grips[nans] = PchipInterpolator(I[~nans], track_grips[~nans], extrapolate=False)(I[nans])
        # track_grips[nans] = np.interp(I[nans], I[~nans], track_grips[~nans], left=np.nan, right=np.nan)
        return track_grips

    def fit_track_grip(self):
        n = [np.sum(~np.isnan(grip)) for grip in self.track_grips]
        if len(n) == 0 or np.max(n) == 0 or n[-1] == 0:
            self.num_previous_steps = 0
            self.track_grip_model = None
            return
        num_previous_steps_predict = n[-1] - 1      # can only predict based on data from this race
        num_previous_steps_train = int(np.max(n)/10)          # leave plenty of data for training
        num_previous_steps = int(np.min([num_previous_steps_predict, num_previous_steps_train, 10]))

        y_inputs, y_targets, x_inputs, x_targets = [], [], [], []
        for data, grip in zip(self.weather_data, self.track_grips):     # loop over data from different races
            if len(grip) > num_previous_steps + 1:
                data = np.array(data)       # (N, D)
                grip = np.array(grip)       # {N, )

                # Deal with nans - trim beginning and end but we have to nterpolate missing values in the middle as we
                # need a continuous time series
                grip = self.interpolate_nans(grip)
                b_no_nan = ~np.isnan(grip)
                grip = grip[b_no_nan]
                data = data[b_no_nan, :]

                y_in, y_tar = self.format_ar_arrays_for_y(data, grip, num_previous_steps)
                x_in, x_tar = self.format_ar_arrays_for_x(data, grip, num_previous_steps)

                y_inputs.append(y_in)
                y_targets.append(y_tar)
                x_inputs.append(x_in)
                x_targets.append(x_tar)

        if len(y_inputs) > 0 and len(x_inputs) and len(np.vstack(x_inputs)):
            y_inputs_all = np.vstack(y_inputs)
            y_targets_all = np.vstack(y_targets)
            x_inputs_all = np.vstack(x_inputs)
            x_targets_all = np.vstack(x_targets)

            if self.track_grip_model_y is None:
                self.track_grip_model_y = LinearRegression()
            if self.track_grip_model_x is None:
                self.track_grip_model_x = LinearRegression()

            self.track_grip_model_y.fit(y_inputs_all, y_targets_all)
            self.track_grip_model_x.fit(x_inputs_all, x_targets_all)
            self.num_previous_steps = num_previous_steps

    @staticmethod
    def autoregressive_forecast(model_y, model_x, historic_x, historic_y, current_x, num_forecast_steps, bound_y=True,
                                bound_x=True):
        # Predict the y at the current time point and then for num_forecast_steps into the future by alternating y and
        # x predictions:
        #       predict y_t given x_t and {y_t-i, x_t-i} i = 1:num_previous_steps
        #       predict x_t+1 given [y_t, x_t] and {y_t-i, x_t-i} i = 1:num_previous_steps
        #
        # If bound_y is True then bounds y to be within (0, 1). If bound_x is True then bounds x to be within (0, 100)

        # Figure out the number of previous steps
        #   > number of model features = D*(num_previous_steps + 1) + num_previous_steps
        #   > num_previous_steps = (number of model features - D) / (D + 1)
        N, D = historic_x.shape
        num_previous_steps = int((model_y.n_features_in_ - D) / (D + 1))
        if historic_x.shape[0] < num_previous_steps:
            raise ValueError("'Refitting track grip model hasn't fixed it, something has gone wrong...")

        if current_x.ndim == 1:
            current_x = current_x[None, :]
        if historic_y.ndim == 1:
            historic_y = historic_y[:, None]

        # Add on current_x
        if 0 == num_previous_steps:
            X = current_x
            y = np.array([np.nan])
        else:
            X = np.vstack([historic_x[-num_previous_steps:, :], current_x])
            y = np.concatenate([historic_y[-num_previous_steps:], [[np.nan]]])             # current_y is unknown

        y_input, _ = WeatherTracker.format_ar_arrays_for_y(X, y, num_previous_steps)           # (1, [x_t, {y_t-i, x_t-i}])

        ys = np.zeros(num_forecast_steps)
        for i in range(num_forecast_steps):
            # Predict current y
            ys[i] = model_y.predict(y_input)
            if bound_y:
                ys[i] = max(min(ys[i], 1.), 0.)

            # Predict next x
            x_input = np.atleast_2d(np.insert(y_input, 0, ys[i]))   # input for x_t+1 has additonal y_t inserted at the start
            x_next = model_x.predict(x_input)
            if bound_x:
                x_next = np.minimum(np.maximum(x_next, 0), 100)

            # Update input for next y
            y_input = np.hstack([x_next, x_input[:, :-(D + 1)]])
        return ys

    @staticmethod
    def format_ar_arrays_for_y(X, y, num_previous_steps=0):
        # Format two arrays used to predict the current value of y given the current value of x and num_previous_steps
        # of x and y.
        # y_inputs will include x at the current time step, x at the num_previous_steps time points and y at the
        # num_previous_steps time points. Hence number of cols = D*(num_previous_steps + 1) + num_previous_steps
        #
        #   num_previous_steps = 2
        #       X = [ x_00, x_01, x_02 ]          y = [ y_0 ]
        #           [ x_10, x_11, x_12 ]              [ y_1 ]
        #           [ x_20, x_21, x_22 ]              [ y_2 ]
        #           [ x_30, x_31, x_32 ]              [ y_3 ]
        #           [ x_40, x_41, x_42 ]              [ y_4 ]
        #
        #                     |- current_x -|   |-------   previous_y, previous_x   -------|
        #       y_inputs =  [ x_20, x_21, x_22, y_1, x_10, x_11, x_12, y_0, x_00, x_01, x_02 ]
        #                   [ x_30, x_31, x_32, y_2, x_20, x_21, x_22, y_1, x_10, x_11, x_12 ]
        #                   [ x_40, x_41, x_42, y_3, x_30, x_31, x_32, y_2, x_20, x_21, x_22 ]
        #
        #       y_targets = [ y_2 ]
        #                   [ y_3 ]
        #                   [ y_4 ]

        N, D = X.shape
        y = y.ravel()
        y_inputs = np.zeros((N - num_previous_steps, D + (D + 1) * num_previous_steps))
        for i in range(num_previous_steps + 1):
            y_inputs[:, i * (D+1):i * (D+1) + D] = X[num_previous_steps - i:N-i, :]
        for i in range(num_previous_steps):
            y_inputs[:, D + i * (D + 1)] = y[num_previous_steps - i - 1:N - i - 1]

        y_targets = y[num_previous_steps:]


        return y_inputs, y_targets[:, None]

    @staticmethod
    def format_ar_arrays_for_x(X, y, num_previous_steps=0):
        # Format two arrays used to predict the next value of x given the current value of x and y and
        # num_previous_steps of x and y. Hence number of cols = D*(num_previous_steps + 1) + num_previous_steps
        #
        #   num_previous_steps = 2
        #       X = [ x_00, x_01, x_02 ]          y = [ y_0 ]
        #           [ x_10, x_11, x_12 ]              [ y_1 ]
        #           [ x_20, x_21, x_22 ]              [ y_2 ]
        #           [ x_30, x_31, x_32 ]              [ y_3 ]
        #           [ x_40, x_41, x_42 ]              [ y_4 ]
        #
        #                     |-- cur_y, cur_x --|   |-------   previous_y, previous_x   -------|
        #       x_inputs =  [ y_2, x_20, x_21, x_22, y_1, x_10, x_11, x_12, y_0, x_00, x_01, x_02 ]
        #                   [ y_3, x_30, x_31, x_32, y_2, x_20, x_21, x_22, y_1, x_10, x_11, x_12 ]
        #
        #       x_targets = [x_30, x_31, x_32]
        #                   [x_40, x_41, x_42]
        #
        # Note, x_inputs has one more dimension and one fewer data point than y_inputs


        N, D = X.shape
        if y.ndim == 1:
            y = y[:, None]
        yx = np.hstack([y, X])
        x_inputs = np.hstack([yx[num_previous_steps-i:N-i-1, :] for i in range(num_previous_steps+1)])
        x_targets = X[num_previous_steps + 1:, :]
        return x_inputs, x_targets

class GripTracker:

    def get_grip(
            self,
            tyre_tracker: TyreTracker,
            weather_tracker: WeatherTracker,
            car_state: Union[CarState, None] = None,
            turns_ahead=0,
            tyre_grip=None,
            tyre_age=None,
            weather_state=None,
            exclude_track=False):

        if car_state is not None:
            tyre_grip = car_state.tyre_grip
            tyre_age = car_state.tyre_age
        if tyre_age is None:
            tyre_age = tyre_tracker.current_tyre_age
        if tyre_grip is None:
            tyre_grip = tyre_tracker.get_measured_tyre_grip(tyre_age)

        # Tyre grip
        if 0 == turns_ahead:
            grip = max(tyre_grip, 0.1)
        else:
            ages = tyre_age + np.arange(1, turns_ahead + 1)
            future_grips = tyre_tracker.forecast_tyre_grip(ages)
            grip = np.maximum(np.concatenate([[tyre_grip], future_grips]), 0.1)

        # Track grip from rain
        if not exclude_track:
            if weather_state is None:
                weather_state = weather_tracker.current_weather_state      # used cached value
            track_grip = weather_tracker.forecast_track_grip(weather_state, turns_ahead)
            if 0 == turns_ahead:
                track_grip = float(track_grip)
            grip *= track_grip

        return np.maximum(grip, 0.1)             # if predicted grip drops too low we can get stuck


class ProDriver(Driver):
    def __init__(self, name, random_action_probability=0.5, random_action_decay=0.96,
                 min_random_action_probability=0.0, *args, **kwargs):

        super().__init__(name, *args, **kwargs)

        self.tyre_tracker = TyreTracker()
        self.weather_tracker = WeatherTracker()
        self.grip_tracker = GripTracker()

        self.random_action_probability = random_action_probability
        self.random_action_decay = random_action_decay
        self.min_random_action_probability = min_random_action_probability

        self.at_turn = False
        self.move_number = 0

        self.track_info = None
        self.straight_ends = []
        self.completed_straights = []
        self.box_box_box = False
        self.last_action_was_random = False
        self.target_speed_grips = None

        self.correct_turns = {}
        self.safety_car_speed = 150         # initialise it to be at a medium speed

        self.sl_data = {action: [] for action in Action.get_sl_actions()}
        self.drs_data = {action: [] for action in [Action.LightThrottle, Action.FullThrottle, Action.Continue]}
        self.end_of_straight_speed = 350        # initialising to something large
        self.lowest_crash_speed = 350
        self.target_speeds = 350 * np.ones(50)

    def update_after_race(self, correct_turns: Dict[Position, Action]):
        # Called after the race by RaceControl
        self.correct_turns.update(correct_turns)            # dictionary mapping Position -> TurnLeft or TurnRight

    def _update_safety_car(self, new_car_state: CarState, result: ActionResult):
        if result.safety_car_speed_exceeded:  # we ended up going too fast so safe speed must be below current speed
            if new_car_state.speed - 10 < self.safety_car_speed:
                self.safety_car_speed = new_car_state.speed - 10
                if self.print_info:
                    print(f'\tDecreasing estimate of safety car speed to {self.safety_car_speed: .1f}')
            elif self.print_info:
                print(f'Safety car speed estimate of {self.safety_car_speed: .1f} already below car speed of '
                      f'{new_car_state.speed: .1f}')

        else:  # our current speed is safe, so safety car speed must be higher
            if new_car_state.speed + 1 > self.safety_car_speed:
                self.safety_car_speed = new_car_state.speed + 1
                if self.print_info:
                    print(f'\tIncreasing estimate of safety car speed to {self.safety_car_speed: .1f}')

    def _choose_turn_direction(self, track_state: TrackState):
        # Check if we need to make a decision about which way to turn
        if track_state.distance_left > 0 and track_state.distance_right > 0:  # both options available, need to decide
            if len(self.correct_turns) > 0:
                # Find the closest turn we have seen previously and turn in the same direction
                distances = np.array([track_state.position.distance_to(turn_position)
                                      for turn_position in self.correct_turns])
                i_closest = np.argmin(distances)
                return list(self.correct_turns.values())[i_closest]

            else:  # First race, no data yet so choose randomly
                return driver_rng().choice([Action.TurnLeft, Action.TurnRight])

        elif track_state.distance_left > 0:  # only left turn
            return Action.TurnLeft

        else:
            return Action.TurnRight  # only right or dead-end

    def _choose_move_from_models(self, speed: float, target_speed: float, drs_active: bool, **kwargs):
        # Test each action to see which will get us closest to our target speed
        actions = Action.get_sl_actions()
        if 0 == speed:  # yes this is technically cheating but you can get stuck here with low grip so bending the rules
            actions = [Action.LightThrottle, Action.FullThrottle]
        next_speeds = np.array([self.estimate_next_speed(action, speed, drs_active, **kwargs) for action in actions])
        errors = next_speeds - target_speed    # difference between predicted next speed and target, +ve => above target

        # The target speed is the maximum safe speed so we want to be under the target if possible. This means we don't
        # necessarily want the action with the smallest error
        if np.any(errors <= 0):            # under or equal to the target speed
            errors[errors > 0] = np.inf    # at least one action gets us under the speed so ignore others even if close

        # Now we can choose the action with the smallest error score. At the start there will be multiple actions with
        # with the same score, so we will choose randomly from these
        min_error = np.min(errors ** 2)
        available_actions = [action for action, error in zip(actions, errors)
                             if np.abs(error ** 2 - min_error) < 1e-3]

        return driver_rng().choice(available_actions)

    def estimate_previous_speed(self, test_input_speeds: np.ndarray, test_output_speeds: np.ndarray, speed):
        errors = (test_output_speeds - speed)**2
        speeds_min_error = test_input_speeds[errors == np.min(errors)]
        return np.max(speeds_min_error)

    def get_data(self, action, drs_active=False):
        if drs_active and action in self.drs_data:
            return self.drs_data[action]
        else:
            return self.sl_data[action]

    def choose_tyres(self, track_info: TrackInfo) -> TyreChoice:
        self.track_info = track_info
        return self.tyre_tracker.choose_tyres(track_info)

    def prepare_for_race(self):
        self.move_number = 0
        self.straight_ends = []
        self.completed_straights = []
        self.weather_tracker.prepare_for_race()

    def choose_aero(self, track_info):
        return AeroSetup.Balanced

    def make_a_move(self, car_state: CarState, track_state: TrackState, weather_state: WeatherState) -> Action:
        self.move_number += 1
        if 1 == self.move_number:
            self.update_target_speeds(grips_up_straight=np.tile(car_state.tyre_grip, track_state.distance_ahead))

        # Store the tyre data
        self.tyre_tracker.update_tyre_data(car_state)

        # Weather
        self.weather_tracker.update_weather_data(weather_state, self.move_number)
        if weather_state.rain_intensity > 0:
            self.weather_tracker.fit_track_grip()
            self.update_target_speeds(track_state.distance_ahead, car_state, weather_state)

        # If we are at the end of the straight and it is not a dead end then choose the turn direction
        if track_state.distance_ahead == 0 and not (track_state.distance_left == 0 and track_state.distance_right == 0
                                                    and car_state.speed > 0):
            # Store this straight
            if track_state.position not in self.straight_ends and track_state.distance_behind > 0:
                self.straight_ends.append(track_state.position)
                self.completed_straights.append(track_state.distance_behind)

            self.at_turn = True
            return self._choose_turn_direction(track_state)

        # Update the target speeds at the start of a straight, to take tyre degradation into account. We could do this
        # every move but limit it to avoid slowing code down too much
        elif track_state.distance_ahead > 0 and self.at_turn:
            if weather_state.rain_intensity == 0:           # we will already have updated them otherwise
                self.update_target_speeds(track_state.distance_ahead, car_state, weather_state)
            self.box_box_box = self.tyre_tracker.should_we_change_tyres(self)

            self.drs_was_active = False         # start of new straight, reset log

        self.at_turn = False

        # Are we changing tyres?
        if self.box_box_box and car_state.speed == 0:
            self.box_box_box = False
            return Action.ChangeTyres

        # Get the target speed
        target_speed = self._get_target_speed(track_state.distance_ahead, track_state.safety_car_active)

        # Get the current grip level
        current_grip = self.grip_tracker.get_grip(
            self.tyre_tracker, self.weather_tracker, car_state, turns_ahead=0, weather_state=weather_state)

        # Choose action that gets us closest to target, or choose randomly
        if driver_rng().rand() > self.random_action_probability:
            action = self._choose_move_from_models(car_state.speed, target_speed, car_state.drs_active,
                                                   grip_multiplier=current_grip)
            self.last_action_was_random = False
        else:
            action = driver_rng().choice(Action.get_sl_actions())
            self.last_action_was_random = True

        # If DRS is available then need to decide whether to open DRS or not.
        if track_state.drs_available and not car_state.drs_active and track_state.distance_ahead > 0:
            # Simulate the straight with and without DRS and check which we think will be faster
            time_no_drs, targets_broken_no_drs, *_ = self.simulate_straight(
                car_state.speed,
                track_state.distance_ahead,
                drs_active=False,
                safety_car_active=track_state.safety_car_active,
                weather_state=weather_state)
            time_drs, targets_broken_drs, *_ = self.simulate_straight(
                car_state.speed,
                track_state.distance_ahead - 1,
                drs_active=True,
                safety_car_active=track_state.safety_car_active,
                weather_state=weather_state)
            targets_broken_drs |= car_state.speed > self.target_speeds[track_state.distance_ahead - 1]
            time_drs = (1 / (car_state.speed + 1)) + time_drs

            if (time_drs < time_no_drs or driver_rng().rand() < self.random_action_probability
                or any(len(data) < 10 for data in self.drs_data.values())) and not targets_broken_drs:
                action = Action.OpenDRS
                self.drs_was_active = True

        self.random_action_probability = max(self.random_action_probability * self.random_action_decay,
                                             self.min_random_action_probability)

        return action

    def _get_target_speed(self, distance_ahead, safety_car_active, target_speeds=None):
        if target_speeds is None:
            target_speeds = self.target_speeds

        if distance_ahead == 0 or self.box_box_box:
            target_speed = 0                                            # dead end - need to stop!!
        else:
            target_speed = target_speeds[distance_ahead - 1]       # target for next step

        if safety_car_active:
            target_speed = min(target_speed, self.safety_car_speed)

        return target_speed

    def estimate_next_speed(self, action: Action, speed, drs_active: bool, grip_multiplier: float = 1.0):
        data = np.array(self.get_data(action, drs_active))
        if data.shape[0] < 2:
            return speed
        interp = interp1d(data[:, 0], data[:, 1], fill_value='extrapolate', assume_sorted=False)
        return speed + interp(speed) * grip_multiplier      # predict delta

    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
                                   action: Action, new_car_state: CarState, new_track_state: TrackState,
                                   result: ActionResult, previous_weather_state: WeatherState):

        if previous_track_state.safety_car_active:
            self._update_safety_car(previous_car_state, result)

        if previous_track_state.distance_ahead == 0:          # end of straight
            if result.crashed or result.spun:
                grip = self.grip_tracker.get_grip(
                    self.tyre_tracker,
                    self.weather_tracker,
                    previous_car_state,
                    weather_state=previous_weather_state)

                self.end_of_straight_speed = min(self.end_of_straight_speed,
                                                 previous_car_state.speed / grip - 10)
                if previous_track_state.distance_left > 0 or previous_track_state.distance_right > 0:
                    self.lowest_crash_speed = min(previous_car_state.speed / grip, self.lowest_crash_speed)
            else:
                previous_grip = self.grip_tracker.get_grip(
                    self.tyre_tracker,
                    self.weather_tracker,
                    previous_car_state,
                    weather_state=previous_weather_state)
                self.end_of_straight_speed = min(max(self.end_of_straight_speed,
                                                 (previous_car_state.speed / previous_grip) + 1),
                                                 self.lowest_crash_speed)

            # Refit tyre model now we have more data
            self.tyre_tracker.fit_tyre_model()
            self.weather_tracker.fit_track_grip()

        # record the change in speed resulting from the action we took
        elif action in self.sl_data and self.move_number > self.weather_tracker.last_raining_move + 30:
            # Remove the grip effect from the delta to get the true dynamics
            if Action.HeavyBrake == action and 0 == new_car_state.speed:
                # Heavy braking delta can take the car "below" zero, which is then capped at 0. When this happens we
                # don't see the effect of the grip multiplier so we don't want to normalise the delta
                normalised_delta = (new_car_state.speed - previous_car_state.speed)
            else:
                normalised_delta = (new_car_state.speed - previous_car_state.speed) / self.grip_tracker.get_grip(
                    self.tyre_tracker,
                    self.weather_tracker,
                    previous_car_state,
                    exclude_track=True)
            normalised_delta = max(normalised_delta, -previous_car_state.speed)   # can't go below zero

            # Record the point if it is not on top of another point (interpolation doesn't like points too close
            # together in x, plus it is also a bit unnecessary) and we are below 200 points (just to keep code
            # performance up)
            current_data = self.get_data(action, previous_car_state.drs_active)
            if 0 == len(current_data):
                closest_distance = 1000
            else:
                closest_distance = np.min(np.abs(np.array(current_data)[:, 0] - previous_car_state.speed))
            if closest_distance > 1 and len(current_data) < 200 and previous_car_state.speed + normalised_delta > 0:
                new_data = [previous_car_state.speed, normalised_delta]
                current_data.append(new_data)

        # Record the track grip and weather data
        if action in self.sl_data:
            expected_delta = self.estimate_next_speed(
                action,
                previous_car_state.speed,
                previous_car_state.drs_active,
                self.grip_tracker.get_grip(
                    self.tyre_tracker,
                    self.weather_tracker,
                    previous_car_state,
                    exclude_track=True)) - previous_car_state.speed
        else:
            expected_delta = 0

        self.weather_tracker.record_action_result(
            previous_weather_state,
            expected_delta,
            new_car_state.speed,
            previous_car_state.speed)

    def update_target_speeds(self, distance_ahead=None, car_state=None, weather_state=None, grips_up_straight=None,
                             assign_to_self=True):
        """ Either need to specify distance_ahead, car_state, and weather_state or a custom set of grips_up_straight """
        if grips_up_straight is None:
            grips = self.grip_tracker.get_grip(
                self.tyre_tracker,
                self.weather_tracker,
                car_state,
                distance_ahead,
                weather_state=weather_state)  # from start to end of straight
            grips *= 0.95                                       # add a little safety margin to our prediction
            grips_up_straight = np.atleast_1d(grips)[::-1]

        previous_targets = np.copy(self.target_speeds)
        target_speeds = np.zeros_like(self.target_speeds)
        speed = self.end_of_straight_speed * grips_up_straight[0]       # modify by expected grip at end of straight

        # Pre compute the changes in speed for quicker look up. As the grip changes down the straight we compute deltas
        # with grip = 1 and then convert to next speeds later
        test_input_speeds = np.linspace(0, 350, 351)
        test_speed_deltas = {action: self.estimate_next_speed(action, test_input_speeds, False, grip_multiplier=1)
                                     - test_input_speeds
                             for action in self.sl_data}

        for i in range(len(self.target_speeds)):
            if np.all(target_speeds[i:] == speed):
                break                   # there won't be any further changes so save some time
            target_speeds[i] = speed
            previous_grip = grips_up_straight[min(i + 1, len(grips_up_straight) - 1)]
            speed = np.nanmax([self.estimate_previous_speed(test_input_speeds,
                                                            test_input_speeds + test_speed_deltas[action]*previous_grip,
                                                            speed)
                               for action in self.sl_data])
            speed = max(speed, 10)

        if assign_to_self:
            self.target_speeds = target_speeds
            self.target_speed_grips = grips_up_straight

        return target_speeds

    def simulate_straight(self, speed, distance_ahead, drs_active, safety_car_active, weather_state=None, grips=None):
        if grips is None:
            grips = self.grip_tracker.get_grip(
                self.tyre_tracker,
                self.weather_tracker,
                car_state=None,
                turns_ahead=distance_ahead,
                tyre_grip=self.tyre_tracker.current_tyre_grip,
                tyre_age=self.tyre_tracker.current_tyre_age,
                weather_state=weather_state)
            target_speeds = self.target_speeds
        else:
            # Custom grips provided, need to get custom target speeds to match them
            target_speeds = self.update_target_speeds(grips_up_straight=grips[::-1], assign_to_self=False)

        speeds = np.zeros(distance_ahead)
        break_target_speed = False
        for d in range(distance_ahead):
            target_speed = self._get_target_speed(distance_ahead - d, safety_car_active, target_speeds)
            action = self._choose_move_from_models(speed, target_speed, drs_active, grip_multiplier=grips[d])
            speeds[d] = self.estimate_next_speed(action, speed, drs_active, grip_multiplier=grips[d])
            speed = speeds[d]
            break_target_speed |= speed > target_speed
        time = np.sum(1 / (speeds + 1))

        return time, break_target_speed, speeds, grips

    def simulate_to_end_of_race(self, start_tyre_age):
        straights_remaining = max(self.track_info.number_of_straights - len(self.straight_ends), 1)
        straight_length = int(self.track_info.average_straight)
        total_length = straights_remaining * (straight_length + 1)        # +1 as we take move turning each straight
        #            |- total # squares to move through -|   |- total # of turns --|
        num_moves = straights_remaining * straight_length + (straights_remaining - 1)

        # grips[i] is grip at time point i used to move from speed[i] to speed[i+1]
        grips = self.grip_tracker.get_grip(
            self.tyre_tracker,
            self.weather_tracker,
            turns_ahead=num_moves,
            tyre_age=start_tyre_age,
            exclude_track=True)
        speeds = 500 * np.ones(num_moves + 1)     # add fake corner in at the end as this driver brakes for the finish
        speeds[0] = self.end_of_straight_speed * grips[0]       # should really be previous grip but close enough
        # speed of turns
        speeds[straight_length::(straight_length+1)] = self.end_of_straight_speed * grips[straight_length::(
                straight_length+1)]
        # Speed starting next straight same as turn
        speeds[straight_length+1::(straight_length+1)] = speeds[straight_length:-2:(straight_length+1)]

        # Pre compute the changes in speed for quicker look up. As the grip changes down the straight we compute deltas
        # with grip = 1 and then convert to next speeds later
        test_input_speeds = np.linspace(0, 350, 351)
        actions = list(self.sl_data.keys())
        test_speed_deltas = np.zeros((test_input_speeds.size, len(actions)))
        for i, action in enumerate(actions):
            test_speed_deltas[:, i] = self.estimate_next_speed(action, test_input_speeds, False, grip_multiplier=1) \
                                      - test_input_speeds

        # First the backwards pass to compute the maximum safe speeds
        for i in range(total_length - 1):
            possible_next_speeds = np.maximum(test_input_speeds[:, None] + test_speed_deltas* grips[-i-2], 0)
            safe_next_speeds = test_input_speeds[np.any(possible_next_speeds <= speeds[-i-1], axis=1)]
            max_safe_input_speed = np.max(safe_next_speeds)
            speeds[-i-2] = np.minimum(max_safe_input_speed, speeds[-i-2])

        # Next the forward pass to work out what we can actually reach
        for i in range(num_moves):
            next_speeds = np.maximum(speeds[i] + grips[i] * test_speed_deltas[int(np.round(speeds[i])), :], 0)
            if np.any(next_speeds < speeds[i+1]):
                speeds[i+1] = np.max(next_speeds * (next_speeds < speeds[i+1]))
            else:
                speeds[i+1] = np.min(next_speeds)

        # Compute time
        time = np.sum(1 / (1 + speeds[:-1]))
        return time
