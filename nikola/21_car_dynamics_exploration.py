import sys
import os

from imports import *


def plot(speed_time, name):
    fig, ax = plt.subplots()
    ax.plot(speed_time)
    ax.set_xlabel('Time')
    ax.set_ylabel('Speed')
    ax.set_xlim((0, ax.get_xlim()[1]))
    ax.set_ylim((0, 400.0))
    ax.grid()
    plt.savefig(name)
    plt.close()


if __name__ == '__main__':

    path = sys.argv[0][:-3]
    if not os.path.exists(path):
        os.mkdir(path)

    dynamics = CarDynamicsModel1(300)

    # Acceleration through time
    for drs_active in [False, True]:

        # Full throttle
        duration = 15
        speed_time = np.zeros((duration,))
        for i in range(1, duration):
            speed_time[i] = dynamics.full_throttle(speed_time[i - 1], drs_active=drs_active)
        plot(speed_time, os.path.join(path, 'full_throttle_drs_{}.png'.format(drs_active)))

        # Light throttle
        duration = 25
        speed_time = np.zeros((duration,))
        for i in range(1, duration):
            speed_time[i] = dynamics.light_throttle(speed_time[i - 1], drs_active=drs_active)
        plot(speed_time, os.path.join(path, 'light_throttle_drs_{}.png'.format(drs_active)))

        # Heavy brake
        duration = 8
        speed_time = np.ones((duration,)) * dynamics.top_speed(0)
        for i in range(1, duration):
            speed_time[i] = dynamics.heavy_brake(speed_time[i - 1], drs_active=drs_active)
        plot(speed_time, os.path.join(path, 'heavy_brake_drs_{}.png'.format(drs_active)))

        # Light brake
        duration = 40
        speed_time = np.ones((duration,)) * dynamics.top_speed(0, drs_active=drs_active)
        for i in range(1, duration):
            speed_time[i] = dynamics.light_brake(speed_time[i - 1])
        plot(speed_time, os.path.join(path, 'light_brake_drs_{}.png'.format(drs_active)))

    # Braking to stop
    for speed in range(100):
        light_brake_speed = dynamics.light_brake(speed)
        heavy_brake_speed = dynamics.heavy_brake(speed)
        print('{:3.2f} {:3.2f} {:3.2f}'.format(speed, light_brake_speed, heavy_brake_speed))
