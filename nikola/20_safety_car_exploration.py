from imports import *


if __name__ == '__main__':

    safety_car = SafetyCar()

    safety_car_speeds = [next(safety_car.safety_car_speeds) for _ in range(2)]
    while safety_car_speeds[0] != safety_car_speeds[-1]:
        safety_car_speeds.append(next(safety_car.safety_car_speeds))

    print('Safety car speeds that are cycled through')
    print(safety_car_speeds)

    sorted_speeds = sorted(safety_car_speeds[:-1])
    differences = [sorted_speeds[i + 1] - sorted_speeds[i] for i in range(len(sorted_speeds) - 1)]
    print('Smallest difference between speeds: {}'.format(min(differences)))

    fig, ax = plt.subplots()
    ax.set_title('Safety car speed cycle')
    ax.set_ylabel('Speed [km/h]')
    ax.plot(safety_car_speeds)
    ax.set_ylim(0.0, 1.1 * max(safety_car_speeds))
    plt.show()
