import matplotlib.pyplot as plt
import parse
import numpy as np

filename = "2022_1022_2257"

with open(f"./src/{filename}.txt") as file:
    rawtxt = file.readlines()
    lines = [line.strip() for line in rawtxt]
    result_x = []
    result_y = []

    fig, ax = plt.subplots()

    cnt = 0
    xsum = 0.0
    ysum = 0.0
    for line in lines:
        cnt += 1
        if cnt % 4 == 0:
            x, y = parse.parse("1st / 2nd: {:f}, {:f}", line)
            xsum += x
            ysum += y
        
        if cnt % (82 * 4) == 0:
            result_x.append(xsum / 82.0)
            result_y.append(ysum / 82.0)
            xsum = 0.0
            ysum = 0.0

    t = np.arange(1, len(result_x) + 1)

    ax.set_xlabel("epoch")
    ax.set_ylabel("predict error(m/s) / actual diff(m/s)")

    ax.plot(t, result_x, label="vert")
    ax.plot(t, result_y, label="hori")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"./src/{filename}.png")
    