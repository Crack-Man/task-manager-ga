from ga import GA
import numpy as np

with open("input.txt", "r") as f:
    n = int(f.readline())
    o_n = np.array(list(map(int, f.readline().split())))
    t_n = np.array(list(map(float, f.readline().split())))
    m = int(f.readline())

    k_m = []
    for _ in range(m):
        k_m.append(list(map(float, f.readline().split())))
    k_m = np.array(k_m)

popsize = 20
ga = GA(popsize, n, m, o_n, t_n, k_m, k_select=3, selection_type="tour")
# ga = GA(popsize, n, m, o_n, t_n, k_m, k_select=3, selection_type="roulette")
# ga = GA(popsize, n, m, o_n, t_n, k_m, k_select=3, selection_type="range")
step = None
for i in range(int(1e3)):
    step = ga.step()
# if not i % 100:
    #     print(*step)
print(*step)