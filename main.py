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

popsize = 50
k_select = 30

# selection_type = "tour"
# selection_type = "roulette"
selection_type = "range"
# selection_type = "sus"
# selection_type = "scaling"

ga = GA(popsize, n, m, o_n, t_n, k_m, k_select=k_select, selection_type=selection_type)

step = None
for i in range(int(1e3)):
    step = ga.step()
print(step[0])
with open("output.txt", "w") as f:
    print(*step[1], file=f)
