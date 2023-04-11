import numpy as np
import random as rnd

class GA:
    def random_spec(self):
        return np.random.randint(1, self.m, self.n)

    def __init__(self, popsize, n, m, o_n, t_n, k_m, k_select=1, selection_type="tour", selection_step=3):
        self.popsize = popsize
        self.n = n
        self.m = m
        self.o_n = o_n
        self.t_n = t_n
        self.k_m = k_m
        self.k_select = k_select
        self.best_p = None
        self.best_rate = None
        self.selection_type = selection_type
        self.selection_step = selection_step
        self.p = np.array([self.random_spec() for _ in range(self.popsize)])

    def fit(self, s):
        times = [0 for _ in range(self.m)]
        for ss, oo, tt in zip(s, self.o_n, self.t_n):
            times[ss - 1] += self.k_m[ss - 1, oo - 1] * tt
        return max(times)

    def two_point_crossover(self, i, j):
        arange = np.arange(self.n)
        np.random.shuffle(arange)
        first, second = arange[0:2]
        first, second = min(first, second), max(first, second)
        if np.random.random() < 0.5:
            return np.concatenate([self.p[i, :first + 1], self.p[j, first + 1:second], self.p[i, second:]])
        return np.concatenate([self.p[j, :first + 1], self.p[i, first + 1:second], self.p[j, second:]])

    def crossover(self, i, j):
        t = np.random.randint(0, self.m)
        return np.concatenate([self.p[i, :t], self.p[j, t:]])

    def selection_roulette(self, fit_rates, p1):
        weights = fit_rates / np.sum(fit_rates)
        idxs = np.arange(0, self.popsize)
        choice = np.random.choice(idxs, self.k_select, p=weights)
        return p1[choice]

    def selection_range(self, p1):
        a = 1.5
        b = 2 - a
        p = [1 / self.popsize * (a - (a - b) * (i - 1) / (self.popsize - 1)) for i in range(self.popsize)]
        idxs = np.arange(0, self.popsize)
        choice = rnd.choices(population=idxs, weights=p, k=self.k_select)
        return p1[choice]

    def selection_sus(self, fit_rates, p1):
        weights = fit_rates / np.sum(fit_rates)
        idxs = np.arange(0, self.popsize)
        choice = np.random.choice(idxs, 1, p=weights)
        for i in range(self.k_select - 1):
            choice = np.append(choice, (choice[0] + (i + 1) * self.selection_step) % self.popsize)
        return p1[choice]

    def selection_scaling(self, fit_rates, p1):
        rng = [30, 100]
        a = (rng[0] - rng[1]) / (fit_rates[-1] - fit_rates[0])
        b = rng[0] - a * fit_rates[-1]
        weights = a * fit_rates + b
        idxs = np.arange(0, self.popsize)
        choice = rnd.choices(population=idxs, weights=weights, k=self.k_select)
        return p1[choice]

    def sort(self, a, b):
        idx = np.argsort(a)
        return a[idx], b[idx]

    def step(self):
        # Find best
        fit_rates = []
        for s in self.p:
            fit_rates.append(self.fit(s))

        fit_rates_sort, p1 = self.sort(np.array(fit_rates), self.p)
        self.best_p = p1[0]
        self.best_rate = fit_rates_sort[0]

        # Selection
        if self.selection_type == "tour":
            p1 = list(p1[:self.k_select])
        elif self.selection_type == "roulette":
            p1 = list(self.selection_roulette(fit_rates_sort, p1))
        elif self.selection_type == "sus":
            p1 = list(self.selection_sus(fit_rates_sort, p1))
        elif self.selection_type == "range":
            p1 = list(self.selection_range(p1))
        elif self.selection_type == "scaling":
            p1 = list(self.selection_scaling(fit_rates_sort, p1))

        # Crossover
        for _ in range(self.popsize - self.k_select):
            i, j = np.random.choice(np.arange(0, self.popsize), 2)
            p1.append(self.two_point_crossover(i, j))
        self.p = np.array(p1)

        # Mutation
        for i in range(self.k_select, self.popsize):
            if np.random.random() < 0.1:
                self.p[i] = self.random_spec()

        return self.best_rate, self.best_p
