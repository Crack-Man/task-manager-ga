import numpy as np
import random as rnd


class GA2:
    def random_spec(self):
        return np.random.randint(1, self.m, self.n)

    def __init__(self,
                 size_population,
                 n,
                 m,
                 o_n,
                 t_n,
                 k_m,
                 max_len_individual,
                 p_mutation_ind=0.5,
                 p_mutation_gen=0.15,
                 size_selection=1):
        self.size_population = size_population
        self.n = n
        self.m = m
        self.o_n = o_n
        self.t_n = t_n
        self.k_m = k_m
        self.size_selection = size_selection
        self.max_len_individual = max_len_individual
        self.p_mutation_ind = p_mutation_ind
        self.p_mutation_gen = p_mutation_gen
        self.best_p = None
        self.best_rate = None
        self.rng = np.random.default_rng()
        self.population = np.array([self.random_spec() for _ in range(self.size_population)])
        self.selected = []
        self.childs = []
        self.fit_rates = np.array([])

    def fit(self, s):
        times = [0 for _ in range(self.m)]
        for ss, oo, tt in zip(s, self.o_n, self.t_n):
            times[ss - 1] += self.k_m[ss - 1, oo - 1] * tt
        return max(times)

    def sort(self):
        idx = np.argsort(self.fit_rates)
        return self.fit_rates[idx], self.population[idx]

    def selection(self):
        self.fit_rates = np.array([self.fit(s) for s in self.population])
        fit_rates_sort, p1 = self.sort()
        self.best_p = p1[0]
        self.best_rate = fit_rates_sort[0]
        self.selected = p1[:self.size_selection]

    def crossover(self):
        new_count = self.size_population - self.size_selection
        parent1 = self.rng.integers(0, self.size_selection, size=new_count)
        parent2 = (self.rng.integers(1, self.size_selection, size=new_count) + parent1) % self.size_selection

        point = self.rng.integers(1, self.max_len_individual - 1, size=new_count)
        a = np.arange(self.max_len_individual)[None] <= point[..., None]
        b = self.selected[parent1]
        self.childs = np.where(
            np.arange(self.max_len_individual)[None] <= point[..., None],
            self.selected[parent1],
            self.selected[parent2]
        )

    def mutation(self):
        mut_childs_mask = self.rng.choice(2, p=(1 - self.p_mutation_ind, self.p_mutation_ind),
                                          size=len(self.childs)) > 0
        mut_childs = self.rng.integers(0, 4, size=(mut_childs_mask.sum(), self.max_len_individual))
        gen_childs_mask = self.rng.random(size=mut_childs.shape) <= self.p_mutation_gen
        self.childs[mut_childs_mask] = np.where(gen_childs_mask, mut_childs, self.childs[mut_childs_mask])

    def step(self):
        self.selection()
        self.crossover()
        self.mutation()
        self.population = np.concatenate([self.selected, self.childs], axis=0)
        return self.best_rate, self.best_p
