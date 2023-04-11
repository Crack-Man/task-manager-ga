import numpy as np

class GeneticAlgorithm:

    def __init__(self,
                 dots: np.ndarray,
                 size: int,
                 max_len_individual: int,
                 size_population: int,
                 size_selection: int,
                 p_mutation_ind: float,
                 p_mutation_gen: float
                 ):
        self.dots = dots  # точки для посещения
        self.size = size  # размер поля
        self.max_len_individual = max_len_individual  # количество шагов который делает каждая особь
        self.size_population = size_population  # количество особей
        self.size_selection = size_selection  # выживаемые особи во время селекции
        self.p_mutation_ind = p_mutation_ind  # вероятность мутации детей
        self.p_mutation_gen = p_mutation_gen  # вероятность мутации генов
        self.rng = np.random.default_rng()  # генератор случайных элементов
        self.population = self.rng.integers(0, 4, size=(self.size_population, self.max_len_individual))

    directions = np.array(((0, 1), (0, -1), (1, 0), (-1, 0)))

    def get_paths(self) -> tuple[np.ndarray, np.ndarray]:
        paths = np.full((self.size_population, self.max_len_individual + 1, 2), self.size // 2)
        visited = np.full((self.size_population, self.size, self.size), -1)  # матрица посещений
        visited[:, self.size // 2, self.size // 2] = 0  # посещение всех центральных ячеек
        idx = np.arange(self.size_population)

        for i in range(self.max_len_individual):
            paths[:, i + 1] = (paths[:, i] + self.directions[self.population[:, i]]) % self.size
            ind = np.ravel_multi_index((idx, paths[:, i + 1, 0], paths[:, i + 1, 1]), visited.shape)
            values = np.ravel(visited)[ind]
            np.ravel(visited)[ind] = np.where(values >= 0, values, i + 1)
        return paths, visited

    def fitness(self) -> np.ndarray:
        _, visited = self.get_paths()
        result = visited[:, self.dots[:, 0], self.dots[:, 1]]
        count = (result >= 0).sum(axis=1)
        steps = result.max(axis=1)
        # steps = np.where(steps == -1, self.max_len_individual, steps)
        steps[steps == -1] = self.max_len_individual

        return count * self.max_len_individual - steps

    def selection(self) -> None:
        self.selected = self.population[np.argsort(self.fitness())[-self.size_selection:]]

    def crossover(self) -> None:
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

    def mutation(self) -> None:
        mut_childs_mask = self.rng.choice(2, p=(1 - self.p_mutation_ind, self.p_mutation_ind),
                                          size=len(self.childs)) > 0
        mut_childs = self.rng.integers(0, 4, size=(mut_childs_mask.sum(), self.max_len_individual))
        gen_childs_mask = self.rng.random(size=mut_childs.shape) <= self.p_mutation_gen
        self.childs[mut_childs_mask] = np.where(gen_childs_mask, mut_childs, self.childs[mut_childs_mask])

    def step(self) -> None:
        self.selection()
        self.crossover()
        self.mutation()
        self.population = np.concatenate([self.selected, self.childs], axis=0)

width, height = 11, 11

field = np.zeros((height, width))
dots = np.array([(1, 1), (1, 9), (7, 3), (9, 8), (3, 5)])

field[dots[:, 0], dots[:, 1]] = 1

ga = GeneticAlgorithm(dots, width, 100, 200, 20, 0.5, 0.15)
for i in range(1000):
    ga.step()