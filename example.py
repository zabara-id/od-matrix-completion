import numpy as np
from src.od_matrix_completion.core import Problem, SubgradientDescent

# Ввод данных
# TODO: заменить эту игрушечную синтетику на реальные данные
n_z = 3
A = np.eye(n_z * n_z)
f = np.arange(1, n_z * n_z + 1, dtype=float)
D_hat = np.ones((n_z, n_z))
L = np.array([3.0, 6.0, 12.0])
W = np.array([7.0, 7.0, 7.0])

# Класс для решения задачи
problem = Problem(
    model = "linear",
    A=A, f_obs=f, D_hat=D_hat, L=L, W=W, gamma=1e-3
)

# Алгоритм оптимизации
optimizer = SubgradientDescent(
    step_size=1e-2,
    schedule="linear",
    project_marginals=True,
    ipf_iters=2,
    max_iters=500,
    verbose=True
)

# Решение задачи
res = problem.solve(algorithm=optimizer)

print("objective=", res.objective, "iters=", res.n_iters)
print(res.D)
