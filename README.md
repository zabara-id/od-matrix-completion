# od-matrix-completion

Установка [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)-окружения:
```bash
uv python install 3.13
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Установка проекта:
```bash
uv sync
```

Добавление новых зависимостей:
```bash
uv add package_name
```

Если добавил не то:
```bash
uv remove package_name
```

## Пример использования (линейная постановка)

Минимальный пример, который показывает, как создать задачу и запустить простой оптимизатор-«заглушку»:

```python
import numpy as np
from od_matrix_completion.core.problem import Problem
from od_matrix_completion.core.base import BaseOptimizer, OptimizationResult


class DummyOptimizer(BaseOptimizer):
    def fit(self, problem: Problem, x0=None, callback=None) -> OptimizationResult:
        # Возьмём начальное приближение и сразу его вернём
        D0 = problem.initial_guess()
        d0 = D0.reshape(-1)
        return OptimizationResult(
            D=D0,
            d=d0,
            objective=problem.objective(D0),
            n_iters=0,
            converged=True,
        )


n_z = 2
A = np.eye(n_z * n_z)  # игрушечная маршрутизация
f = np.array([1.0, 2.0, 3.0, 4.0])
D_hat = np.ones((n_z, n_z))

prob = Problem(A=A, f_obs=f, D_hat=D_hat, gamma=0.1)
res = prob.solve(algorithm=DummyOptimizer())
print(res.D)
```

### Субградиентный спуск

Готовый простой метод: `SubgradientDescent`. Он делает (суб)градиентный шаг по `d = vec(D)`,
затем проецирует `D` на неотрицательность и (опционально) на маргиналии через IPF.

```python
import numpy as np
from od_matrix_completion.core import Problem, SubgradientDescent

n_z = 3
A = np.eye(n_z * n_z)
f = np.arange(1, n_z * n_z + 1, dtype=float)
L = np.array([3.0, 6.0, 12.0])
W = np.array([7.0, 7.0, 7.0])
D_hat = np.ones((n_z, n_z))

prob = Problem(A=A, f_obs=f, D_hat=D_hat, L=L, W=W, gamma=0.05)
opt = SubgradientDescent(step_size=1e-2, schedule="sqrt", project_marginals=True, ipf_iters=2, max_iters=500)
res = prob.solve(algorithm=opt)
print("objective=", res.objective, "iters=", res.n_iters)
print(res.D)
```

### Проксимальный (KL) спуск

Алгоритм `KLProximalDescent` выполняет мультипликативный апдейт в KL‑геометрии
и хорошо сочетается с IPF‑проекцией маргиналий. При `gamma=0` он вырождается в
классический exponentiated gradient.

```python
import numpy as np
from od_matrix_completion.core import Problem, KLProximalDescent

n_z = 3
A = np.eye(n_z * n_z)
f = np.arange(1, n_z * n_z + 1, dtype=float)
D_hat = np.ones((n_z, n_z))
L = np.array([3.0, 6.0, 12.0])
W = np.array([7.0, 7.0, 7.0])

prob = Problem(A=A, f_obs=f, D_hat=D_hat, L=L, W=W, gamma=0.1)
opt = KLProximalDescent(step_size=1e-2, schedule="sqrt", project_marginals=True, ipf_iters=2, max_iters=500)
res = prob.solve(algorithm=opt)
print("objective=", res.objective, "iters=", res.n_iters)
print(res.D)
```
