from __future__ import annotations
from bayes_opt import BayesianOptimization

from sklearn.gaussian_process.kernels import Matern

from collections.abc import Callable, Mapping

from numpy.random import RandomState
from scipy.optimize import NonlinearConstraint

from bayes_opt.acquisition import AcquisitionFunction
from bayes_opt.domain_reduction import DomainTransformer

from custom_gaussian_process_regressor import CustomGaussianProcessRegressor


class CustomBayesianOptimization(BayesianOptimization):
    def __init__(
            self,
            f: Callable[..., float] | None,
            pbounds: Mapping[str, tuple[float, float]],
            acquisition_function: AcquisitionFunction | None = None,
            constraint: NonlinearConstraint | None = None,
            random_state: int | RandomState | None = None,
            verbose: int = 2,
            bounds_transformer: DomainTransformer | None = None,
            allow_duplicate_points: bool = False,
            coefficients: list = None,
            intercept: float = None,
            old_mean: float = None,
            old_std: float = None,
    ):
        super().__init__(f, pbounds, acquisition_function, constraint, random_state, verbose, bounds_transformer, allow_duplicate_points)
        self._gp = CustomGaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
            coefficients=coefficients,
            intercept=intercept,
            old_mean=old_mean,
            old_std=old_std,
        )
