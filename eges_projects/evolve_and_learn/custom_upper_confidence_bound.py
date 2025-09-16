import numpy as np
from bayes_opt import TargetSpace
from bayes_opt.acquisition import UpperConfidenceBound
from numpy.typing import NDArray
from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor

Float = np.floating[Any]


class CustomUpperConfidenceBound(UpperConfidenceBound):

    def suggest(
            self,
            gp: GaussianProcessRegressor,
            target_space: TargetSpace,
            n_random: int = 10_000,
            n_l_bfgs_b: int = 10,
            fit_gp: bool = True,
    ) -> NDArray[Float]:
        return super().suggest(gp, target_space, n_random, 1, fit_gp)