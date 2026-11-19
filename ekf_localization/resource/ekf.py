import numpy as np
from numpy.linalg import inv


class RobotEKF:
    """EKF from the lectures. Changes vs the lecture version:
       1) predict(): Jacobians are evaluated at the PREVIOUS mean (before the mean is propagated)
       2) update(): works for scalar / 1-element measurements (needed by the IMU update, Task 2)
    """

    def __init__(self, dim_x=1, dim_u=1, eval_gux=None, eval_Gt=None, eval_Vt=None):
        self.mu = np.zeros((dim_x))      # mean state estimate
        self.Sigma = np.eye(dim_x)       # covariance state estimate
        self.Mt = np.eye(dim_u)          # process noise (in command space)

        self.eval_gux = eval_gux
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt

        self._I = np.eye(dim_x)

    def predict(self, u, sigma_u, g_extra_args=()):
        args = (*self.mu, *u)                                  # previous mean
        Gt = self.eval_Gt(*args, *g_extra_args)
        Vt = self.eval_Vt(*args, *g_extra_args)
        self.mu = self.eval_gux(self.mu, u, sigma_u, *g_extra_args)
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

    def update(self, z, eval_hx, eval_Ht, Qt, Ht_args=(), hx_args=(), residual=np.subtract, **kwargs):
        z = np.atleast_1d(np.asarray(z, float))
        Qt = np.atleast_2d(Qt).astype(float)
        Ht = np.atleast_2d(eval_Ht(*Ht_args)).astype(float)

        SigmaHT = self.Sigma @ Ht.T
        self.S = Ht @ SigmaHT + Qt
        self.K = SigmaHT @ inv(self.S)

        z_hat = np.atleast_1d(np.asarray(eval_hx(*hx_args), float))
        y = residual(z, z_hat, **kwargs)
        self.mu = self.mu + self.K @ y

        # Joseph form: numerically stable
        I_KH = self._I - self.K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ Qt @ self.K.T
