import numpy as np
import torch
from torch import nn
from helper import *


class BSpline(nn.Module):
    """
    Class for computing the B-spline funcions b_i(x)
    and constructing the penality matrix S.

    # Args
        start: float or int; start of the region
        end: float or int; end of the region
        n_bases: int; number of spline bases
        spline_order: int; spline order

    # Methods
        - **getS(add_intercept=False)** - Get the penalty matrix S
              - Args
                     - **add_intercept**: bool. If true, intercept column is added to the returned matrix.
              - Returns
                     - `np.array`, of shape `(n_bases + add_intercept, n_bases + add_intercept)`
        - **predict(x, add_intercept=False)** - For some x, predict the bn(x) for each base
              - Args
                     - **x**: np.array; Vector of dimension 1
                     - **add_intercept**: bool; If True, intercept column is added to the to the final array
              - Returns
                     - `torch.tensor`, of shape `(len(x), n_bases + (add_intercept))`
    """

    def __init__(self, start=0, end=1, n_bases=10, spline_order=3, intercept=1):
        super().__init__()

        self.start = start
        self.end = end
        self.n_bases = n_bases
        self.spline_order = spline_order
        self.add_intercept = True if intercept == 1 else False

        self.knots = get_knots(self.start, self.end, self.n_bases, self.spline_order)

        self.S = get_S(self.n_bases, self.spline_order)

    def __repr__(self):
        return "BSpline(start={0}, end={1}, n_bases={2}, spline_order={3})".format(
            self.start, self.end, self.n_bases, self.spline_order
        )

    def getS(self):
        """Get the penalty matrix S
        Returns:
            torch.tensor, of shape (n_bases + add_intercept, n_bases + add_intercept)
        """
        S = self.S
        if self.add_intercept is True:
            # S <- cbind(0, rbind(0, S)) # in R
            zeros = np.zeros_like(S[:1, :])
            S = np.vstack([zeros, S])

            zeros = np.zeros_like(S[:, :1])
            S = np.hstack([zeros, S])
        return S

    def forward(self, x):
        """For some x, predict the bn(x) for each base
        Args:
            x: torch.tensor
            add_intercept: bool; should we add the intercept to the final array
        Returns:
            torch.tensor, of shape (len(x), n_bases + (add_intercept))
        """
        # sanity check
        if x.min() < self.start:
            raise Warning("x.min() < self.start")
        if x.max() > self.end:
            raise Warning("x.max() > self.end")

        return get_X_spline_torch(
            x=x,
            knots=self.knots,
            n_bases=self.n_bases,
            spline_order=self.spline_order,
            add_intercept=self.add_intercept,
        )
        # return get_X_spline(x=x, knots=self.knots, n_bases=self.n_bases, spline_order=self.spline_order, add_intercept=self.add_intercept)

    def get_config(self):
        return {
            "start": self.start,
            "end": self.end,
            "n_bases": self.n_bases,
            "spline_order": self.spline_order,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _trunc(x, minval=None, maxval=None):
    """Truncate vector values to have values on range [minval, maxval]"""
    x = torch.clone(x)
    if minval != None:
        x[x < minval] = minval
    if maxval != None:
        x[x > maxval] = maxval
    return x


def encodeSplines(x, n_bases=5, spline_order=3, start=None, end=None, warn=True):
    """Function for the class `EncodeSplines`.
    Expansion by generating B-spline basis functions for each x
    and each n (spline-index) with `scipy.interpolate.splev`,
    based on the pre-placed equidistant knots on [start, end] range.

    # Arguments
        x: a torch.tensor of positions
        n_bases int: Number of spline bases.
        spline_order: 2 for quadratic, 3 for qubic splines
        start, end: range of values. If None, they are inferred from the data
        as minimum and maximum value.
        warn: Show warnings.

    # Returns
        `torch.tensor` of shape `(x.shape[0], x.shape[1], channels, n_bases)`
    """

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))

    if start is None:
        start = torch.amin(x)  # should be np.nanmin
    else:
        if x.min() < start:
            if warn:
                print(
                    "WARNING, x.min() < start for some elements. Truncating them to start: x[x < start] = start"
                )
            x = _trunc(x, minval=start)
    if end is None:
        end = torch.amax(x)  # should be np.nanmax
    else:
        if x.max() > end:
            if warn:
                print(
                    "WARNING, x.max() > end for some elements. Truncating them to end: x[x > end] = end"
                )
            x = _trunc(x, maxval=end)
    bs = BSpline(start, end, n_bases=n_bases, spline_order=spline_order)

    # concatenate x to long
    assert len(x.shape) == 2
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    x_long = x.reshape((-1,))

    # shape = (n_rows * n_cols, n_bases)
    x_feat = bs.predict(x_long, add_intercept=False)

    x_final = x_feat.reshape((n_rows, n_cols, n_bases))
    return x_final


if __name__ == "__main__":
    
    obs = -30 * torch.rand(10)
    # print(obs)
    bspline_args = {'start': -32, 'end': 0, 'n_bases': 7, 'spline_order': 2, 'intercept': 1}
    # bspline_args = {'start': -30, 'end': 0, 'n_bases': 16, 'spline_order': 2}
    bspline = BSpline(**bspline_args)
    output = bspline.predict(obs)

    import matplotlib.pyplot as plt
    x_plot = torch.linspace(bspline.start, bspline.end, 1000)
    basis_plot = bspline.predict(x_plot)

    plt.figure(figsize=(10, 6))
    for i in range(bspline.n_bases):
        plt.plot(x_plot, basis_plot[:, i], label=f'Basis {i + 1}')
    
    plt.scatter(obs, torch.zeros_like(obs), color='red', label='Observations')
    plt.title(f'B-spline Bases (Order {bspline.spline_order}, {bspline.n_bases} Bases)')
    plt.xlabel('x')
    plt.ylabel('Basis Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # beta_hat = torch.tensor([ 0.0000, -0.1320,  0.8180,  0.1750,  0.1690, -0.0490,  0.4600,  0.2010])  # old
    beta_hat = torch.tensor([-0.0030,  0.0000,  0.0160,  0.1960, -0.0150, -1.0230,  0.4580,  0.3650])
    predict = basis_plot @ beta_hat
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, predict, label=f'w-function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    plt.savefig('w_function.png')
    