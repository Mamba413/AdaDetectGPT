import numpy as np
import scipy.interpolate as si
import torch


def get_knots(start, end, n_bases=5, spline_order=3):
    """
    Arguments:
        x; torch.tensor of dim 1
    """
    x_range = end - start
    start = start - x_range * 0.001
    end = end + x_range * 0.001
    # mgcv annotation
    m = spline_order - 1
    nk = n_bases - m  # number of interior knots
    dknots = (end - start) / (nk - 1)
    knots = torch.linspace(
        start=start - dknots * (m + 1), end=end + dknots * (m + 1), steps=nk + 2 * m + 2
    )
    return knots.float()


def get_X_spline(x, knots, n_bases=5, spline_order=3, add_intercept=True):
    """
    Returns:
        torch.tensor of shape [len(x), n_bases + (add_intercept)]
    # BSpline formula
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline
    """
    cuda = False
    if x.is_cuda:
        cuda = True
    if len(x.shape) != 1:
        raise ValueError("x has to be 1 dimentional")
    tck = [knots, torch.zeros(n_bases), spline_order]
    X = torch.zeros([len(x), n_bases], dtype=torch.float32)
    x = x.cpu().numpy()  # TODO: tensor interpolation?
    for i in range(n_bases):
        vec = torch.zeros(n_bases, dtype=torch.float32)
        vec[i] = 1.0
        tck[1] = vec
        if cuda:
            X[:, i] = torch.from_numpy(si.splev(x, tck, der=0)).to(
                "cuda"
            )  # TODO: specify cuda number
        else:
            X[:, i] = torch.from_numpy(si.splev(x, tck, der=0))
    if add_intercept is True:
        ones = torch.ones_like(X[:, :1])
        X = torch.hstack([ones, X])
    return X

def _torch_splev(x: torch.Tensor,
                 t: torch.Tensor,
                 c: torch.Tensor,
                 k: int) -> torch.Tensor:
    """
    Cox–de Boor recursion in pure torch.
    Returns spline evaluation y at points x.
    
    x: (N,) tensor of evaluation points
    t: (M,) knot vector
    c: (n_bases,) coefficient vector (one-hot for basis)
    k: spline order (degree)
    """
    # x: (N,)
    # initial basis: degree 0, N_i shape=(M-1, N)
    N_knots = t.numel()
    N_i = ((x.unsqueeze(0) >= t[:-1].unsqueeze(1)) &
           (x.unsqueeze(0) <  t[1:].unsqueeze(1))).float()
    # include right boundary
    N_i[-1, (x == t[-1]).nonzero(as_tuple=True)[0]] = 1.0

    # recursive build for deg = 1..k
    for deg in range(1, k+1):
        curr_rows, T = N_i.shape
        new_rows = curr_rows - 1
        N_i_new = torch.zeros((new_rows, T), device=x.device, dtype=x.dtype)
        for i in range(new_rows):
            denom1 = t[i + deg]   - t[i]
            denom2 = t[i + deg + 1] - t[i + 1]

            term1 = 0.0
            if denom1.item() > 0:
                term1 = ((x - t[i]) / denom1).unsqueeze(0) * N_i[i:i+1]
            term2 = 0.0
            if denom2.item() > 0:
                term2 = ((t[i + deg + 1] - x) / denom2).unsqueeze(0) * N_i[i+1:i+2]

            N_i_new[i:i+1] = term1 + term2
        N_i = N_i_new

    # After k recursions, N_i has shape (n_bases, N)
    # combine with coefficients c of shape (n_bases,)
    y = (c.unsqueeze(1) * N_i).sum(dim=0)  # yields (N,)
    return y

def get_X_spline_torch(x: torch.Tensor,
                       knots: torch.Tensor,
                       n_bases: int = 5,
                       spline_order: int = 3,
                       add_intercept: bool = True) -> torch.Tensor:
    """
    Pure-Torch version of BSpline basis evaluation.
    Args:
        x:          (N,) 1-D tensor (CPU or CUDA)
        knots:      (n_bases + spline_order + 1,) knot vector
        n_bases:    number of basis functions d
        spline_order: degree k
        add_intercept: if True, prepend a column of 1s
    Returns:
        X:          (N, d + intercept) tensor
    """
    if x.dim() != 1:
        raise ValueError("x must be 1-D tensor")

    device = x.device
    N = x.size(0)
    d = n_bases

    # prepare output
    X = torch.zeros(N, d, device=device, dtype=torch.float32)

    # compute each basis
    for i in range(d):
        # one-hot coefficient for basis i
        c = torch.zeros(d, device=device, dtype=torch.float32)
        c[i] = 1.0
        X[:, i] = _torch_splev(x, knots.to(device), c, spline_order)

    # optional intercept
    if add_intercept:
        ones = torch.ones(N, 1, device=device, dtype=X.dtype)
        X = torch.cat([ones, X], dim=1)

    return X


def get_S(n_bases=5, spline_order=3, add_intercept=True):
    # mvcv R-code
    # S<-diag(object$bs.dim);
    # if (m[2]) for (i in 1:m[2]) S <- diff(S)
    # object$S <- list(t(S)%*%S)  # get penalty
    # object$S[[1]] <- (object$S[[1]]+t(object$S[[1]]))/2 # exact symmetry

    S = np.identity(n_bases)
    m2 = spline_order - 1  # m[2] is the same as m[1] by default

    # m2 order differences
    for i in range(m2):
        S = np.diff(S, axis=0)  # same as diff() in R
    S = np.dot(S.T, S)
    S = (S + S.T) / 2  # exact symmetry
    if add_intercept is True:
        # S <- cbind(0, rbind(0, S)) # in R
        zeros = np.zeros_like(S[:1, :])
        S = np.vstack([zeros, S])
        zeros = np.zeros_like(S[:, :1])
        S = np.hstack([zeros, S])
    return S.astype(np.float32)


def corr2d_stack(X, K):
    """iterate through the 0th dimension (channel dimension) of `X` and
    `K`. multiply them and stack together
    """
    out = torch.stack([torch.matmul(x, k) for x, k in zip(X, K)]).squeeze(-1)
    out = out.permute((1, 2, 0))
    return out
