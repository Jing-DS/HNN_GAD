import torch
from .utils import acosh, sqrt, clamp, sabs, sign


EXP_MAX_NORM = 100.

@torch.jit.script
def tanh(x):
    return x.clamp(-15, 15).tanh()


@torch.jit.script
def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


@torch.jit.script
def arsinh(x: torch.Tensor):
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(1e-15).log().to(x.dtype)


@torch.jit.script
def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)


@torch.jit.script
def tan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
            + 1382 / 155925 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x + 1 / 3 * k * x ** 3
    elif order == 2:
        return x + 1 / 3 * k * x ** 3 + 2 / 15 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


@torch.jit.script
def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
            - 1 / 11 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x ** 3
    elif order == 2:
        return x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5 - 1 / 7 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


@torch.jit.script
def arsin_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + k * x ** 3 / 6
            + 3 / 40 * k ** 2 * x ** 5
            + 5 / 112 * k ** 3 * x ** 7
            + 35 / 1152 * k ** 4 * x ** 9
            + 63 / 2816 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x + k * x ** 3 / 6
    elif order == 2:
        return x + k * x ** 3 / 6 + 3 / 40 * k ** 2 * x ** 5
    elif order == 3:
        return x + k * x ** 3 / 6 + 3 / 40 * k ** 2 * x ** 5 + 5 / 112 * k ** 3 * x ** 7
    elif order == 4:
        return (
            x
            + k * x ** 3 / 6
            + 3 / 40 * k ** 2 * x ** 5
            + 5 / 112 * k ** 3 * x ** 7
            + 35 / 1152 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


@torch.jit.script
def sin_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - k * x ** 3 / 6
            + k ** 2 * x ** 5 / 120
            - k ** 3 * x ** 7 / 5040
            + k ** 4 * x ** 9 / 362880
            - k ** 5 * x ** 11 / 39916800
            # + o(k**6)
        )
    elif order == 1:
        return x - k * x ** 3 / 6
    elif order == 2:
        return x - k * x ** 3 / 6 + k ** 2 * x ** 5 / 120
    elif order == 3:
        return x - k * x ** 3 / 6 + k ** 2 * x ** 5 / 120 - k ** 3 * x ** 7 / 5040
    elif order == 4:
        return (
            x
            - k * x ** 3 / 6
            + k ** 2 * x ** 5 / 120
            - k ** 3 * x ** 7 / 5040
            + k ** 4 * x ** 9 / 362880
        )
    else:
        raise RuntimeError("order not in [-1, 5]")



@torch.jit.script
def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return tan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * tanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.clamp_max(1e38).tan()
    else:
        tan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.clamp_max(1e38).tan(), tanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, tan_k_zero_taylor(x, k, order=1), tan_k_nonzero)


@torch.jit.script
def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)


@torch.jit.script
def arsin_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return arsin_k_zero_taylor(x, k)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * arsinh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.asin()
    else:
        arsin_k_nonzero = (
            torch.where(
                k_sign.gt(0),
                scaled_x.clamp(-1 + 1e-7, 1 - 1e-7).asin(),
                arsinh(scaled_x),
            )
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, arsin_k_zero_taylor(x, k, order=1), arsin_k_nonzero)


@torch.jit.script
def sin_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return sin_k_zero_taylor(x, k)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * torch.sinh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.sin()
    else:
        sin_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.sin(), torch.sinh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, sin_k_zero_taylor(x, k, order=1), sin_k_nonzero)


def inner(u, v, *, keepdim=False, dim=-1):
    r"""
    Minkowski inner product.

    .. math::
        \langle\mathbf{u}, \mathbf{v}\rangle_{\mathcal{L}}:=-u_{0} v_{0}+u_{1} v_{1}+\ldots+u_{d} v_{d}

    Parameters
    ----------
    u : tensor
        vector in ambient space
    v : tensor
        vector in ambient space
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(u, v, keepdim=keepdim, dim=dim)


def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        # return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
        #     dim=dim, keepdim=True
        # )
        return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(
            dim=dim, keepdim=True
        )


def inner0(v, *, k, keepdim=False, dim=-1):
    r"""
    Minkowski inner product with zero vector.

    Parameters
    ----------
    v : tensor
        vector in ambient space
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner0(v, k=k, keepdim=keepdim, dim=dim)


def _inner0(v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = -v.narrow(dim, 0, 1)
    if keepdim is False:
        res = res.squeeze(dim)
    return res


def cinner(x, y):
    x = x.clone()
    x.narrow(-1, 0, 1).mul_(-1)
    return x @ y.transpose(-1, -2)


def dist(x, y, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid.

    .. math::

        d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})=\sqrt{k} \operatorname{arcosh}\left(-\frac{\langle\mathbf{x}, \mathbf{y}\rangle_{\mathcal{L}}}{k}\right)

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    y : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim)


def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    return acosh(d / k)


def dist0(x, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid to zero point.

    .. math::

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and zero point
    """
    return _dist0(x, k=k, keepdim=keepdim, dim=dim)


def _dist0(x, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner0(x, k=k, dim=dim, keepdim=keepdim)
    return acosh(d / k)


def cdist(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor):
    # tmp = torch.ones(x.shape[-1], device=x.device)
    # tmp[0] = -1
    x = x.clone()
    x.narrow(-1, 0, 1).mul_(-1)
    return acosh(-(x @ y.transpose(-1, -2)))


def project(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{k+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim)


@torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    right_ = x.narrow(dim, 1, dn)
    left_ = torch.sqrt(
        k + (right_ * right_).sum(dim=dim, keepdim=True)
    )
    x = torch.cat((left_, right_), dim=dim)
    return x


def project_polar(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid from polar coordinates.

    ... math::
        \pi((\mathbf{d}, r))=(\sqrt{k} \sinh (r/\sqrt{k}) \mathbf{d}, \cosh (r / \sqrt{k}))

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_polar(x, k=k, dim=dim)


def _project_polar(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    r = x.narrow(dim, -1, 1)
    res = torch.cat(
        (
            torch.cosh(r / torch.sqrt(k)),
            torch.sqrt(k) * torch.sinh(r / torch.sqrt(k)) * d,
        ),
        dim=dim,
    )
    return res


def project_u(x, v, *, k, dim=-1):
    r"""
    Projection of the vector on the tangent space.

    ... math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, 1}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \mathbf{x} / k

    Parameters
    ----------
    x: tensor
        point on the Hyperboloid
    v: tensor
        vector in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_u(x, v, k=k, dim=dim)


def _project_u(x, v, k: torch.Tensor, dim: int = -1):
    return v.addcmul(_inner(x, v, dim=dim, keepdim=True), x / k)


def project_u0(u):
    narrowed = u.narrow(-1, 0, 1)
    vals = torch.zeros_like(u)
    vals[..., 0:1] = narrowed
    return u - vals


def norm(u, *, keepdim=False, dim=-1):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Hyperboloid.

    .. math::

        \|\mathbf{v}\|_{\mathcal{L}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle_{\mathcal{L}}}

    Parameters
    ----------
    u : tensor
        tangent vector on Hyperboloid
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(u, keepdim=keepdim, dim=dim)


def _norm(u, keepdim: bool = False, dim: int = -1):
    return sqrt(_inner(u, u, keepdim=keepdim))


def expmap(x, u, *, k, dim=-1):
    r"""
    Compute exponential map on the Hyperboloid.

    .. math::

        \exp _{\mathbf{x}}^{k}(\mathbf{v})=\cosh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}


    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    u : tensor
        unit speed vector on Hyperboloid
    k: tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, k=k, dim=dim)


def _expmap(x, u, k: torch.Tensor, dim: int = -1):
    # nomin = (_norm(u, keepdim=True, dim=dim) / torch.sqrt(k)).clamp_max(10.)
    nomin = (_norm(u, keepdim=True, dim=dim))
    u = u / nomin
    nomin = nomin.clamp_max(EXP_MAX_NORM)
    # mask = nomin.lt(EXP_MAX_NORM)
    # if (~mask).any():
    #     nomin_mask = nomin.masked_scatter(mask, torch.ones_like(nomin))
    #     u = u / nomin_mask
    #     nomin = (_norm(u, keepdim=True, dim=dim))
    p = torch.cosh(nomin) * x + torch.sinh(nomin) * u
    return p


def expmap0(u, *, k, dim=-1):
    r"""
    Compute exponential map for Hyperboloid from :math:`0`.

    Parameters
    ----------
    u : tensor
        speed vector on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)


def _expmap0(u, k: torch.Tensor, dim: int = -1):
    # nomin = (_norm(u, keepdim=True, dim=dim) / torch.sqrt(k)).clamp_max(10.)
    nomin = (_norm(u, keepdim=True, dim=dim))
#     print(nomin[0][0:10])
    u = u / nomin
    nomin = nomin.clamp_max(EXP_MAX_NORM)
#     print(nomin[0][0:10])
    # mask = nomin.lt(EXP_MAX_NORM)
    # if (~mask).any():
    #     nomin_mask = nomin.masked_scatter(mask, torch.ones_like(nomin))
    #     u = u / nomin_mask
    #     nomin = (_norm(u, keepdim=True, dim=dim))
    l_v = torch.cosh(nomin)
    r_v = torch.sinh(nomin) * u
#     print(l_v[0][0:10],l_v[0][0:10])
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p


def logmap(x, y, *, k, dim=-1):
    r"""
    Compute logarithmic map for two points :math:`x` and :math:`y` on the manifold.

    .. math::

        \log _{\mathbf{x}}^{k}(\mathbf{y})=d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})
            \frac{\mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}}{\left\|
            \mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}\right\|_{\mathcal{L}}}

    The result of Logarithmic map is a vector such that

    .. math::

        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))


    Parameters
    ----------
    x : tensor
        starting point on Hyperboloid
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k=k, dim=dim)


def _logmap(x, y, k, dim: int = -1):
    dist_ = _dist(x, y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * _inner(x, y, keepdim=True) * x
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom
    # alpha = -inner(x, y, k, keepdim=True)             # 没用到，不确定是不是对的
    # nom = acosh(alpha)
    # denom = (alpha * alpha - 1).sqrt()
    # return nom / denom * (y - alpha * x)

def clogmap(x, y):
    alpha = (-cinner(x, y).unsqueeze(-1)).clamp_min(1 + 1e-6)
    nom = acosh(alpha)
    denom = (alpha * alpha - 1).sqrt()
    return nom / denom * (y.unsqueeze(-3) - alpha * x.unsqueeze(-2))


def logmap0(y, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`y` from :math:`0` on the manifold.

    Parameters
    ----------
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k=k, dim=dim)


def _logmap0(y, k, dim: int = -1):
    # dist_ = _dist0(y, k=k, dim=dim, keepdim=True)
    # nomin_ = 1.0 / k * _inner0(y, k=k, keepdim=True) * torch.sqrt(k)
    # dn = y.size(dim) - 1
    # nomin = torch.cat((nomin_ + y.narrow(dim, 0, 1),
    #                    y.narrow(dim, 1, dn)), dim)
    # denom = _norm(nomin, keepdim=True)
    # return dist_ * nomin / denom
    alpha = -_inner0(y, k, keepdim=True)
    zero_point = torch.zeros(y.shape[-1], device=y.device)
    zero_point[0] = 1
    mapped = acosh(alpha) / torch.sqrt(alpha * alpha - 1) * (y - alpha * zero_point)
    return mapped


def logmap0back(x, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`0` from :math:`x` on the manifold.

    Parameters
    ----------
    x : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0back(x, k=k, dim=dim)


def _logmap0back(x, k, dim: int = -1):
    dist_ = _dist0(x, k=k, dim=dim, keepdim=True)
    nomin_ = 1.0 / k * _inner0(x, k=k, keepdim=True) * x
    dn = nomin_.size(dim) - 1
    nomin = torch.cat(
        (nomin_.narrow(dim, 0, 1) + 1, nomin_.narrow(dim, 1, dn)), dim
    )
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom
    # y = torch.zeros(x.shape[-1], device=x.device)
    # y[0] = k.sqrt()
    # return _logmap(x, y, k, dim)


def egrad2rgrad(x, grad, *, k, dim=-1):
    r"""
    Translate Euclidean gradient to Riemannian gradient on tangent space of :math:`x`.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, k}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \frac{\mathbf{x}}{k}

    Parameters
    ----------
    x : tensor
        point on the Hyperboloid
    grad : tensor
        Euclidean gradient for :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in `
    """
    return _egrad2rgrad(x, grad, k=k, dim=dim)


def _egrad2rgrad(x, grad, k, dim: int = -1):
    grad.narrow(-1, 0, 1).mul_(-1)
    grad = grad.addcmul(_inner(x, grad, dim=dim, keepdim=True), x / k)
    return grad


def parallel_transport(x, y, v, *, k, dim=-1):
    r"""
    Perform parallel transport on the Hyperboloid.

    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, k=k, dim=dim)


def _parallel_transport(x, y, v, k, dim: int = -1):
    # lmap = _logmap(x, y, k=k, dim=dim)
    # nom = _inner(lmap, v, keepdim=True)
    # denom = _dist(x, y, k=k, dim=dim, keepdim=True) ** 2
    # p = v - nom / denom * (lmap + _logmap(y, x, k=k, dim=dim))
    # return p
    nom = _inner(y, v, keepdim=True)
    denom = torch.clamp_min(k - _inner(x, y, keepdim=True), 1e-7)
    # return v + nom / denom * (x + y)
    return v.addcmul(nom / denom, x + y)


def parallel_transport0(y, v, *, k, dim=-1):
    r"""
    Perform parallel transport from zero point.

    Parameters
    ----------
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport0(y, v, k=k, dim=dim)


def _parallel_transport0(y, v, k, dim: int = -1):
    # lmap = _logmap0(y, k=k, dim=dim)
    # nom = _inner(lmap, v, keepdim=True)
    # denom = _dist0(y, k=k, dim=dim, keepdim=True) ** 2
    # p = v - nom / denom * (lmap + _logmap0back(y, k=k, dim=dim))
    # return p
    nom = _inner(y, v, keepdim=True)
    denom = torch.clamp_min(k - _inner0(y, k=k, keepdim=True), 1e-7)
    zero_point = torch.zeros_like(y)
    zero_point[..., 0] = 1
    # return v + nom / denom * (y + zero_point)
    return v.addcmul(nom / denom, y + zero_point)


def parallel_transport0back(x, v, *, k, dim: int = -1):
    r"""
    Perform parallel transport to the zero point.

    Special case parallel transport with last point at zero that
    can be computed more efficiently and numerically stable

    Parameters
    ----------
    x : tensor
        target point
    v : tensor
        vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, k=k, dim=dim)


def _parallel_transport0back(x, v, k, dim: int = -1):
    # lmap = _logmap0back(x, k=k, dim=dim)
    # nom = _inner(lmap, v, keepdim=True)
    # denom = _dist0(x, k=k, dim=dim, keepdim=True) ** 2
    # p = v - nom / denom * (lmap + _logmap0(x, k=k, dim=dim))
    # return p
    nom = _inner0(v, k=k, keepdim=True)
    denom = torch.clamp_min(k - _inner0(x, k=k, keepdim=True), 1e-7)
    zero_point = torch.zeros_like(x)
    zero_point[..., 0] = 1
    # return v + nom / denom * (x + zero_point)
    return v.addcmul(nom / denom, x + zero_point)

def mobius_add(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector addition.

    .. math::

        x \oplus_\kappa y =
        \frac{
            (1 - 2 \kappa \langle x, y\rangle - \kappa \|y\|^2_2) x +
            (1 + \kappa \|x\|_2^2) y
        }{
            1 - 2 \kappa \langle x, y\rangle + \kappa^2 \|x\|^2_2 \|y\|^2_2
        }

    .. plot:: plots/extended/stereographic/mobius_add.py

    In general this operation is not commutative:

    .. math::

        x \oplus_\kappa y \ne y \oplus_\kappa x

    But in some cases this property holds:

    * zero vector case

    .. math::

        \mathbf{0} \oplus_\kappa x = x \oplus_\kappa \mathbf{0}

    * zero curvature case that is same as Euclidean addition

    .. math::

        x \oplus_0 y = y \oplus_0 x

    Another useful property is so called left-cancellation law:

    .. math::

        (-x) \oplus_\kappa (x \oplus_\kappa y) = y

    Parameters
    ----------
    x : tensor
        point on the manifold
    y : tensor
        point on the manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius addition
    """
    return _mobius_add(x, y, k, dim=dim)



@torch.jit.script
def _mobius_add(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k ** 2 * x2 * y2
    # minimize denom (omit K to simplify th notation)
    # 1)
    # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
    # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
    # 2)
    # {y + x * <y, y> = 0
    # {x + y * <x, x> = 0
    # 3)
    # {- y/<y, y> = x
    # {- x/<x, x> = y
    # 4)
    # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
    return num / denom.clamp_min(1e-15)


def mobius_sub(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector subtraction.

    The Möbius subtraction can be represented via the Möbius addition as
    follows:

    .. math::

        x \ominus_\kappa y = x \oplus_\kappa (-y)

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius subtraction
    """
    return _mobius_sub(x, y, k, dim=dim)



def _mobius_sub(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    return _mobius_add(x, -y, k, dim=dim)


def gyration(
    a: torch.Tensor, b: torch.Tensor, u: torch.Tensor, *, k: torch.Tensor, dim=-1
):
    r"""
    Compute the gyration of :math:`u` by :math:`[a,b]`.

    The gyration is a special operation of gyrovector spaces. The gyrovector
    space addition operation :math:`\oplus_\kappa` is not associative (as
    mentioned in :func:`mobius_add`), but it is gyroassociative, which means

    .. math::

        u \oplus_\kappa (v \oplus_\kappa w)
        =
        (u\oplus_\kappa v) \oplus_\kappa \operatorname{gyr}[u, v]w,

    where

    .. math::

        \operatorname{gyr}[u, v]w
        =
        \ominus (u \oplus_\kappa v) \oplus (u \oplus_\kappa (v \oplus_\kappa w))

    We can simplify this equation using the explicit formula for the Möbius
    addition [1]. Recall,

    .. math::

        A = - \kappa^2 \langle u, w\rangle \langle v, v\rangle
            - \kappa \langle v, w\rangle
            + 2 \kappa^2 \langle u, v\rangle \langle v, w\rangle\\
        B = - \kappa^2 \langle v, w\rangle \langle u, u\rangle
            + \kappa \langle u, w\rangle\\
        D = 1 - 2 \kappa \langle u, v\rangle
            + \kappa^2 \langle u, u\rangle \langle v, v\rangle\\

        \operatorname{gyr}[u, v]w = w + 2 \frac{A u + B v}{D}.

    Parameters
    ----------
    a : tensor
        first point on manifold
    b : tensor
        second point on manifold
    u : tensor
        vector field for operation
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of automorphism

    References
    ----------
    [1]  A. A. Ungar (2009), A Gyrovector Space Approach to Hyperbolic Geometry
    """
    return _gyration(a, b, u, k, dim=dim)



@torch.jit.script
def _gyration(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    # non-simplified
    # mupv = -_mobius_add(u, v, K)
    # vpw = _mobius_add(u, w, K)
    # upvpw = _mobius_add(u, vpw, K)
    # return _mobius_add(mupv, upvpw, K)
    # simplified
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    K2 = k ** 2
    a = -K2 * uw * v2 - k * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 + k * uw
    d = 1 - 2 * k * uv + K2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(1e-15)


def mobius_coadd(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector coaddition.

    The addition operation :math:`\oplus_\kappa` is neither associative, nor
    commutative. In contrast, the coaddition :math:`\boxplus_\kappa` (or
    cooperation) is an associative operation that is defined as follows.

    .. math::

        a \boxplus_\kappa b
        =
        b \boxplus_\kappa a
        =
        a\operatorname{gyr}[a, -b]b\\
        = \frac{
            (1 + \kappa \|y\|^2_2) x + (1 + \kappa \|x\|_2^2) y
            }{
            1 + \kappa^2 \|x\|^2_2 \|y\|^2_2
        },

    where :math:`\operatorname{gyr}[a, b]v = \ominus_\kappa (a \oplus_\kappa b)
    \oplus_\kappa (a \oplus_\kappa (b \oplus_\kappa v))`

    The following right cancellation property holds

    .. math::

        (a \boxplus_\kappa b) \ominus_\kappa b = a\\
        (a \oplus_\kappa b) \boxminus_\kappa b = a

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius coaddition

    """
    return _mobius_coadd(x, y, k, dim=dim)


# TODO: check numerical stability with Gregor's paper!!!
@torch.jit.script
def _mobius_coadd(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    # x2 = x.pow(2).sum(dim=dim, keepdim=True)
    # y2 = y.pow(2).sum(dim=dim, keepdim=True)
    # num = (1 + K * y2) * x + (1 + K * x2) * y
    # denom = 1 - K ** 2 * x2 * y2
    # avoid division by zero in this way
    # return num / denom.clamp_min(1e-15)
    #
    return _mobius_add(x, _gyration(x, -y, y, k=k, dim=dim), k, dim=dim)


def mobius_cosub(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector cosubtraction.

    The Möbius cosubtraction is defined as follows:

    .. math::

        a \boxminus_\kappa b = a \boxplus_\kappa -b

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius cosubtraction

    """
    return _mobius_cosub(x, y, k, dim=dim)


@torch.jit.script
def _mobius_cosub(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    return _mobius_coadd(x, -y, k, dim=dim)


# TODO: can we make this operation somehow safer by breaking up the
# TODO: scalar multiplication for K>0 when the argument to the
# TODO: tan function gets close to pi/2+k*pi for k in Z?
# TODO: one could use the scalar associative law
# TODO: s_1 (X) s_2 (X) x = (s_1*s_2) (X) x
# TODO: to implement a more stable Möbius scalar mult
def mobius_scalar_mul(r: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius scalar multiplication.

    .. math::

        r \otimes_\kappa x
        =
        \tan_\kappa(r\tan_\kappa^{-1}(\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to the Euclidean scalar multiplication

    * `n-addition` property

    .. math::

         r \otimes_\kappa x = x \oplus_\kappa \dots \oplus_\kappa x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_\kappa x
         =
         r_1 \otimes_\kappa x \oplus r_2 \otimes_\kappa x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_\kappa x = r_1 \otimes_\kappa (r_2 \otimes_\kappa x)

    * Monodistributivity

    .. math::

         r \otimes_\kappa (r_1 \otimes x \oplus r_2 \otimes x) =
         r \otimes_\kappa (r_1 \otimes x) \oplus r \otimes (r_2 \otimes x)

    * Scaling property

    .. math::

        |r| \otimes_\kappa x / \|r \otimes_\kappa x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : tensor
        scalar for multiplication
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius scalar multiplication
    """
    return _mobius_scalar_mul(r, x, k, dim=dim)



@torch.jit.script
def _mobius_scalar_mul(
    r: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(r * artan_k(x_norm, k), k) * (x / x_norm)
    return res_c

def geodesic(
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1
):
    r"""
    Compute the point on the path connecting :math:`x` and :math:`y` at time :math:`x`.

    The path can also be treated as an extension of the line segment to an
    unbounded geodesic that goes through :math:`x` and :math:`y`. The equation
    of the geodesic is given as:

    .. math::

        \gamma_{x\to y}(t)
        =
        x \oplus_\kappa t \otimes_\kappa ((-x) \oplus_\kappa y)

    The properties of the geodesic are the following:

    .. math::

        \gamma_{x\to y}(0) = x\\
        \gamma_{x\to y}(1) = y\\
        \dot\gamma_{x\to y}(t) = v

    Furthermore, the geodesic also satisfies the property of local distance
    minimization:

    .. math::

         d_\kappa(\gamma_{x\to y}(t_1), \gamma_{x\to y}(t_2)) = v|t_1-t_2|

    "Natural parametrization" of the curve ensures unit speed geodesics which
    yields the above formula with :math:`v=1`.

    However, we can always compute the constant speed :math:`v` from the points
    that the particular path connects:

    .. math::

        v = d_\kappa(\gamma_{x\to y}(0), \gamma_{x\to y}(1)) = d_\kappa(x, y)


    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        point on the geodesic going through x and y
    """
    x = lorentz_to_poincare(x, -k)
    y = lorentz_to_poincare(y, -k)
    return poincare_to_lorentz(_geodesic(t, x, y, -k, dim=dim), -k)

@torch.jit.script
def _geodesic(
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    # this is not very numerically stable
    v = _mobius_add(-x, y, k, dim=dim)
    tv = _mobius_scalar_mul(t, v, k, dim=dim)
    gamma_t = _mobius_add(x, tv, k, dim=dim)
    return gamma_t

def geodesic_unit(t, x, u, *, k):
    r"""
    Compute unit speed geodesic at time :math:`t` starting from :math:`x` with direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{\mathbf{x} \rightarrow \mathbf{u}}^{k}(t)=\cosh \left(\frac{t}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{t}{\sqrt{k}}\right) \mathbf{u}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point
    u : tensor
        unit direction vector
    k : tensor
        manifold negative curvature

    Returns
    -------
    tensor
        the point on geodesic line
    """
    return _geodesic_unit(t, x, u, k=k)


def _geodesic_unit(t, x, u, k):
    return (
        torch.cosh(t) * x
        + torch.sinh(t) * u
    )


def lorentz_to_poincare(x, k, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disk.

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+\sqrt{k}}

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Poincare disk
    """
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + 1)


def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    r"""
    Diffeomorphism that maps from Poincare disk to Hyperboloid.

    .. math::

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\sqrt{k} \left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Hyperboloid
    """
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = (
        torch.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res
