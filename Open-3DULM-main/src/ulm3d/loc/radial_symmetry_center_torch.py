import torch
import torch.nn.functional as F
from typing import Tuple

def uniform_filter_3d(x: torch.Tensor, size: int = 3) -> torch.Tensor:
    """
    3D uniform filter using conv3d.
    x: (Nx, Ny, Nz)
    """
    kernel = torch.ones(
        (1, 1, size, size, size),
        device=x.device,
        dtype=x.dtype,
    ) / (size ** 3)

    x = x.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y = F.conv3d(x, kernel, padding=1)
    return y.squeeze(0).squeeze(0)


def radial_symmetry_center_3d_torch(
    I: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Torch implementation of radial symmetry super-localization.

    Args:
        I (torch.Tensor): 3D intensity patch (Nx, Ny, Nz)

    Returns:
        (x, y, z): super-localized coordinates (float)
    """

    device = I.device
    dtype = I.dtype
    size_0, size_1, size_2 = I.shape

    # Shifted derivatives
    dIdu = I[1:, 1:, 1:] - I[:-1, :-1, :-1]
    dIdv = I[1:, :-1, :-1] - I[:-1, 1:, 1:]
    dIdw = I[1:, 1:, :-1] - I[:-1, :-1, 1:]

    # Smoothed derivatives
    fdu = uniform_filter_3d(dIdu)
    fdv = uniform_filter_3d(dIdv)
    fdw = uniform_filter_3d(dIdw)

    # Derivatives along x, y, z
    eps = 1e-12
    dId_1 = 0.5 * (fdv - fdu) + 0.5 * (dIdv - dIdu)
    dId_2 = 0.5 * (fdw - fdv) + 0.5 * (dIdw - dIdv)
    dId_0 = 0.5 * (fdu - fdw) + 0.5 * (dIdu - dIdw)

    # Gradient magnitude
    mag_dI = fdu**2 + fdv**2 + fdw**2
    sdI3 = mag_dI.sum() + eps
    
    # --- Robust coordinate vectors built from mag_dI shape ---
    n0, n1, n2 = mag_dI.shape
    
    v0 = torch.arange(
    -(n0) / 2.0 + 0.5,
    (n0) / 2.0 + 0.5,
    device=device,
    dtype=dtype,
    )
    v1 = torch.arange(
        -(n1) / 2.0 + 0.5,
        (n1) / 2.0 + 0.5,
        device=device,
        dtype=dtype,
    )
    v2 = torch.arange(
        -(n2) / 2.0 + 0.5,
        (n2) / 2.0 + 0.5,
        device=device,
        dtype=dtype,
    )



    # Initial guess
    guess_0 = torch.einsum("ijk,i->", mag_dI, v0) / sdI3
    guess_1 = torch.einsum("ijk,j->", mag_dI, v1) / sdI3
    guess_2 = torch.einsum("ijk,k->", mag_dI, v2) / sdI3

    # Weight matrix
    denom = dId_0**2 + dId_1**2 + dId_2**2 + eps

    dist = torch.sqrt(
    (v0[:, None, None] - guess_0) ** 2
    + (v1[None, :, None] - guess_1) ** 2
    + (v2[None, None, :] - guess_2) ** 2
    ) + eps


    W = mag_dI / denom / dist

    # Omega matrices
    Omega_11 = -2 * (dId_1**2 + dId_2**2)
    Omega_22 = -2 * (dId_2**2 + dId_0**2)
    Omega_33 = -2 * (dId_1**2 + dId_0**2)
    Omega_12 = 2 * dId_1 * dId_0
    Omega_13 = 2 * dId_2 * dId_0
    Omega_23 = 2 * dId_2 * dId_1

    # Matrix M
    M_11 = torch.sum(W * Omega_11)
    M_12 = torch.sum(W * Omega_12)
    M_13 = torch.sum(W * Omega_13)
    M_22 = torch.sum(W * Omega_22)
    M_23 = torch.sum(W * Omega_23)
    M_33 = torch.sum(W * Omega_33)

    # Vector B
    B0 = torch.sum(W * (Omega_11 * v0[:, None, None]
                      + Omega_12 * v1[None, :, None]
                      + Omega_13 * v2[None, None, :]))
    
    B1 = torch.sum(W * (Omega_12 * v0[:, None, None]
                      + Omega_22 * v1[None, :, None]
                      + Omega_23 * v2[None, None, :]))
    
    B2 = torch.sum(W * (Omega_13 * v0[:, None, None]
                      + Omega_23 * v1[None, :, None]
                      + Omega_33 * v2[None, None, :]))


    # Inverse of symmetric 3x3 matrix
    alpha = M_22 * M_33 - M_23**2
    delta = M_11 * M_33 - M_13**2
    phi = M_11 * M_22 - M_12**2
    beta = M_12 * M_33 - M_13 * M_23
    gamma = M_12 * M_23 - M_13 * M_22
    epsilon = M_11 * M_23 - M_13 * M_12

    det = M_11 * alpha - M_12 * beta + M_13 * gamma + eps

    Minv = torch.stack(
        [
            torch.stack([alpha, -beta, gamma]),
            torch.stack([-beta, delta, -epsilon]),
            torch.stack([gamma, -epsilon, phi]),
        ]
    ) / det
    B = torch.stack([B0, B1, B2])
    superloc = Minv @ B

    if torch.isnan(superloc).any():
        return (
            guess_0.item(),
            guess_1.item(),
            guess_2.item(),
        )

    return (
        superloc[0].item(),
        superloc[1].item(),
        superloc[2].item(),
    )
