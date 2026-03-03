import torch
import torch.nn.functional as F
from typing import Tuple
from loguru import logger
import torch._dynamo
def uniform_filter_3d_batch(x: torch.Tensor, size: int = 3, ) -> torch.Tensor:
   
    """
    x: (B, D, H, W)
    """
    B = x.shape[0]
    kernel = torch.ones(
        (1, 1, size, size, size),
        device=x.device,
        dtype=x.dtype,
    ) / (size ** 3)
    pad = size // 2
    x = x.unsqueeze(1)           # (B,1,D,H,W)
    # y = F.conv3d(x, kernel, padding=pad)
    y = F.avg_pool3d(x, kernel_size=size, stride=1, padding=pad, divisor_override=size**3)
    return y.squeeze(1)          # (B,D,H,W)


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
    pad = size//2
    y = F.conv3d(x, kernel, padding=pad)
    return y.squeeze(0).squeeze(0)


def radial_symmetry_center_3d_torch_batch(
    I: torch.Tensor,
    inverse_matrix_version = "inverse",
    reg_eps: float = 1e-6,
    log = None
    ) -> torch.Tensor:
    """
    Batch version of radial symmetry center localization

    Args:
        I: (B, D, H, W) float32 tensor

    Returns:
        sub_pos: (B, 3) tensor [x,y,z]
    """
    if log is None: 
        log = []
    device = I.device
    dtype = I.dtype
    eps = 1e-12

    B = I.shape[0]

    # --- Shifted derivatives ---
    dIdu = I[:, 1:, 1:, 1:] - I[:, :-1, :-1, :-1]
    dIdv = I[:, 1:, :-1, :-1] - I[:, :-1, 1:, 1:]
    dIdw = I[:, 1:, 1:, :-1] - I[:, :-1, :-1, 1:]

    # --- Smoothed derivatives ---
    fdu = uniform_filter_3d_batch(dIdu)
    fdv = uniform_filter_3d_batch(dIdv)
    fdw = uniform_filter_3d_batch(dIdw)

    # --- Cartesian derivatives ---
    dId_1 = 0.5 * (fdv - fdu + dIdv - dIdu)
    dId_2 = 0.5 * (fdw - fdv + dIdw - dIdv)
    dId_0 = 0.5 * (fdu - fdw + dIdu - dIdw)

    # --- Gradient magnitude ---
    mag_dI = fdu**2 + fdv**2 + fdw**2                     # (B,D-1,H-1,W-1)
    sdI3 = mag_dI.sum(dim=(1, 2, 3)) + eps                # (B,)

    # --- Coordinate vectors ---
    n0, n1, n2 = mag_dI.shape[1:]
    v0 = torch.arange(-(n0)/2 + 0.5, (n0)/2 + 0.5, device=device, dtype=dtype)
    v1 = torch.arange(-(n1)/2 + 0.5, (n1)/2 + 0.5, device=device, dtype=dtype)
    v2 = torch.arange(-(n2)/2 + 0.5, (n2)/2 + 0.5, device=device, dtype=dtype)
    
    V0 = v0[None, :, None, None]
    V1 = v1[None, None, :, None]
    V2 = v2[None, None, None, :]
    # --- Initial guess (batch einsum) ---
    guess_0 = torch.einsum("bijk,i->b", mag_dI, v0) / sdI3
    guess_1 = torch.einsum("bijk,j->b", mag_dI, v1) / sdI3
    guess_2 = torch.einsum("bijk,k->b", mag_dI, v2) / sdI3

    # --- Weights ---
    denom = torch.clamp(dId_0**2 + dId_1**2 + dId_2**2, min=eps)

    dist = (
    (V0 - guess_0[:, None, None, None])**2
    + (V1 - guess_1[:, None, None, None])**2
    + (V2 - guess_2[:, None, None, None])**2
    ) + eps

    W = mag_dI / denom * torch.rsqrt(dist)

    # --- Omega matrices ---
    Omega_11 = -2 * (dId_1**2 + dId_2**2)
    Omega_22 = -2 * (dId_2**2 + dId_0**2)
    Omega_33 = -2 * (dId_1**2 + dId_0**2)
    Omega_12 = 2 * dId_1 * dId_0
    Omega_13 = 2 * dId_2 * dId_0
    Omega_23 = 2 * dId_2 * dId_1

    # --- Matrix M ---
    M_11 = torch.sum(W * Omega_11, dim=(1, 2, 3))
    M_12 = torch.sum(W * Omega_12, dim=(1, 2, 3))
    M_13 = torch.sum(W * Omega_13, dim=(1, 2, 3))
    M_22 = torch.sum(W * Omega_22, dim=(1, 2, 3))
    M_23 = torch.sum(W * Omega_23, dim=(1, 2, 3))
    M_33 = torch.sum(W * Omega_33, dim=(1, 2, 3))
    M = torch.stack(
        [
            torch.stack([M_11, M_12, M_13], dim=1),
            torch.stack([M_12, M_22, M_23], dim=1),
            torch.stack([M_13, M_23, M_33], dim=1),
        ],
        dim=1,
        )   # (B, 3, 3)
    condit = torch.linalg.cond(M)
    if "conditionnement" in log:
        logger.debug(
            f"Conditioning stats → min: {condit.min().item():.2e}, "
            f"max: {condit.max().item():.2e}, "
            f"mean: {condit.mean().item():.2e}"
        )

        threshold = 1e8
        bad_mask = condit > threshold

        if bad_mask.any():
            n_bad = bad_mask.sum().item()
            logger.warning(
                f"{n_bad} ill-conditioned matrices detected "
                f"(cond > {threshold:.0e}). "
                f"Max cond = {condit.max().item():.2e}"
            )
    # --- Vector B ---
    B0 = torch.sum(W * (
        Omega_11 * v0[None, :, None, None]
        + Omega_12 * v1[None, None, :, None]
        + Omega_13 * v2[None, None, None, :]
    ), dim=(1, 2, 3))

    B1 = torch.sum(W * (
        Omega_12 * v0[None, :, None, None]
        + Omega_22 * v1[None, None, :, None]
        + Omega_23 * v2[None, None, None, :]
    ), dim=(1, 2, 3))

    B2 = torch.sum(W * (
        Omega_13 * v0[None, :, None, None]
        + Omega_23 * v1[None, None, :, None]
        + Omega_33 * v2[None, None, None, :]
    ), dim=(1, 2, 3))

    if inverse_matrix_version == "pseudo_inverse V1":
        M = torch.stack(
        [
            torch.stack([M_11, M_12, M_13], dim=1),
            torch.stack([M_12, M_22, M_23], dim=1),
            torch.stack([M_13, M_23, M_33], dim=1),
        ],
        dim=1,
        )   # (B, 3, 3)

        B = torch.stack([B0, B1, B2], dim=1)   # (B, 3)

        # --- Robust solve ---
        cond = torch.linalg.cond(M)

        superloc = torch.empty_like(B)

        good = cond < 1e8

        # Fast + precise
        superloc[good] = torch.linalg.solve(M[good], B[good])

        # Stable fallback (continuous)
        superloc[~good] = torch.linalg.lstsq(M[~good], B[~good]).solution

        return superloc
    elif inverse_matrix_version == "pseudo_inverse V2":
        M = torch.stack(
        [
            torch.stack([M_11, M_12, M_13], dim=1),
            torch.stack([M_12, M_22, M_23], dim=1),
            torch.stack([M_13, M_23, M_33], dim=1),
        ],
        dim=1,
        )   # (B, 3, 3)

        B = torch.stack([B0, B1, B2], dim=1)   # (B, 3)
        # --- Robust solve (regularized) ---
        eye = torch.eye(3, device=device, dtype=dtype)[None]
        superloc = torch.linalg.solve(M + reg_eps * eye, B)

        return superloc
    else :

        # --- Inverse symmetric matrix ---
        alpha = M_22 * M_33 - M_23**2
        delta = M_11 * M_33 - M_13**2
        phi   = M_11 * M_22 - M_12**2
        beta  = M_12 * M_33 - M_13 * M_23
        gamma = M_12 * M_23 - M_13 * M_22
        epsilon_m = M_11 * M_23 - M_13 * M_12

        det = M_11 * alpha - M_12 * beta + M_13 * gamma + eps

        # --- Final super-localization ---
        x = ( alpha * B0 - beta * B1 + gamma * B2) / det
        y = (-beta * B0 + delta * B1 - epsilon_m * B2) / det
        z = ( gamma * B0 - epsilon_m * B1 + phi * B2) / det

        superloc = torch.stack([x, y, z], dim=1)   # (B,3)

        # fallback NaN → guess
        nan_mask = torch.isnan(superloc).any(dim=1)
        superloc[nan_mask, 0] = guess_0[nan_mask]
        superloc[nan_mask, 1] = guess_1[nan_mask]
        superloc[nan_mask, 2] = guess_2[nan_mask]

        return superloc
    
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
