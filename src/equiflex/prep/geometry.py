"""Geometric utilities for CB placement from backbone atoms."""

import torch
import math


# Ideal tetrahedral geometry constants
CA_CB_DIST = 1.521  # Angstroms
# CB sits opposite the bisector of N-CA-C, tilted out of backbone plane
# For ideal sp3 carbon: angle between bisector and CB direction
_CB_TILT_RAD = math.radians(54.75)
_COS_TILT = math.cos(_CB_TILT_RAD)
_SIN_TILT = math.sin(_CB_TILT_RAD)


def place_cb_ideal(
    n_pos: torch.Tensor,
    ca_pos: torch.Tensor,
    c_pos: torch.Tensor,
    ca_cb_dist: float = CA_CB_DIST,
) -> torch.Tensor:
    """Place CB using ideal tetrahedral geometry from backbone N, CA, C.

    CB direction: flip the N-CA / C-CA bisector, then tilt out of the
    backbone plane by ~54.75 degrees (tetrahedral half-angle).

    Works on any leading batch dimensions: [B, 3], [B, N, 3], etc.

    Args:
        n_pos, ca_pos, c_pos: backbone atom coordinates, same shape [..., 3]
        ca_cb_dist: CA-CB bond length

    Returns:
        CB coordinates, same shape as inputs
    """
    u_n = n_pos - ca_pos
    u_c = c_pos - ca_pos
    u_n = u_n / (u_n.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    u_c = u_c / (u_c.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    # Bisector of N-CA-C, flipped to point away from backbone
    bisector = -(u_n + u_c)
    bisector = bisector / (bisector.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    # Normal to backbone plane (N-CA-C)
    plane_normal = torch.cross(u_n, u_c, dim=-1)
    plane_normal = plane_normal / (plane_normal.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    # CB direction: tilt from bisector toward plane normal
    cb_dir = bisector * _COS_TILT + plane_normal * _SIN_TILT

    return ca_pos + ca_cb_dist * cb_dir
