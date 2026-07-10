import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FAAFusionConcat(nn.Module):
    """Fourier Angle Alignment followed by YOLO-style channel concatenation."""

    def __init__(
        self,
        c_high: int,
        c_low: int,
        dim: int = 1,
        m: int = 7,
        c_mid: int = 16,
        eps: float = 1e-8,
        layer_scale_init_value: float = 1e-3,
        max_angle: float = math.pi / 4,
    ):
        super().__init__()

        if m % 2 != 1:
            raise ValueError("FAAFusionConcat window size `m` must be odd.")

        self.c_high = c_high
        self.c_low = c_low
        self.dim = dim
        self.m = m
        self.c_mid = c_mid
        self.eps = eps
        self.max_angle = max_angle

        self.layer_scale = nn.Parameter(torch.full((1, 1, 1, 1), layer_scale_init_value), requires_grad=True)
        self.proj_high = nn.Conv2d(c_high, c_mid, kernel_size=1, bias=False)
        self.proj_low = nn.Conv2d(c_low, c_mid, kernel_size=1, bias=False)
        self.recon = nn.Conv2d(c_mid, c_high, kernel_size=1, bias=False)

        self._init_freq_grids(m)

    def _init_freq_grids(self, m: int):
        h_freq = torch.fft.fftfreq(m, d=1.0) * m
        w_freq = torch.fft.fftfreq(m, d=1.0) * m
        h_grid, w_grid = torch.meshgrid(h_freq, w_freq, indexing="ij")

        rho = torch.sqrt(h_grid**2 + w_grid**2)
        theta = torch.atan2(h_grid, w_grid)
        mask = rho > self.eps

        self.register_buffer("valid_thetas", theta[mask])
        self.register_buffer("valid_rhos", rho[mask])
        self.register_buffer("mask_flat", mask.reshape(-1))

    def _estimate_main_direction(self, x_local: torch.Tensor) -> torch.Tensor:
        """Estimate patch dominant direction from the Fourier magnitude spectrum."""
        dtype = x_local.dtype
        bn = x_local.shape[0]

        x_fft = torch.fft.fft2(x_local.squeeze(1).float(), norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft.abs() + self.eps

        mag_flat = mag.reshape(bn, -1)
        mag_valid = mag_flat[:, self.mask_flat]
        rho_valid = self.valid_rhos.to(device=x_local.device, dtype=mag_valid.dtype)
        theta_valid = self.valid_thetas.to(device=x_local.device, dtype=mag_valid.dtype)

        weighted_energy = mag_valid * rho_valid.unsqueeze(0)
        max_idx = torch.argmax(weighted_energy, dim=1)
        return theta_valid[max_idx].to(dtype=dtype)

    def _rotate_spatial_patch(self, patch: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Rotate local patches with a normalized affine grid."""
        k = patch.shape[0]
        dtype = patch.dtype
        device = patch.device
        theta = theta.to(device=device, dtype=dtype)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rot_mat = torch.zeros(k, 2, 3, device=device, dtype=dtype)
        rot_mat[:, 0, 0] = cos_t
        rot_mat[:, 0, 1] = -sin_t
        rot_mat[:, 1, 0] = sin_t
        rot_mat[:, 1, 1] = cos_t

        grid = F.affine_grid(rot_mat, patch.size(), align_corners=False)
        return F.grid_sample(patch, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _align_high_to_low(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """Align high-level semantic features to low-level directional cues."""
        b, _, h, w = x_low.shape
        pad = self.m // 2
        n = h * w

        xh_proj = self.proj_high(x_high)
        xl_proj = self.proj_low(x_low)
        xh_aligned_cmid = torch.zeros_like(xh_proj)

        ones = torch.ones(1, 1, h, w, device=x_low.device, dtype=x_low.dtype)
        ones_unfold = F.unfold(ones, kernel_size=self.m, stride=1, padding=pad)
        ones_fold = F.fold(ones_unfold, output_size=(h, w), kernel_size=self.m, stride=1, padding=pad)

        for c in range(self.c_mid):
            xl_c = xl_proj[:, c : c + 1]
            xh_c = xh_proj[:, c : c + 1]

            xl_unfold = F.unfold(xl_c, kernel_size=self.m, stride=1, padding=pad)
            xh_unfold = F.unfold(xh_c, kernel_size=self.m, stride=1, padding=pad)

            xl_patches = xl_unfold.transpose(1, 2).reshape(b * n, 1, self.m, self.m)
            xh_patches = xh_unfold.transpose(1, 2).reshape(b * n, 1, self.m, self.m)

            theta_low = torch.remainder(self._estimate_main_direction(xl_patches), math.pi)
            theta_high = torch.remainder(self._estimate_main_direction(xh_patches), math.pi)
            delta_theta = theta_low - theta_high
            delta_theta = (delta_theta + math.pi / 2) % math.pi - math.pi / 2
            delta_theta = torch.clamp(delta_theta, -self.max_angle, self.max_angle)

            xh_rotated = self._rotate_spatial_patch(xh_patches, delta_theta)
            xh_rotated_flat = xh_rotated.reshape(b, n, -1).transpose(1, 2)
            xh_aligned_map = F.fold(
                xh_rotated_flat,
                output_size=(h, w),
                kernel_size=self.m,
                stride=1,
                padding=pad,
            )
            xh_aligned_cmid[:, c : c + 1] = xh_aligned_map / (ones_fold + self.eps)

        xh_recon = self.recon(xh_aligned_cmid)
        return x_high + self.layer_scale * (xh_recon - x_high)

    def forward(self, x) -> torch.Tensor:
        """Align x[0] to x[1], then concatenate them along `dim`."""
        if len(x) != 2:
            raise ValueError("FAAFusionConcat expects exactly two inputs: [x_high, x_low].")

        x_high, x_low = x[0], x[1]
        if x_high.shape[-2:] != x_low.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode="nearest")

        x_high_aligned = self._align_high_to_low(x_high, x_low)
        return torch.cat([x_high_aligned, x_low], dim=self.dim)
