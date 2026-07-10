import torch
import math

def fourier_transform(points, K=3):
    """
    Perform Discrete Fourier Transform on a set of contour points.
    
    Args:
        points (torch.Tensor): Tensor of shape (..., N, 2) representing (x, y) coordinates.
        K (int): Order of Fourier series.
        
    Returns:
        coeffs (torch.Tensor): Tensor of shape (..., 2K+1, 2) representing complex coefficients (real, imag).
    """
    N = points.shape[-2]
    device = points.device
    dtype = points.dtype
    
    # x + i*y
    # Force float32 for complex operations to avoid ComplexHalf issues with AMP/Deterministic mode
    points_f32 = points.to(torch.float32)
    complex_points = torch.complex(points_f32[..., 0], points_f32[..., 1]) # (..., N)
    
    # k values from -K to K
    ks = torch.arange(-K, K + 1, device=device, dtype=torch.float32)
    ns = torch.arange(N, device=device, dtype=torch.float32)
    
    # Compute exponent: -i * 2 * pi * k * n / N in float32
    exponent = -2j * math.pi * ks[:, None] * ns[None, :] / N # (2K+1, N)
    
    # (..., 1, N) * (2K+1, N) -> (..., 2K+1, N)
    terms = complex_points.unsqueeze(-2) * torch.exp(exponent)
    
    # Sum over N points and divide by N
    coeffs_complex = torch.sum(terms, dim=-1) / N # (..., 2K+1)
    
    # Convert back to (real, imag) tensor and cast back to original dtype
    coeffs_real = torch.stack([coeffs_complex.real, coeffs_complex.imag], dim=-1) # (..., 2K+1, 2) keep float32
    
    return coeffs_real

def inverse_fourier_transform(coeffs, num_points=100, K=3):
    """
    Perform Inverse Fourier Transform to reconstruct contour points.
    
    Args:
        coeffs (torch.Tensor): Tensor of shape (..., 2K+1, 2) representing complex coefficients (real, imag).
        num_points (int): Number of points to reconstruct.
        K (int): Order of Fourier series.
        
    Returns:
        points (torch.Tensor): Tensor of shape (..., num_points, 2) representing (x, y) coordinates.
    """
    device = coeffs.device
    dtype = coeffs.dtype
    
    # Convert (real, imag) back to complex
    # Force float32 for complex operations to avoid ComplexHalf issues with AMP/Deterministic mode
    coeffs_f32 = coeffs.to(torch.float32)
    coeffs_complex = torch.complex(coeffs_f32[..., 0], coeffs_f32[..., 1]) # (..., 2K+1)
    
    ks = torch.arange(-K, K + 1, device=device, dtype=torch.float32)
    ns = torch.arange(num_points, device=device, dtype=torch.float32)
    
    # Compute exponent: i * 2 * pi * k * n / num_points in float32
    exponent = 2j * math.pi * ks[:, None] * ns[None, :] / num_points # (2K+1, num_points)
    
    # (..., 2K+1, 1) * (2K+1, num_points) -> (..., 2K+1, num_points)
    terms = coeffs_complex.unsqueeze(-1) * torch.exp(exponent)
    
    # Sum over coefficients
    points_complex = torch.sum(terms, dim=-2) # (..., num_points)
    
    # Convert back to (x, y) coordinates and cast back to original dtype
    points = torch.stack([points_complex.real, points_complex.imag], dim=-1) # (..., num_points, 2) keep float32
    
    return points
