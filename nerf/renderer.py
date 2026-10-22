import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        sigma_delta = rays_density * deltas
        alpha = 1.0 - torch.exp(-sigma_delta)

        T = torch.exp(-torch.cat(
            (torch.zeros_like(sigma_delta[:, :1]), torch.cumsum(sigma_delta[:, :-1], dim=1)),
            dim=1,
        ))

        weights = T * alpha

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights, cur_ray_bundle.sample_lengths)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    e = 0.5 * torch.exp(-torch.abs(signed_distance) / beta)
    return alpha * torch.where(signed_distance > 0, e, 1.0 - e)

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta)

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'feature': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out

class CoarseFineVolumeRenderer(VolumeRenderer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_pts_fine = cfg.n_pts_fine

    def _render_pass(self, implicit_fn, ray_bundle):
        n_pts = ray_bundle.sample_shape[1]
        implicit_output = implicit_fn(ray_bundle)
        density = implicit_output['density']
        feature = implicit_output['feature']

        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        weights = self._compute_weights(deltas.view(-1, n_pts, 1), density.view(-1, n_pts, 1))
        feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))
        depth = self._aggregate(weights, ray_bundle.sample_lengths)
        return feature, depth, weights

    def forward(self, sampler, implicit_fn, ray_bundle):
        implicit_fn_coarse, implicit_fn_fine = implicit_fn
        B = ray_bundle.shape[0]
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Coarse pass
            cur_ray_bundle = sampler(cur_ray_bundle)
            feature_coarse, _, weights = self._render_pass(implicit_fn_coarse, cur_ray_bundle)

            # Fine samples
            z_coarse = cur_ray_bundle.sample_lengths[..., 0]
            z_mids = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])
            z_fine = sample_pdf(
                z_mids, weights[:, 1:-1, 0].detach(), self.n_pts_fine,
                det=not torch.is_grad_enabled(),
            )
            z_all, _ = torch.sort(torch.cat([z_coarse, z_fine.detach()], dim=-1), dim=-1)

            # Fine pass
            cur_ray_bundle = cur_ray_bundle._replace(
                sample_points=cur_ray_bundle.origins[:, None, :] + z_all[..., None] * cur_ray_bundle.directions[:, None, :],
                sample_lengths=z_all[..., None],
            )
            feature, depth, _ = self._render_pass(implicit_fn_fine, cur_ray_bundle)

            chunk_outputs.append({
                'feature': feature,
                'feature_coarse': feature_coarse,
                'depth': depth,
            })

        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out    

def sample_pdf(bins, weights, n_samples, det=False, eps=1e-5):
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device).expand(cdf.shape[0], n_samples)
    else:
        u = torch.rand(cdf.shape[0], n_samples, device=bins.device)
    u = u.contiguous()

    idx = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(idx - 1, min=0)
    above = torch.clamp(idx, max=cdf.shape[-1] - 1)

    cdf_below, cdf_above = torch.gather(cdf, 1, below), torch.gather(cdf, 1, above)
    bin_below, bin_above = torch.gather(bins, 1, below), torch.gather(bins, 1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom

    return bin_below + t * (bin_above - bin_below)

renderer_dict = {
    'volume': VolumeRenderer,
    'volume_sdf': VolumeSDFRenderer,
    'volume_coarse_fine': CoarseFineVolumeRenderer,
}
