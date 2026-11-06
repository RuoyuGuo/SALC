"""CLIP DCA Sparsegen with train-time donor LayerDrop."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import BACKBONES

from ..dino_v2 import DinoVisionTransformer
from ..pruning.adapter import DeltaBlock
from .models import LayerNorm, Transformer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * x / (rms + self.eps)


class DropSparsegen(nn.Module):
    """Sparsegen with train-time random or margin-based donor LayerDrop."""

    def __init__(
        self,
        dim,
        num_layers,
        mlp_hidden=64,
        drop_mode='margin',
        random_drop_rate=0.2,
        usage_margin=0.15,
        max_drop_rate=0.4,
        usage_momentum=0.99,
        usage_source='ema',
        drop_gamma=2.0,
        active_eps=1e-6,
        warmup_iters=500,
        min_layers_to_drop=4,
        min_active_layers_for_drop=1.5,
        renorm_after_drop=True,
    ):
        super().__init__()
        assert drop_mode in ('margin', 'random', 'usage'), \
            f'invalid drop_mode: {drop_mode}'
        assert usage_source in ('ema', 'batch'), \
            f'invalid usage_source: {usage_source}'
        assert 0.0 <= random_drop_rate < 1.0, \
            f'invalid random_drop_rate: {random_drop_rate}'
        assert 0.0 <= usage_margin < 1.0, \
            f'invalid usage_margin: {usage_margin}'
        assert 0.0 <= max_drop_rate < 1.0, \
            f'invalid max_drop_rate: {max_drop_rate}'
        assert 0.0 < usage_momentum < 1.0, \
            f'invalid usage_momentum: {usage_momentum}'
        assert drop_gamma > 0.0, f'invalid drop_gamma: {drop_gamma}'
        assert active_eps >= 0.0, f'invalid active_eps: {active_eps}'
        assert warmup_iters >= 0, f'invalid warmup_iters: {warmup_iters}'
        assert min_layers_to_drop >= 1, \
            f'invalid min_layers_to_drop: {min_layers_to_drop}'
        assert min_active_layers_for_drop >= 0.0, \
            f'invalid min_active_layers_for_drop: {min_active_layers_for_drop}'

        self.dim = dim
        self.num_layers = num_layers
        self.eps = 1e-2
        self.drop_mode = drop_mode
        self.random_drop_rate = random_drop_rate
        self.usage_margin = usage_margin
        self.max_drop_rate = max_drop_rate
        self.usage_momentum = usage_momentum
        self.usage_source = usage_source
        self.drop_gamma = drop_gamma
        self.active_eps = active_eps
        self.warmup_iters = warmup_iters
        self.min_layers_to_drop = min_layers_to_drop
        self.min_active_layers_for_drop = min_active_layers_for_drop
        self.renorm_after_drop = renorm_after_drop

        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, 1, bias=False)
        self.lambda_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1, bias=True),
            nn.LogSigmoid(),
        )

        self.register_buffer('usage_ema', torch.zeros(num_layers))
        self.register_buffer(
            'active_layers_ema',
            torch.tensor(float(num_layers)),
        )
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lambda_mlp[0].weight, a=math.sqrt(5))
        nn.init.constant_(self.lambda_mlp[0].bias, -2.5)
        nn.init.kaiming_uniform_(self.lambda_mlp[2].weight, a=math.sqrt(5))
        nn.init.constant_(self.lambda_mlp[2].bias, -2.5)

    def _sparsegen_project(self, z, lam):
        Y, _, _ = z.shape
        orig_dtype = z.dtype
        z = z.float()
        lam = lam.float().squeeze(-1).unsqueeze(0)
        z_sorted, _ = torch.sort(z, dim=0, descending=True)
        z_cumsum = torch.cumsum(z_sorted, dim=0)
        k_range = torch.arange(
            1,
            Y + 1,
            device=z.device,
            dtype=z.dtype,
        ).view(Y, 1, 1)
        z_check = torch.gt(1.0 - lam + k_range * z_sorted, z_cumsum)
        k_z = z_check.float().sum(dim=0).clamp_min(1)
        tausum = (z_check.float() * z_sorted).sum(dim=0)
        lam_2d = lam.squeeze(0)
        tau_z = ((tausum - 1.0 + lam_2d) / k_z).unsqueeze(0)
        prob = z.sub(tau_z).clamp(min=0)
        prob = prob / torch.clamp(1.0 - lam, min=self.eps)
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            print('[DropSparsegen] WARNING: prob has NaN/Inf, clamping')
            prob = torch.nan_to_num(
                prob,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        return prob.to(orig_dtype)

    def _is_layerdrop_active(self):
        return (
            self.training
            and self.num_layers >= self.min_layers_to_drop
            and self.step_count.item() >= self.warmup_iters
        )

    @staticmethod
    def _usage_weights(weights):
        # Keep CLIP cls in the donor-usage statistics for apples-to-apples
        # comparisons with the original sparsegen behavior.
        return weights

    @torch.no_grad()
    def _update_usage_ema(self, weights):
        usage_weights = self._usage_weights(weights.detach()).float()
        batch_usage = usage_weights.mean(dim=(1, 2))
        self.usage_ema.mul_(self.usage_momentum).add_(
            batch_usage * (1.0 - self.usage_momentum)
        )

        active_count = (usage_weights > self.active_eps).float().sum(dim=0)
        mean_active_layers = active_count.mean().clamp(min=1.0)
        self.active_layers_ema.mul_(self.usage_momentum).add_(
            mean_active_layers * (1.0 - self.usage_momentum)
        )
        self.step_count += 1

    @torch.no_grad()
    def _get_usage_and_active_layers(self, weights=None, device=None):
        if weights is not None:
            device = weights.device
        device = device or self.usage_ema.device

        if self.usage_source == 'batch' and weights is not None:
            usage_weights = self._usage_weights(weights.detach()).float()
            usage = usage_weights.mean(dim=(1, 2)).clamp(min=0.0)
            active_count = (usage_weights > self.active_eps).float().sum(dim=0)
            active_layers = active_count.mean().clamp(min=1.0).to(device)
            return usage.to(device), active_layers

        usage = self.usage_ema.float().to(device).clamp(min=0.0)
        active_layers = self.active_layers_ema.float().to(device).clamp(
            min=1.0)
        return usage, active_layers

    @torch.no_grad()
    def _compute_drop_probs(self, weights=None, device=None):
        if weights is not None:
            device = weights.device
        device = device or self.usage_ema.device
        if not self._is_layerdrop_active():
            return torch.zeros(
                self.num_layers,
                device=device,
                dtype=torch.float32,
            )

        usage, active_layers = self._get_usage_and_active_layers(
            weights=weights,
            device=device,
        )
        if active_layers <= self.min_active_layers_for_drop:
            return torch.zeros(
                self.num_layers,
                device=device,
                dtype=torch.float32,
            )

        if self.drop_mode == 'random':
            return torch.full(
                (self.num_layers,),
                self.random_drop_rate,
                device=device,
                dtype=torch.float32,
            )

        usage_sum = usage.sum()
        if usage_sum <= 0:
            return torch.zeros(
                self.num_layers,
                device=device,
                dtype=torch.float32,
            )

        usage = usage / usage_sum.clamp_min(self.eps)
        target = torch.clamp(
            1.0 / active_layers + self.usage_margin,
            max=1.0 - self.eps,
        )
        relative_overuse = ((usage - target) / (1.0 - target)).clamp(
            min=0.0,
            max=1.0,
        )
        drop_probs = self.max_drop_rate * relative_overuse.pow(
            self.drop_gamma)
        return drop_probs.clamp(
            min=0.0,
            max=self.max_drop_rate,
        )

    def _apply_layerdrop(self, weights):
        Y, _, _ = weights.shape
        device = weights.device
        dtype = weights.dtype
        drop_probs = self._compute_drop_probs(weights=weights).to(dtype=dtype)
        keep_mask = torch.bernoulli(1.0 - drop_probs).to(
            device=device,
            dtype=dtype,
        )
        if keep_mask.sum() == 0:
            keep_mask[torch.randint(Y, (1,), device=device)] = 1.0

        masked_weights = weights * keep_mask.view(Y, 1, 1)
        if not self.renorm_after_drop:
            return masked_weights

        weight_sum = masked_weights.sum(dim=0, keepdim=True)
        needs_fallback = weight_sum < self.eps
        num_kept = keep_mask.sum().clamp(min=1.0)
        uniform_fallback = (
            keep_mask.view(Y, 1, 1) / num_kept
        ).expand_as(weights)
        normal_weights = masked_weights / weight_sum.clamp(min=self.eps)
        return torch.where(needs_fallback, uniform_fallback, normal_weights)

    def forward(self, tokens_across_depth, mask_idx=None):
        Y, _, _, _ = tokens_across_depth.shape
        assert Y == self.num_layers
        scores = self.proj(self.norm(tokens_across_depth)).squeeze(-1)
        current_token = tokens_across_depth[-1]
        lam_raw = self.lambda_mlp(current_token)
        lam = lam_raw + (1.0 - self.eps)
        weights = self._sparsegen_project(scores, lam)
    
        if mask_idx is not None:
            ids = [mask_idx] if isinstance(mask_idx, int) else list(mask_idx)
            ids = [int(j) for j in ids if 0 <= int(j) < Y]
            if ids:
                keep = torch.ones(Y, device=weights.device, dtype=weights.dtype)
                keep[torch.as_tensor(ids, device=weights.device)] = 0.0
                weights = weights * keep.view(-1, 1, 1)
    
        if self.training:
            self._update_usage_ema(weights)
            if self._is_layerdrop_active():
                weights = self._apply_layerdrop(weights)
                
        return (tokens_across_depth * weights.unsqueeze(-1)).sum(dim=0)

    def get_forward(self, tokens_across_depth):
        Y, B, N, D = tokens_across_depth.shape
        assert Y == self.num_layers
        scores = self.proj(self.norm(tokens_across_depth)).squeeze(-1)
        current_token = tokens_across_depth[-1]
        lam_raw = self.lambda_mlp(current_token)
        lam = lam_raw + (1.0 - self.eps)
        weights = self._sparsegen_project(scores, lam)

        if self.training:
            self._update_usage_ema(weights)
            if self._is_layerdrop_active():
                weights = self._apply_layerdrop(weights)

        return (tokens_across_depth * weights.unsqueeze(-1)).sum(dim=0), weights, lam
    
    @torch.no_grad()
    def get_diagnostics(self):
        usage = self.usage_ema.float().clamp(min=0.0)
        usage_norm = usage / usage.sum().clamp_min(self.eps)
        active_layers = self.active_layers_ema.float().clamp(min=1.0)
        target = torch.clamp(
            1.0 / active_layers + self.usage_margin,
            max=1.0 - self.eps,
        )
        drop_probs = self._compute_drop_probs(device=usage.device)
        return dict(
            step=int(self.step_count.item()),
            warmup_done=bool(self.step_count.item() >= self.warmup_iters),
            active=bool(self._is_layerdrop_active()),
            drop_mode=self.drop_mode,
            usage_source=self.usage_source,
            usage_ema=usage.cpu().tolist(),
            usage_norm=usage_norm.cpu().tolist(),
            active_layers_ema=float(active_layers.item()),
            min_active_layers_for_drop=float(
                self.min_active_layers_for_drop),
            active_support_ok=bool(
                active_layers.item() > self.min_active_layers_for_drop),
            active_usage_target=float(target.item()),
            drop_probs=drop_probs.cpu().tolist(),
            max_drop_prob=float(drop_probs.max().item()),
            random_drop_rate=float(self.random_drop_rate),
            usage_margin=float(self.usage_margin),
            max_drop_rate=float(self.max_drop_rate),
            drop_gamma=float(self.drop_gamma),
            renorm_after_drop=bool(self.renorm_after_drop),
        )


@BACKBONES.register_module()
class CLIPVisionTransformerDCA_DropSparsegen(nn.Module):
    """Standalone CLIP DCA backbone using DropSparsegen."""

    def __init__(
        self,
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        drop_path_rate=0.0,
        out_indices=[3, 5, 7, 11],
        pretrained=None,
        get_embeddings=False,
        ignore_last_attn=False,
        disable_vlm_adapter=None,
        disable_vfm_adapter=None,
        sparsegen_drop_mode='margin',
        sparsegen_random_drop_rate=0.2,
        sparsegen_usage_margin=0.15,
        sparsegen_max_drop_rate=0.4,
        sparsegen_usage_momentum=0.99,
        sparsegen_usage_source='ema',
        sparsegen_drop_gamma=2.0,
        sparsegen_active_eps=1e-6,
        sparsegen_warmup_iters=500,
        sparsegen_min_layers_to_drop=4,
        sparsegen_min_active_layers_for_drop=1.5,
        sparsegen_renorm_after_drop=True,

        
        close_layer_vlm = None,
        close_layer_vfm = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if disable_vlm_adapter == 'all':
            self.disable_vlm_layers = set(range(layers))
        else:
            self.disable_vlm_layers = set(disable_vlm_adapter or [])
        if disable_vfm_adapter == 'all':
            self.disable_vfm_layers = set(range(layers))
        else:
            self.disable_vfm_layers = set(disable_vfm_adapter or [])

        self.embed_dim = width
        self.output_dim = output_dim
        self.pretrained = pretrained
        self.patch_size = patch_size
        self.layers = layers
        self.heads = heads

        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(
            3,
            width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (self.input_resolution[0] // patch_size)
                * (self.input_resolution[1] // patch_size) + 1,
                width,
            )
        )
        self.spatial_size = (
            self.input_resolution[0] // patch_size,
            self.input_resolution[1] // patch_size,
        )
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(
            width,
            layers,
            heads,
            drop_path_rate=drop_path_rate,
        )
        self.out_indices = out_indices
        self.ignore_last_attn = ignore_last_attn

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.fpn_dim = width + 1024
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, 2, stride=2),
            nn.SyncBatchNorm(self.fpn_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, 2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, 2, stride=2)
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dinov2 = DinoVisionTransformer(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            img_size=512,
            ffn_layer='mlp',
            init_values=1e-05,
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        self._load_dinov2()

        self.vlm_adapter = nn.ModuleList(
            [DeltaBlock(dim=width, bottleneck_dim=64) for _ in range(layers)]
        )
        self.vfm_adapter = nn.ModuleList(
            [DeltaBlock(dim=1024, bottleneck_dim=64) for _ in range(layers)]
        )
        self.adapter_proj3 = nn.Linear(1024, self.embed_dim)

        assert close_layer_vlm is None or isinstance(close_layer_vlm, list), \
            f"close_layer_vlm must be list or None, got {type(close_layer_vlm)}"
        assert close_layer_vfm is None or isinstance(close_layer_vfm, list), \
            f"close_layer_vfm must be list or None, got {type(close_layer_vfm)}"
        
        self.close_layer_vlm = close_layer_vlm
        self.close_layer_vfm = close_layer_vfm
        
        self.sparsegen_for_vlm = nn.ModuleList()
        self.sparsegen_for_vfm = nn.ModuleList()
        for i in range(layers):
            n_input = i + 1
            self.sparsegen_for_vlm.append(
                DropSparsegen(
                    dim=1024,
                    num_layers=n_input,
                    drop_mode=sparsegen_drop_mode,
                    random_drop_rate=sparsegen_random_drop_rate,
                    usage_margin=sparsegen_usage_margin,
                    max_drop_rate=sparsegen_max_drop_rate,
                    usage_momentum=sparsegen_usage_momentum,
                    usage_source=sparsegen_usage_source,
                    drop_gamma=sparsegen_drop_gamma,
                    active_eps=sparsegen_active_eps,
                    warmup_iters=sparsegen_warmup_iters,
                    min_layers_to_drop=sparsegen_min_layers_to_drop,
                    min_active_layers_for_drop=(
                        sparsegen_min_active_layers_for_drop),
                    renorm_after_drop=sparsegen_renorm_after_drop,
                )
            )
            self.sparsegen_for_vfm.append(
                DropSparsegen(
                    dim=width,
                    num_layers=n_input,
                    drop_mode=sparsegen_drop_mode,
                    random_drop_rate=sparsegen_random_drop_rate,
                    usage_margin=sparsegen_usage_margin,
                    max_drop_rate=sparsegen_max_drop_rate,
                    usage_momentum=sparsegen_usage_momentum,
                    usage_source=sparsegen_usage_source,
                    drop_gamma=sparsegen_drop_gamma,
                    active_eps=sparsegen_active_eps,
                    warmup_iters=sparsegen_warmup_iters,
                    min_layers_to_drop=sparsegen_min_layers_to_drop,
                    min_active_layers_for_drop=(
                        sparsegen_min_active_layers_for_drop),
                    renorm_after_drop=sparsegen_renorm_after_drop,
                )
            )

    def _load_dinov2(self):
        dinov2_sd = torch.load('pretrained/dinov2_vitl14_pretrain.pth')
        if 'pos_embed' in dinov2_sd:
            pe = dinov2_sd['pos_embed']
            embed_sz = pe.shape[-1]
            n_patches = self.dinov2.patch_embed.num_patches
            n_extra = self.dinov2.pos_embed.shape[-2] - n_patches
            orig = int((pe.shape[-2] - n_extra) ** 0.5)
            new = int(n_patches ** 0.5)
            if orig != new:
                print(f'Position interpolate from {orig}x{orig} to {new}x{new}')
                extra = pe[:, :n_extra]
                pos = pe[:, n_extra:]
                pos = pos.reshape(-1, orig, orig, embed_sz).permute(0, 3, 1, 2)
                pos = F.interpolate(
                    pos,
                    size=(new, new),
                    mode='bicubic',
                    align_corners=False,
                )
                pos = pos.permute(0, 2, 3, 1).flatten(1, 2)
                dinov2_sd['pos_embed'] = torch.cat((extra, pos), dim=1)

                pw = dinov2_sd['patch_embed.proj.weight']
                ps = self.dinov2.patch_embed.patch_size
                dinov2_sd['patch_embed.proj.weight'] = F.interpolate(
                    pw.float(),
                    size=ps,
                    mode='bicubic',
                    align_corners=False,
                )
        self.dinov2.load_state_dict(dinov2_sd, strict=True)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        print('backbone:', pretrained)
        if isinstance(pretrained, str):
            ckpt = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            sd = {}
            for k in ckpt:
                if k.startswith('visual.'):
                    sd[k.replace('visual.', '')] = ckpt[k]

            if 'positional_embedding' in sd:
                if self.positional_embedding.shape != sd['positional_embedding'].shape:
                    print(
                        f'Resize pos_embed from {sd["positional_embedding"].shape}'
                        f' to {self.positional_embedding.shape}'
                    )
                    cls_pos = sd['positional_embedding'][0:1, :]
                    orig = int(sd['positional_embedding'][1:].shape[0] ** 0.5)
                    sp = F.interpolate(
                        sd['positional_embedding'][1:].reshape(
                            1,
                            orig,
                            orig,
                            self.embed_dim,
                        ).permute(0, 3, 1, 2),
                        size=self.spatial_size,
                        mode='bilinear',
                    )
                    sp = sp.reshape(
                        self.embed_dim,
                        self.spatial_size[0] * self.spatial_size[1],
                    ).permute(1, 0)
                    sd['positional_embedding'] = torch.cat([cls_pos, sp], dim=0)

            conv1_weight = sd.get('conv1.weight', self.conv1.weight)
            if self.conv1.weight.shape != conv1_weight.shape:
                print(
                    f'Resize patch_embed from {sd["conv1.weight"].shape}'
                    f' to {self.conv1.weight.shape}'
                )
                sd['conv1.weight'] = F.interpolate(
                    sd['conv1.weight'],
                    size=self.conv1.weight.shape[-2:],
                    mode='bilinear',
                )

            u, w = self.load_state_dict(sd, False)
            print(u, w, 'are misaligned params in vision transformer')

    @staticmethod
    def convert_list_to_tensor(lst):
        return torch.stack(lst, dim=1) if lst else None

    def forward(self, x: torch.Tensor, use_adapter=True, train_loss=False):
        dev = x.device
        img_mean = torch.tensor(
            [v * 255 for v in [0.48145466, 0.4578275, 0.40821073]],
            device=dev,
        ).view(1, 3, 1, 1)
        img_std = torch.tensor(
            [v * 255 for v in [0.26862954, 0.26130258, 0.27577711]],
            device=dev,
        ).view(1, 3, 1, 1)
        original_x = x * img_std + img_mean

        dino_mean = torch.tensor(
            [v * 255 for v in [0.485, 0.456, 0.406]],
            device=dev,
        ).view(1, 3, 1, 1)
        dino_std = torch.tensor(
            [v * 255 for v in [0.229, 0.224, 0.225]],
            device=dev,
        ).view(1, 3, 1, 1)
        dinov2_x = self.dinov2.prepare_tokens_with_masks(
            (original_x - dino_mean) / dino_std
        )

        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=dev),
                x,
            ],
            dim=1,
        )

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(
            pos[1:].reshape(
                1,
                self.spatial_size[0],
                self.spatial_size[1],
                C,
            ).permute(0, 3, 1, 2),
            size=(H, W),
            mode='bilinear',
        )
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        features = []
        vlm_history = []
        vfm_history = []
        vlm_feature_list = [] if train_loss else None
        vfm_feature_list = [] if train_loss else None

        for i, blk in enumerate(self.transformer.resblocks):
            if self.ignore_last_attn:
                mask = torch.empty(x.shape[0], x.shape[0])
                mask.fill_(float('-inf'))
                mask.fill_diagonal_(0)
                self.transformer.resblocks[-1].attn_mask = mask

            x = blk(x)
            dinov2_x = self.dinov2.blocks[i](dinov2_x)

            if use_adapter:
                x_bf = x.permute(1, 0, 2)

                all_vfm = vfm_history + [dinov2_x]
                all_vlm = vlm_history + [x_bf]

                # if i not in self.disable_vlm_layers:
                #     with torch.cuda.amp.autocast(enabled=False):
                #         vfm_cross = self.sparsegen_for_vlm[i](
                #             torch.stack(all_vfm, dim=0).float()
                #         )
                #         x_delta = self.vlm_adapter[i](vfm_cross)
                #     x = x + x_delta.permute(1, 0, 2).to(x.dtype)

                # if i not in self.disable_vfm_layers:
                #     with torch.cuda.amp.autocast(enabled=False):
                #         vlm_cross = self.sparsegen_for_vfm[i](
                #             torch.stack(all_vlm, dim=0).float()
                #         )
                #         dinov2_delta = self.vfm_adapter[i](vlm_cross)
                #     dinov2_x = dinov2_x + dinov2_delta.to(dinov2_x.dtype)

                
                # VFM→VLM 分支 (可关闭)
                # close_layer_vfm: VFM 的第 k 层不许被 VLM 借 → 屏蔽 sparsegen_for_vlm 的历史索引 k
                if i not in self.disable_vlm_layers:
                    # print(f'turn of VFM {self.close_layer_vfm}')
                    with torch.cuda.amp.autocast(enabled=False):
                        vfm_cross = self.sparsegen_for_vlm[i](
                            torch.stack(all_vfm, dim=0).float(),
                            mask_idx=self.close_layer_vfm)
                        x_delta = self.vlm_adapter[i](vfm_cross)
                    x = x + x_delta.permute(1, 0, 2).to(x.dtype)
                
                # VLM→VFM 分支 (可关闭)
                # close_layer_vlm: VLM 的第 k 层不许被 VFM 借 → 屏蔽 sparsegen_for_vfm 的历史索引 k
                if i not in self.disable_vfm_layers:
                    
                    # print(f'turn of VLM {self.close_layer_vlm}')
                    with torch.cuda.amp.autocast(enabled=False):
                        vlm_cross = self.sparsegen_for_vfm[i](
                            torch.stack(all_vlm, dim=0).float(),
                            mask_idx=self.close_layer_vlm)
                        dinov2_delta = self.vfm_adapter[i](vlm_cross)
                    dinov2_x = dinov2_x + dinov2_delta.to(dinov2_x.dtype)

                vlm_history.append(x.permute(1, 0, 2))
                vfm_history.append(dinov2_x)

            if train_loss:
                vlm_feature_list.append(x.permute(1, 0, 2))
                vfm_feature_list.append(dinov2_x)

            if i in self.out_indices:
                xp = torch.cat(
                    [
                        dinov2_x[:, 1:, :].permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                        .contiguous(),
                        x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                        .contiguous(),
                    ],
                    dim=1,
                )
                features.append(xp.contiguous())

        if train_loss:
            vlm_feature = self.convert_list_to_tensor(vlm_feature_list)[
                :, :, 1:, :
            ]
            vfm_feature = self.convert_list_to_tensor(vfm_feature_list)[
                :, :, 1:, :
            ]
        else:
            vlm_feature = None
            vfm_feature = None

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            x = x.permute(1, 0, 2) + self.adapter_proj3(dinov2_x)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(
                B,
                H,
                W,
                -1,
            ).permute(0, 3, 1, 2)
            features.append([global_embedding, visual_embedding])

        if train_loss:
            return tuple(features), dict(
                vlm_feature=vlm_feature,
                vfm_feature=vfm_feature,
            )
        return tuple(features)

    def compute_test(
            self,
            x: torch.Tensor,
            dino_only=False,
            clip_only=False,
            use_adapter=True,
            include_neck=False,
        ):
            """Inference-cost probe for single- vs dual-backbone comparison.
    
            Modes:
            - dino_only=True:  run only the DINOv2 branch (no CLIP, no SALC).
            - clip_only=True:  run only the CLIP branch (no DINOv2, no SALC).
            - default (dual):  run both branches;
                use_adapter=True  -> full SALC fusion (deployed setting),
                use_adapter=False -> two independent backbones (isolates the
                                     cost of the SALC modules themselves).
    
            The layer history is kept in preallocated buffers (written once per
            layer) instead of re-stacking the full history at every layer,
            which removes O(L^2) redundant tensor copies. Numerically identical
            to the stack-based version.
    
            By default the FPN neck and the embedding projection are skipped in
            ALL modes so that the timing isolates the encoder(+fusion) cost and
            stays comparable across modes. Set include_neck=True (dual mode
            only) to additionally run the FPN + embedding path.
            """
            assert not (dino_only and clip_only), \
                'dino_only and clip_only are mutually exclusive'
            run_clip = not dino_only
            run_dino = not clip_only
            if dino_only or clip_only:
                use_adapter = False  # SALC fusion requires both streams
    
            dev = x.device
            B = x.shape[0]
    
            if run_dino:
                img_mean = torch.tensor(
                    [v * 255 for v in [0.48145466, 0.4578275, 0.40821073]],
                    device=dev,
                ).view(1, 3, 1, 1)
                img_std = torch.tensor(
                    [v * 255 for v in [0.26862954, 0.26130258, 0.27577711]],
                    device=dev,
                ).view(1, 3, 1, 1)
                original_x = x * img_std + img_mean
    
                dino_mean = torch.tensor(
                    [v * 255 for v in [0.485, 0.456, 0.406]],
                    device=dev,
                ).view(1, 3, 1, 1)
                dino_std = torch.tensor(
                    [v * 255 for v in [0.229, 0.224, 0.225]],
                    device=dev,
                ).view(1, 3, 1, 1)
                dinov2_x = self.dinov2.prepare_tokens_with_masks(
                    (original_x - dino_mean) / dino_std
                )
    
            if run_clip:
                x = self.conv1(x)
                _, C, H, W = x.shape
                x = x.reshape(B, C, -1).permute(0, 2, 1)
                x = torch.cat(
                    [
                        self.class_embedding.to(x.dtype)
                        + torch.zeros(
                            B, 1, x.shape[-1], dtype=x.dtype, device=dev
                        ),
                        x,
                    ],
                    dim=1,
                )
    
                pos = self.positional_embedding.to(x.dtype)
                cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
                spatial_pos = F.interpolate(
                    pos[1:].reshape(
                        1,
                        self.spatial_size[0],
                        self.spatial_size[1],
                        C,
                    ).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode='bilinear',
                )
                spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
                pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
    
                x = x + pos
                x = self.ln_pre(x)
                x = x.permute(1, 0, 2)
                num_layers = len(self.transformer.resblocks)
            else:
                # spatial grid inferred from the DINOv2 token count
                n_patches = dinov2_x.shape[1] - 1
                H = W = int(n_patches ** 0.5)
                num_layers = len(self.dinov2.blocks)
    
            features = []
    
            # Preallocated history buffers (replaces per-layer torch.stack).
            # vfm_buf[i] / vlm_buf[i] hold the post-fusion features of layer i;
            # during layer i's own fusion they temporarily hold the pre-fusion
            # features, matching the original stack semantics exactly.
            if use_adapter:
                vfm_buf = torch.empty(
                    (num_layers, B, dinov2_x.shape[1], dinov2_x.shape[2]),
                    device=dev, dtype=torch.float32,
                )
                vlm_buf = torch.empty(
                    (num_layers, B, x.shape[0], x.shape[2]),
                    device=dev, dtype=torch.float32,
                )
    
            for i in range(num_layers):
                if run_clip:
                    if self.ignore_last_attn:
                        mask = torch.empty(x.shape[0], x.shape[0])
                        mask.fill_(float('-inf'))
                        mask.fill_diagonal_(0)
                        self.transformer.resblocks[-1].attn_mask = mask
                    x = self.transformer.resblocks[i](x)
    
                if run_dino:
                    dinov2_x = self.dinov2.blocks[i](dinov2_x)
    
                if use_adapter:
                    # write current (pre-fusion) features into the buffers
                    vfm_buf[i] = dinov2_x.float()
                    vlm_buf[i] = x.permute(1, 0, 2).float()
    
                    if i not in self.disable_vlm_layers:
                        with torch.cuda.amp.autocast(enabled=False):
                            vfm_cross = self.sparsegen_for_vlm[i](
                                vfm_buf[:i + 1]
                            )
                            x_delta = self.vlm_adapter[i](vfm_cross)
                        x = x + x_delta.permute(1, 0, 2).to(x.dtype)
    
                    if i not in self.disable_vfm_layers:
                        with torch.cuda.amp.autocast(enabled=False):
                            vlm_cross = self.sparsegen_for_vfm[i](
                                vlm_buf[:i + 1]
                            )
                            dinov2_delta = self.vfm_adapter[i](vlm_cross)
                        dinov2_x = dinov2_x + dinov2_delta.to(dinov2_x.dtype)
    
                    # overwrite with post-fusion features so later layers see
                    # the fused history (same as the original implementation)
                    vfm_buf[i] = dinov2_x.float()
                    vlm_buf[i] = x.permute(1, 0, 2).float()
    
                if i in self.out_indices:
                    parts = []
                    if run_dino:
                        parts.append(
                            dinov2_x[:, 1:, :].permute(0, 2, 1)
                            .reshape(B, -1, H, W)
                            .contiguous()
                        )
                    if run_clip:
                        parts.append(
                            x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1)
                            .reshape(B, -1, H, W)
                            .contiguous()
                        )
                    features.append(torch.cat(parts, dim=1).contiguous())
    
            if include_neck and run_clip and run_dino:
                ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
                for i in range(len(features)):
                    features[i] = ops[i](features[i])
    
                if self.get_embeddings:
                    x = x.permute(1, 0, 2) + self.adapter_proj3(dinov2_x)
                    x = self.ln_post(x)
                    x = x @ self.proj
    
                    global_embedding = x[:, :1]
                    visual_embedding = x[:, 1:].reshape(
                        B,
                        H,
                        W,
                        -1,
                    ).permute(0, 3, 1, 2)
                    features.append([global_embedding, visual_embedding])
    
            return tuple(features)
    
    def get_forward(self, x: torch.Tensor, use_adapter=True, train_loss=False):
        dev = x.device
        img_mean = torch.tensor(
            [v * 255 for v in [0.48145466, 0.4578275, 0.40821073]],
            device=dev,
        ).view(1, 3, 1, 1)
        img_std = torch.tensor(
            [v * 255 for v in [0.26862954, 0.26130258, 0.27577711]],
            device=dev,
        ).view(1, 3, 1, 1)
        original_x = x * img_std + img_mean

        dino_mean = torch.tensor(
            [v * 255 for v in [0.485, 0.456, 0.406]],
            device=dev,
        ).view(1, 3, 1, 1)
        dino_std = torch.tensor(
            [v * 255 for v in [0.229, 0.224, 0.225]],
            device=dev,
        ).view(1, 3, 1, 1)
        dinov2_x = self.dinov2.prepare_tokens_with_masks(
            (original_x - dino_mean) / dino_std
        )

        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=dev),
                x,
            ],
            dim=1,
        )

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(
            pos[1:].reshape(
                1,
                self.spatial_size[0],
                self.spatial_size[1],
                C,
            ).permute(0, 3, 1, 2),
            size=(H, W),
            mode='bilinear',
        )
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        features = []
        vlm_history = []
        vfm_history = []
        vlm_info_w = []
        vfm_info_w = []

        
        vlm_info_lam = []
        vfm_info_lam = []

        for i, blk in enumerate(self.transformer.resblocks):
            if self.ignore_last_attn:
                mask = torch.empty(x.shape[0], x.shape[0])
                mask.fill_(float('-inf'))
                mask.fill_diagonal_(0)
                self.transformer.resblocks[-1].attn_mask = mask

            x = blk(x)
            dinov2_x = self.dinov2.blocks[i](dinov2_x)

            if use_adapter:
                x_bf = x.permute(1, 0, 2)

                all_vfm = vfm_history + [dinov2_x]
                all_vlm = vlm_history + [x_bf]

                if i not in self.disable_vlm_layers:
                    with torch.cuda.amp.autocast(enabled=False):
                        vfm_cross, vlm_w, vlm_lam = self.sparsegen_for_vlm[i].get_forward(
                            torch.stack(all_vfm, dim=0).float()
                        )
                        x_delta = self.vlm_adapter[i](vfm_cross)
                    x = x + x_delta.permute(1, 0, 2).to(x.dtype)

                if i not in self.disable_vfm_layers:
                    with torch.cuda.amp.autocast(enabled=False):
                        vlm_cross, vfm_w, vfm_lam = self.sparsegen_for_vfm[i].get_forward(
                            torch.stack(all_vlm, dim=0).float()
                        )
                        dinov2_delta = self.vfm_adapter[i](vlm_cross)
                    dinov2_x = dinov2_x + dinov2_delta.to(dinov2_x.dtype)

                vlm_history.append(x.permute(1, 0, 2))
                vfm_history.append(dinov2_x)

                vlm_info_w.append(vlm_w)
                vfm_info_w.append(vfm_w)

                vlm_info_lam.append(vlm_lam)
                vfm_info_lam.append(vfm_lam)


            if train_loss:
                vlm_feature_list.append(x.permute(1, 0, 2))
                vfm_feature_list.append(dinov2_x)

            if i in self.out_indices:
                xp = torch.cat(
                    [
                        dinov2_x[:, 1:, :].permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                        .contiguous(),
                        x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1)
                        .reshape(B, -1, H, W)
                        .contiguous(),
                    ],
                    dim=1,
                )
                features.append(xp.contiguous())

        if train_loss:
            vlm_feature = self.convert_list_to_tensor(vlm_feature_list)[
                :, :, 1:, :
            ]
            vfm_feature = self.convert_list_to_tensor(vfm_feature_list)[
                :, :, 1:, :
            ]
        else:
            vlm_feature = None
            vfm_feature = None

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            x = x.permute(1, 0, 2) + self.adapter_proj3(dinov2_x)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(
                B,
                H,
                W,
                -1,
            ).permute(0, 3, 1, 2)
            features.append([global_embedding, visual_embedding])

        if train_loss:
            return tuple(features), dict(
                vlm_feature=vlm_feature,
                vfm_feature=vfm_feature,
            )
        return tuple(features), vlm_info_w, vlm_info_lam, vfm_info_w, vfm_info_lam
