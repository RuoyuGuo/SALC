import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES

from .modeling_siglip import (
    PytorchGELUTanh,
    SiglipEncoder,
    SiglipMultiheadAttentionPoolingHead,
    SiglipVisionEmbeddings,
)
from ..dino_v2 import DinoVisionTransformer
from ..pruning.adapter import DeltaBlock


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
        Y, B, N = z.shape
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

    @torch.no_grad()
    def _update_usage_ema(self, weights):
        batch_usage = weights.detach().float().mean(dim=(1, 2))
        self.usage_ema.mul_(self.usage_momentum).add_(
            batch_usage * (1.0 - self.usage_momentum)
        )

        active_count = (weights.detach() > self.active_eps).float().sum(dim=0)
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
            usage = weights.detach().float().mean(dim=(1, 2)).clamp(min=0.0)
            active_count = (weights.detach() > self.active_eps).float().sum(dim=0)
            active_layers = active_count.mean().clamp(min=1.0).to(device)
            return usage.to(device), active_layers

        usage = self.usage_ema.float().to(device).clamp(min=0.0)
        active_layers = self.active_layers_ema.float().to(device).clamp(min=1.0)
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
        drop_probs = self.max_drop_rate * relative_overuse.pow(self.drop_gamma)
        return drop_probs.clamp(
            min=0.0,
            max=self.max_drop_rate,
        )

    def _apply_layerdrop(self, weights):
        Y, B, N = weights.shape
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

    def forward(self, tokens_across_depth):
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


class SelfGateBlock(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0):
        super().__init__()
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.gate_layer = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.scale = scale

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.gate_layer[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate_layer[0].bias)
            nn.init.zeros_(self.gate_layer[2].weight)
            nn.init.zeros_(self.gate_layer[2].bias)

    def forward(self, x):
        delta = self.adapter(x)
        w = self.gate_layer(x)
        return self.scale * w * delta


@BACKBONES.register_module()
class SiglipVisionTransformerDCA_DropSparsegen(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act=PytorchGELUTanh(),
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        out_indices=[7, 11, 15, 23],
        pretrained=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        drop_path_rate=0.0,
        output_dim=None,
        get_embeddings=True,
        ignore_last_attn=False,
        detach_history=False,
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
        **kwargs,
    ):
        super().__init__()
        del drop_path_rate
        del kwargs
        self.image_size = image_size
        self.embed_dim = hidden_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.num_hidden_layers = num_hidden_layers
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.get_embeddings = get_embeddings
        self.ignore_last_attn = ignore_last_attn
        self.detach_history = detach_history
        self.spatial_size = (self.image_size // patch_size, self.image_size // patch_size)

        if disable_vlm_adapter == 'all':
            self.disable_vlm_layers = set(range(num_hidden_layers))
        else:
            self.disable_vlm_layers = set(disable_vlm_adapter or [])
        if disable_vfm_adapter == 'all':
            self.disable_vfm_layers = set(range(num_hidden_layers))
        else:
            self.disable_vfm_layers = set(disable_vfm_adapter or [])

        self.embeddings = SiglipVisionEmbeddings(
            hidden_size,
            image_size,
            patch_size,
            num_channels,
        )
        self.encoder = SiglipEncoder(
            hidden_size,
            hidden_act,
            num_attention_heads,
            intermediate_size,
            attention_dropout,
            layer_norm_eps,
            num_hidden_layers,
        )
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.use_head = True
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(
                hidden_size,
                num_attention_heads,
                layer_norm_eps,
                hidden_act,
                intermediate_size,
            )

        self.fpn_dim = self.embed_dim + 1024
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(self.fpn_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2)
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
            ffn_layer="mlp",
            init_values=1e-05,
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        self._load_dinov2()

        self.vlm_adapter = nn.ModuleList(
            [SelfGateBlock(dim=self.embed_dim, bottleneck_dim=64) for _ in range(num_hidden_layers)]
        )
        self.vfm_adapter = nn.ModuleList(
            [SelfGateBlock(dim=1024, bottleneck_dim=64) for _ in range(num_hidden_layers)]
        )
        self.adapter_proj3 = nn.Linear(1024, self.embed_dim)

        self.sparsegen_for_vlm = nn.ModuleList()
        self.sparsegen_for_vfm = nn.ModuleList()
        for i in range(num_hidden_layers):
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
                    dim=self.embed_dim,
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
        dinov2_state_dict = torch.load('pretrained/dinov2_vitl14_pretrain.pth')
        if 'pos_embed' in dinov2_state_dict:
            pos_embed_checkpoint = dinov2_state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.dinov2.patch_embed.num_patches
            num_extra_tokens = self.dinov2.pos_embed.shape[-2] - num_patches
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            new_size = int(num_patches ** 0.5)
            if orig_size != new_size:
                print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                dinov2_state_dict['pos_embed'] = torch.cat((extra_tokens, pos_tokens), dim=1)

                patch_embed_proj = dinov2_state_dict['patch_embed.proj.weight']
                patch = self.dinov2.patch_embed.patch_size
                dinov2_state_dict['patch_embed.proj.weight'] = F.interpolate(
                    patch_embed_proj.float(), size=patch, mode='bicubic', align_corners=False
                )
        self.dinov2.load_state_dict(dinov2_state_dict, strict=True)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        print('backbone:', pretrained)
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location='cpu')
            self.logit_scale = checkpoint['logit_scale']
            self.logit_bias = checkpoint['logit_bias']

            state_dict = {}
            for k in checkpoint.keys():
                if k.startswith('vision_model.'):
                    state_dict[k.replace('vision_model.', '')] = checkpoint[k]

            pos_key = 'embeddings.position_embedding.weight'
            if pos_key in state_dict:
                if self.embeddings.position_embedding.weight.shape != state_dict[pos_key].shape:
                    print(
                        f"Resize the pos_embed shape from {state_dict[pos_key].shape} "
                        f"to {self.embeddings.position_embedding.weight.shape}"
                    )
                    orig_size = int(state_dict[pos_key].shape[0] ** 0.5)
                    spatial_pos = F.interpolate(
                        state_dict[pos_key].reshape(1, orig_size, orig_size, self.embed_dim).permute(0, 3, 1, 2),
                        size=self.spatial_size,
                        mode='bilinear',
                    )
                    spatial_pos = spatial_pos.reshape(self.embed_dim, self.spatial_size[0] * self.spatial_size[1]).permute(1, 0)
                    state_dict[pos_key] = spatial_pos

            patch_key = 'embeddings.patch_embedding.weight'
            if patch_key in state_dict and self.embeddings.patch_embedding.weight.shape != state_dict[patch_key].shape:
                print(
                    f"Resize the patch_embed shape from {state_dict[patch_key].shape} "
                    f"to {self.embeddings.patch_embedding.weight.shape}"
                )
                state_dict[patch_key] = F.interpolate(
                    state_dict[patch_key],
                    size=self.embeddings.patch_embedding.weight.shape[-2:],
                    mode='bilinear',
                )

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    @staticmethod
    def convert_list_to_tensor(lst):
        return torch.stack(lst, dim=1) if lst else None

    def forward(
        self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        interpolate_pos_encoding=False,
        use_adapter=True,
        train_loss=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict
        del output_hidden_states
        del return_dict

        dev = pixel_values.device
        img_mean = torch.tensor([v * 255 for v in [0.48145466, 0.4578275, 0.40821073]], device=dev).view(1, 3, 1, 1)
        img_std = torch.tensor([v * 255 for v in [0.26862954, 0.26130258, 0.27577711]], device=dev).view(1, 3, 1, 1)
        original_pixel_values = pixel_values * img_std + img_mean

        dino_mean = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]], device=dev).view(1, 3, 1, 1)
        dino_std = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]], device=dev).view(1, 3, 1, 1)
        dinov2_x = self.dinov2.prepare_tokens_with_masks((original_pixel_values - dino_mean) / dino_std)

        siglip_mean = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]], device=dev).view(1, 3, 1, 1)
        siglip_std = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]], device=dev).view(1, 3, 1, 1)
        normalized_pixel_values = (original_pixel_values - siglip_mean) / siglip_std
        hidden_states = self.embeddings(normalized_pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        features = []
        vlm_history = []
        vfm_history = []
        vlm_feature_list = [] if train_loss else None
        vfm_feature_list = [] if train_loss else None

        for i, encoder_layer in enumerate(self.encoder.layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=output_attentions,
            )[0]
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
            dinov2_patch = dinov2_x[:, 1:, :]

            if use_adapter:
                all_vfm = vfm_history + [dinov2_patch]
                all_vlm = vlm_history + [hidden_states]

                if i not in self.disable_vlm_layers:
                    with torch.cuda.amp.autocast(enabled=False):
                        vfm_cross = self.sparsegen_for_vlm[i](torch.stack(all_vfm, dim=0).float())
                        x_delta = self.vlm_adapter[i](vfm_cross)
                    hidden_states = hidden_states + x_delta.to(hidden_states.dtype)

                if i not in self.disable_vfm_layers:
                    with torch.cuda.amp.autocast(enabled=False):
                        vlm_cross = self.sparsegen_for_vfm[i](torch.stack(all_vlm, dim=0).float())
                        dinov2_delta = self.vfm_adapter[i](vlm_cross)
                    dinov2_x = torch.cat([dinov2_x[:, :1, :], dinov2_patch + dinov2_delta.to(dinov2_patch.dtype)], dim=1)
                    dinov2_patch = dinov2_x[:, 1:, :]

                
                
                next_vlm = hidden_states.detach() if self.detach_history else hidden_states
                next_vfm = dinov2_patch.detach() if self.detach_history else dinov2_patch
                vlm_history.append(next_vlm)
                vfm_history.append(next_vfm)

            if train_loss:
                vlm_feature_list.append(hidden_states)
                vfm_feature_list.append(dinov2_patch)

            if i in self.out_indices:
                xp = torch.cat([
                    dinov2_patch.permute(0, 2, 1).reshape(dinov2_patch.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous(),
                    hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous(),
                ], dim=1)
                features.append(xp.contiguous())

        vlm_feature = self.convert_list_to_tensor(vlm_feature_list) if train_loss else None
        vfm_feature = self.convert_list_to_tensor(vfm_feature_list) if train_loss else None

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            last_hidden_state = self.post_layernorm(hidden_states) + self.dinov2.norm(dinov2_x[:, 1:, :])
            pooler_output = self.head(last_hidden_state) if self.use_head else None
            global_embedding = pooler_output[0]
            visual_embedding = pooler_output[1].reshape(
                last_hidden_state.shape[0], self.spatial_size[0], self.spatial_size[1], -1
            ).permute(0, 3, 1, 2)
            features.append([global_embedding, visual_embedding])

        if train_loss:
            return tuple(features), dict(vlm_feature=vlm_feature, vfm_feature=vfm_feature)
        return tuple(features)

    def get_forward(
        self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        interpolate_pos_encoding=False,
        use_adapter=True,
        train_loss=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict
        del output_hidden_states
        del return_dict

        dev = pixel_values.device
        img_mean = torch.tensor([v * 255 for v in [0.48145466, 0.4578275, 0.40821073]], device=dev).view(1, 3, 1, 1)
        img_std = torch.tensor([v * 255 for v in [0.26862954, 0.26130258, 0.27577711]], device=dev).view(1, 3, 1, 1)
        original_pixel_values = pixel_values * img_std + img_mean

        dino_mean = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]], device=dev).view(1, 3, 1, 1)
        dino_std = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]], device=dev).view(1, 3, 1, 1)
        dinov2_x = self.dinov2.prepare_tokens_with_masks((original_pixel_values - dino_mean) / dino_std)

        siglip_mean = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]], device=dev).view(1, 3, 1, 1)
        siglip_std = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]], device=dev).view(1, 3, 1, 1)
        normalized_pixel_values = (original_pixel_values - siglip_mean) / siglip_std
        hidden_states = self.embeddings(normalized_pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        features = []
        vlm_history = []
        vfm_history = []
        vlm_info_w = []
        vfm_info_w = []

        
        vlm_info_lam = []
        vfm_info_lam = []

        for i, encoder_layer in enumerate(self.encoder.layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=output_attentions,
            )[0]
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
            dinov2_patch = dinov2_x[:, 1:, :]

            if use_adapter:
                all_vfm = vfm_history + [dinov2_patch]
                all_vlm = vlm_history + [hidden_states]

                if i not in self.disable_vlm_layers:
                    with torch.cuda.amp.autocast(enabled=False):
                        vfm_cross, vlm_w, vlm_lam = self.sparsegen_for_vlm[i].get_forward(torch.stack(all_vfm, dim=0).float())
                        x_delta = self.vlm_adapter[i](vfm_cross)
                    hidden_states = hidden_states + x_delta.to(hidden_states.dtype)
                    
                if i not in self.disable_vfm_layers:
                    with torch.cuda.amp.autocast(enabled=False):
                        vlm_cross, vfm_w, vfm_lam = self.sparsegen_for_vfm[i].get_forward(torch.stack(all_vlm, dim=0).float())
                        dinov2_delta = self.vfm_adapter[i](vlm_cross)
                    dinov2_x = torch.cat([dinov2_x[:, :1, :], dinov2_patch + dinov2_delta.to(dinov2_patch.dtype)], dim=1)
                    dinov2_patch = dinov2_x[:, 1:, :]
                
                next_vlm = hidden_states.detach() if self.detach_history else hidden_states
                next_vfm = dinov2_patch.detach() if self.detach_history else dinov2_patch
                vlm_history.append(next_vlm)
                vfm_history.append(next_vfm)


                vlm_info_w.append(vlm_w)
                vfm_info_w.append(vfm_w)

                vlm_info_lam.append(vlm_lam)
                vfm_info_lam.append(vfm_lam)

            if i in self.out_indices:
                xp = torch.cat([
                    dinov2_patch.permute(0, 2, 1).reshape(dinov2_patch.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous(),
                    hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous(),
                ], dim=1)
                features.append(xp.contiguous())

        vlm_feature = self.convert_list_to_tensor(vlm_feature_list) if train_loss else None
        vfm_feature = self.convert_list_to_tensor(vfm_feature_list) if train_loss else None

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            last_hidden_state = self.post_layernorm(hidden_states) + self.dinov2.norm(dinov2_x[:, 1:, :])
            pooler_output = self.head(last_hidden_state) if self.use_head else None
            global_embedding = pooler_output[0]
            visual_embedding = pooler_output[1].reshape(
                last_hidden_state.shape[0], self.spatial_size[0], self.spatial_size[1], -1
            ).permute(0, 3, 1, 2)
            features.append([global_embedding, visual_embedding])

        return tuple(features), vlm_info_w, vlm_info_lam, vfm_info_w, vfm_info_lam

