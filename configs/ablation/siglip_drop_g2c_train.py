dist_params = dict(backend='nccl')
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=28000)
checkpoint_config = dict(by_epoch=False, interval=28000, max_keep_ckpts=1)
evaluation = dict(
    interval=4000, metric='mIoU', save_best='mIoU', rule='greater')
IMG_MEAN = [122.7709383, 116.7460125, 104.09373615000001]
IMG_VAR = [68.5005327, 66.6321579, 70.32316304999999]
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615000001],
    std=[68.5005327, 66.6321579, 70.32316304999999],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[122.7709383, 116.7460125, 104.09373615000001],
        std=[68.5005327, 66.6321579, 70.32316304999999],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[122.7709383, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316304999999],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
src_dataset_dict = dict(
    type='GTADataset',
    data_root='data/gta',
    img_dir='images',
    ann_dir='labels',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[122.7709383, 116.7460125, 104.09373615000001],
            std=[68.5005327, 66.6321579, 70.32316304999999],
            to_rgb=True),
        dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
        dict(type='ToMask'),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
    ])
tgt_dataset_dict = dict(
    type='CityscapesDataset',
    data_root='data/cityscapes',
    img_dir='leftImg8bit/val',
    ann_dir='gtFine/val',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[122.7709383, 116.7460125, 104.09373615000001],
                    std=[68.5005327, 66.6321579, 70.32316304999999],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=16,
    train=dict(
        type='UGDataset',
        source=dict(
            type='GTADataset',
            data_root='data/gta',
            img_dir='images',
            ann_dir='labels',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(type='Resize', ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[122.7709383, 116.7460125, 104.09373615000001],
                    std=[68.5005327, 66.6321579, 70.32316304999999],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='ToMask'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
            ]),
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=100, min_crop_ratio=0.5)),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
model = dict(
    type='MFuser_SIGLIP',
    pretrained='pretrained/siglip_vitl16_384.pth',
    token_embed_dim=1024,
    text_dim=1024,
    context_length=5,
    backbone=dict(
        type='SiglipVisionTransformerDCA_DropSparsegen',
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=512,
        intermediate_size=4096,
        attention_dropout=0.0,
        pretrained=None,
        out_indices=[7, 11, 15, 23],
        patch_size=16,
        sparsegen_drop_mode='margin',
        sparsegen_random_drop_rate=0.2,
        sparsegen_usage_margin=0.15,
        sparsegen_max_drop_rate=0.4,
        sparsegen_usage_momentum=0.99,
        sparsegen_usage_source='ema',
        sparsegen_drop_gamma=2.0,
        sparsegen_active_eps=1e-06,
        sparsegen_warmup_iters=500,
        sparsegen_min_layers_to_drop=4,
        sparsegen_renorm_after_drop=True),
    text_encoder=dict(
        type='SiglipTextTransformer',
        context_length=10,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        attention_dropout=0.0,
        pretrained=None),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=1,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    decode_head=dict(
        type='tqdmHead',
        in_channels=[2048, 2048, 2048, 2048],
        feat_channels=256,
        out_channels=256,
        in_index=[0, 1, 2, 3],
        num_things_classes=8,
        num_stuff_classes=11,
        num_queries=19,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='tqdmMSDeformAttnPixelDecoder',
            num_text_embeds=19,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_heads=8,
                            num_levels=3,
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=False,
                            norm_cfg=None,
                            init_cfg=None),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=False)
                    ],
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True),
                    feedforward_channels=2048,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(type='IdentityAssigner', num_cls=19),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(
            panoptic_on=True,
            semantic_on=False,
            instance_on=True,
            max_per_image=100,
            iou_thr=0.8,
            filter_low_score=True),
        text_proj=dict(text_in_dim=1024, text_out_dim=256)),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=1e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            text_encoder=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0),
            sparsegen=dict(lr_mult=10.0))))
work_dir = './work_dirs_d/siglip_drop_sparsegen_vit-l_1e-4_28k-g2c-512/siglip_drop_sparsegen_vit-l_1e-4_28k-g2c-512'
gpu_ids = range(0, 1)
