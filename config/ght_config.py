# model settings

model_cfg = dict(
    backbone=dict(type='HGT',
                  in_channel=30,
                  embed_dims=[30, 30, 30, 30],
                  num_heads=[1, 1, 1, 1],
                  mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True,
                  depths=[4, 4, 4, 4],
                  sr_ratios=[8, 8, 8, 8],
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  drop_path_rate=0.,
                  num_stages=4,
                  gradient_direction_number=30,
                  gnn_dim=16,
                  gnn_out=4,
                  K=6
                  )
)

# train
data_cfg = dict(
    batch_size=1,
    num_workers=1,
    is_voxel=False,
    need_edge=True,
    edge=dict(
        bval_name='bvals',
        bvec_name='bvecs',
        image_shape=[120, 160],
        angle=60
    ),
    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        epoches=100,
    ),
    test=dict(
        batch_size=1
    )
)

# optimizer
optimizer_cfg = dict(
    type='Adam',
    lr=0.0005)

# learning
lr_config = dict(is_scheduler=False)

