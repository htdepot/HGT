# model settings

model_cfg = dict(
    backbone=dict(type='NestedUNet',
                  in_channel=30,
                  out_channel=3)
)

# train
data_cfg = dict(
    batch_size=16,
    num_workers=1,
    is_voxel=False,
    need_edge=False,
    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        epoches=200,
    ),
    test=dict(
        batch_size=16
    )
)

# optimizer
optimizer_cfg = dict(
    type='Adam',
    lr=0.001)

# learning
lr_config = dict(is_scheduler=False)

