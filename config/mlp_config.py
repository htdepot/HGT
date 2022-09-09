# model settings

model_cfg = dict(
    backbone=dict(type='MLP',
                  in_channel=30,
                  hidden=150,
                  out_channel=3,
                  drop_out=0.1)
)

# train
data_cfg = dict(
    batch_size=128,
    num_workers=1,
    is_voxel=True,
    need_edge=False,
    train=dict(
        pretrained_flag=True,
        pretrained_weights='./parameter/mlp_best_parameter.pkl',
        epoches=190,
    ),
    test=dict(
        batch_size=2000
    )
)

# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0)

# learning
lr_config = dict(is_scheduler=True,
                 type='StepLR',
                 step=1,
                 gamma=0.9)

