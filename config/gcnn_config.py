# model settings

model_cfg = dict(
    backbone=dict(type='GCNN',
                  in_channel=30,
                  hidden=150,
                  out_channel=3,
                  K=6,
                  features=4)
)

# train
data_cfg = dict(
    batch_size=128,
    num_workers=1,
    is_voxel=True,
    need_edge=False,
    edge=dict(
        bval_name='bvals',
        bvec_name='bvecs',
        image_shape=[0, 0],
        angle=60
    ),
    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        epoches=100,
    ),
    test=dict(
        batch_size=2000
    )
)

# optimizer
optimizer_cfg = dict(
    type='Adam',
    lr=0.0005)

# learning
lr_config = dict(is_scheduler=False)

