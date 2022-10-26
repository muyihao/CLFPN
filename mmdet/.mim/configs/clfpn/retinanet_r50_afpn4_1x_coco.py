_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='AFPN4_LSC5',
            in_channels=256,
            num_levels=5,
            refine_level=2
            )
    ])