_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='AFPN4_C',
            in_channels=256,
            num_levels=5,
            refine_level=2
            )
    ])
