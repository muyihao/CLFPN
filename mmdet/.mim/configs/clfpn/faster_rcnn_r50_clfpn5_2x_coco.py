_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='CEMFPN2',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            dilations=[3, 6, 9, 12, 15]),
        dict(
            type='AFPN4_LSC5',
            in_channels=256,
            num_levels=5,
            refine_level=2
            )
    ])

