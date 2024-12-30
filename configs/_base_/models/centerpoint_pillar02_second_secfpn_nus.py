voxel_size = [0.2, 0.2, 8]
model = dict(
    type='CenterPoint', 
    data_preprocessor=dict( # Voxel preprocessing
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20, # maximum points per voxel
            voxel_size=voxel_size,
            max_voxels=(30000, 40000))), # (train, inference)
    pts_voxel_encoder=dict( # extract featrue from voxel/pillar
        type='PillarFeatureNet', # simplified version of PointNet 
        in_channels=5,
        feat_channels=[64], # output feature channel
        with_distance=False, # not using distance information
        voxel_size=(0.2, 0.2, 8), 
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False), # not using legacy code
    pts_middle_encoder=dict( # scatter pillar to 2d scene (512, 512, 64) feature
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)), # number of grids = (51.2+51.2)/0.2 = 512 # point_cloud_range/voxel_size
    pts_backbone=dict( # extract feature for detection
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256], # multi-scale
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2], # 512 -> 256 -> 128 -> 64 # final output 64 x 64 x 256
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict( # output feature map of single scale
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128], # make channel size consistent 
        upsample_strides=[0.5, 1, 2], # 64 / 0,5 = 128, 256 / 2 = 128
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead', # detection head
        in_channels=sum([128, 128, 128]),
        tasks=[ # define categories - 6 classes
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict( # parallel heads e.g. (1, 2) = (height scalar, # of layers)
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict( # decoding regression heads
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'), # loss classification
        loss_bbox=dict( # loss regression
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=[-51.2, -51.2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))
