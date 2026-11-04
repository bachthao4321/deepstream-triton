"""TSN (ResNet18) trên UCF101 — config XÁC NHẬN TOOLCHAIN (CLAUDE.md mục 7:
"chạy thử 1 vòng train ngắn bằng MMAction2 (vài epoch) để xác nhận toolchain
hoạt động trước khi train full").

Kiến trúc nhẹ theo khuyến nghị CLAUDE.md giai đoạn 3 (TSN). ResNet18 (thay vì
ResNet50 mặc định trong config chính thức MMAction2) để train nhanh hơn khi
chỉ cần xác nhận pipeline chạy được, không phải để đạt accuracy tốt nhất.

Dùng UCF101 gốc (101 lớp hành động chung — ApplyEyeMakeup, PlayGuitar...),
KHÔNG PHẢI lớp hành vi giám sát thật (fall, loitering...). Khi đã xác nhận
toolchain chạy ổn, đổi dataset + num_classes sang tập hành vi thật (tự gán
nhãn video giám sát) rồi train lại config tương tự.

max_epochs cố tình để THẤP (3) — chỉ cần thấy loss giảm + checkpoint xuất ra
là coi như toolchain OK; tăng lên (vd 50-100) khi train full cho model dùng
thật."""

_base_ = []

# ── Đường dẫn dataset (sinh bởi scripts/download_ucf101.sh + prepare_dataset.py) ──
DATA_ROOT = "/ssd1/thaokb/deepstream/workspace/datasets/ucf101"
VIDEOS_DIR = f"{DATA_ROOT}/videos"
ANN_TRAIN = f"{DATA_ROOT}/train_list.txt"
ANN_VAL = f"{DATA_ROOT}/val_list.txt"

NUM_CLASSES = 101

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type="Recognizer2D",
    backbone=dict(
        type="ResNet",
        pretrained="torchvision://resnet18",
        depth=18,
        norm_eval=False,
    ),
    cls_head=dict(
        type="TSNHead",
        num_classes=NUM_CLASSES,
        in_channels=512,              # ResNet18 feature dim (ResNet50 = 2048)
        spatial_type="avg",
        consensus=dict(type="AvgConsensus", dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips="prob",
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape="NCHW",
    ),
    train_cfg=None,
    test_cfg=None,
)

# ── Pipeline ─────────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

val_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

# ── Dataloader ───────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="VideoDataset",
        ann_file=ANN_TRAIN,
        data_prefix=dict(video=VIDEOS_DIR),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="VideoDataset",
        ann_file=ANN_VAL,
        data_prefix=dict(video=VIDEOS_DIR),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="AccMetric")
test_evaluator = val_evaluator

# ── Vòng train ngắn để xác nhận toolchain (KHÔNG phải train full) ──────────────
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=3, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="MultiStepLR", begin=0, end=3, by_epoch=True, milestones=[2], gamma=0.1),
]

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2),
)

# ── Runtime (mmengine) ─────────────────────────────────────────────────────────
default_scope = "mmaction"
default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20, ignore_last=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="auto"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)
log_level = "INFO"
load_from = None
resume = False

work_dir = "/ssd1/thaokb/deepstream/workspace/deepstream-rtsp-tracker/training/mmaction2/work_dirs/tsn_r18_ucf101_toolchain"
