#python 运行程序
DETECTRON2_DATASETS=/home/bingxing2/gpuuser206/mmdetection/data python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 8 \
    --config-file configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml \
    OUTPUT_DIR outputs/coco_swin_large WANDB.NAME coco_swin_large
# DETECTRON2_DATASETS=/home/bingxing2/gpuuser206/mmdetection/data python train_net.py --num-gpus 1 \
#     --config-file configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml \
#     OUTPUT_DIR outputs/coco_swin_large WANDB.NAME coco_swin_large