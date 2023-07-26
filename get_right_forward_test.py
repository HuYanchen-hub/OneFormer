# 对齐 forward 
import sys 
sys.path.append("/home/bingxing2/gpuuser206/OneFormer") 
sys.path.append("/home/bingxing2/gpuuser206/OneFormer/detectron2/projects/DeepLab") 
import torch  
import pickle as pkl  
from mmdet.registry import MODELS  
from mmengine.config import Config
from deeplab import add_deeplab_config
from detectron2.checkpoint import DetectionCheckpointer
from mmdet.utils import register_all_modules
register_all_modules()
# from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

# from detectron2.projects.DeepLab.deeplab import add_deeplab_config

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from detectron2.config import get_cfg 
from detectron2.modeling import build_model  
 
 
img = torch.rand((1, 3, 256, 256)).cuda() 
img_metas = [{ 
    "img_shape": (250, 250, 3), 
    "pad_shape": (256, 256, 3), 
    "metainfo": (256, 256,3)
}] 
 
#################################! MMDetection model 
# 配置文件中只需要完成 model 部分的配置即可 
# mmdet_cfg_path = "/home/bingxing2/gpuuser206/mmdetection/projects/OneFormer/configs/oneformer_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco_instance.py" 
mmdet_cfg_path = "/home/bingxing2/gpuuser206/mmdetection/projects/OneFormer/configs/oneformer_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco_panoptic.py" 
mmdet_cfg = Config.fromfile(mmdet_cfg_path) 
 
# 构建模型并加载权重 
checkpoint_path = "/home/bingxing2/gpuuser206/mmdetection/150_16_swin_l_oneformer_coco_100ep.pth" 

checkpoint = torch.load(checkpoint_path)["state_dict"] 
mmdet_detector = MODELS.build(mmdet_cfg.model) 
fw = open("./cur_model.txt", 'w') 
print(mmdet_detector, file=fw)
mmdet_detector.load_state_dict(checkpoint) 
mmdet_detector = mmdet_detector.cuda() 
mmdet_detector.eval() 
# data = open('one_former__model.txt', 'w')
# print(mmdet_detector, file=data)
 
with torch.no_grad(): 
    mmdet_features = mmdet_detector.extract_feat(img)
    mmdet_out = mmdet_detector.panoptic_head.forward(mmdet_features, img_metas)
    
 
#################################! Detectron2 model 
# 加载配置文件 
det2_cfg_path = "configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml" 
det2_cfg = get_cfg() 
add_deeplab_config(det2_cfg)
add_common_config(det2_cfg)
add_swin_config(det2_cfg)
add_dinat_config(det2_cfg)
add_convnext_config(det2_cfg)
add_oneformer_config(det2_cfg)
det2_cfg.merge_from_file(det2_cfg_path) 
det2_cfg.freeze() 
 
# 构建模型并加载权重 
det2_checkpoint_path = "checkpoints/150_16_swin_l_oneformer_coco_100ep.pth" 
# with open(det2_checkpoint_path, "rb") as f: 
#     det2_weight = pkl.load(f)["model"] 
# MaskFormer 官方 repo 中提供的权重是 np.array，需要转为为 tensor 
# det2_weight = { 
#     k: torch.from_numpy(v).cuda() 
#     for k, v in det2_weight.items() 
# } 
det2_detector = build_model(det2_cfg) 
checkpointer = DetectionCheckpointer(det2_detector)
checkpointer.load(det2_checkpoint_path)
# det2_detector.load_state_dict(det2_weight) 
det2_detector = det2_detector.cuda() 
det2_detector.eval() 
# data = open('det2_model.txt', 'w')
# print(det2_detector, file=data)

# print(det2_detector)
 
with torch.no_grad(): 
    tasks = det2_detector.task_tokenizer("The task is panoptic").cuda().unsqueeze(0) #(1, 77)
    tasks = det2_detector.task_mlp(tasks.float()) #(1,256)

    # img = (img - det2_detector.pixel_mean) / det2_detector.pixel_std 
    det2_features = det2_detector.backbone(img) 
    det2_out = det2_detector.sem_seg_head(det2_features, tasks)


print("features0")
print("mmdet: ", mmdet_features[0].sum()) 
print("det2: ", list(det2_features.values())[0].sum())
print("features1")
print("mmdet: ", mmdet_features[1].sum()) 
print("det2: ", list(det2_features.values())[1].sum())
print("features2")
print("mmdet: ", mmdet_features[2].sum()) 
print("det2: ", list(det2_features.values())[2].sum())
print("features3")
print("mmdet: ", mmdet_features[3].sum()) 
print("det2: ", list(det2_features.values())[3].sum())
print("pred_logits") 
print("mmdet: ", mmdet_out[0][-1].sum()) 
print("det2: ", det2_out["pred_logits"].sum()) 
print("pred_masks") 
print("mmdet: ", mmdet_out[1][-1].sum()) 
print("det2: ", det2_out["pred_masks"].sum()) 