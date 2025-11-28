import os
import json
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine.hooks import HookBase
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from torch.utils.tensorboard import SummaryWriter

# ================================
# CONFIG
# ================================
DATA_ROOT = "/home/ucloud/deepfashion2_original_images"
OUTPUT_DIR = "./df2_output"
TENSORBOARD_DIR = "./runs/df2_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANN_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANN_DIR   = os.path.join(DATA_ROOT, "validation", "annos")

# ================================
# UTILITY: Map category_id to consecutive integers
# ================================
def build_category_mapping(ann_dir):
    cat_ids = set()
    for f in os.listdir(ann_dir):
        if not f.endswith(".json"):
            continue
        ann = json.load(open(os.path.join(ann_dir, f)))
        for key in ["item1", "item2"]:
            if key in ann:
                cat_ids.add(ann[key]["category_id"])
    cat_ids = sorted(list(cat_ids))
    cat_id_map = {cid: i for i, cid in enumerate(cat_ids)}
    return cat_id_map, len(cat_ids)

CATEGORY_MAP, NUM_CLASSES = build_category_mapping(TRAIN_ANN_DIR)

# ================================
# LOAD DATASET
# ================================
def parse_deepfashion2(img_dir, ann_dir):
    dataset_dicts = []
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg",".png"))])
    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, img_file.rsplit(".",1)[0] + ".json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path, "r") as f:
            ann = json.load(f)

        height, width = utils.read_image(img_path).shape[:2]

        record = {
            "file_name": img_path,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": []
        }

        for key in ["item1","item2"]:
            if key not in ann:
                continue
            item = ann[key]
            if "bounding_box" not in item or "category_id" not in item:
                continue

            # convert category to 0..N-1
            cat_id = CATEGORY_MAP[item["category_id"]]

            obj = {
                "bbox": item["bounding_box"],
                "bbox_mode": utils.BoxMode.XYXY_ABS,
                "category_id": cat_id,
                "segmentation": item.get("segmentation", []),
                "keypoints": item.get("landmarks", [])
            }
            record["annotations"].append(obj)
        dataset_dicts.append(record)
    return dataset_dicts

# REGISTER DATASETS
DatasetCatalog.register("df2_train", lambda: parse_deepfashion2(TRAIN_IMG_DIR, TRAIN_ANN_DIR))
DatasetCatalog.register("df2_val", lambda: parse_deepfashion2(VAL_IMG_DIR, VAL_ANN_DIR))
MetadataCatalog.get("df2_train").set(thing_classes=[f"class_{i}" for i in range(NUM_CLASSES)])

# ================================
# CUSTOM MAPPER (RESIZE ONLY)
# ================================
def custom_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
    aug_input = T.AugInput(image)
    transforms = aug(aug_input)
    image = aug_input.image
    dataset_dict["image"] = torch.as_tensor(image.transpose(2,0,1).copy())
    annos = [utils.transform_instance_annotations(obj, transforms, image.shape[:2])
              for obj in dataset_dict.pop("annotations")]
    dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
    return dataset_dict

# ================================
# HOOKS
# ================================
class TensorboardHook(HookBase):
    def __init__(self, writer):
        self.writer = writer
    def after_step(self):
        if self.trainer.iter % 20 == 0:
            for k, v in self.trainer.storage.latest().items():
                self.writer.add_scalar(k, v, self.trainer.iter)

class ValidationHook(HookBase):
    def __init__(self, cfg, writer, val_period=5000):
        self.cfg = cfg.clone()
        self.writer = writer
        self.val_period = val_period
    def after_step(self):
        if self.trainer.iter % self.val_period == 0 and self.trainer.iter > 0:
            evaluator = COCOEvaluator("df2_val", self.cfg, False, output_dir=OUTPUT_DIR)
            val_loader = build_detection_test_loader(self.cfg, "df2_val")
            metrics = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            if "bbox" in metrics:
                self.writer.add_scalar("val/mAP_bbox", metrics["bbox"]["AP"], self.trainer.iter)
            if "segm" in metrics:
                self.writer.add_scalar("val/mAP_mask", metrics["segm"]["AP"], self.trainer.iter)
            if "keypoints" in metrics:
                self.writer.add_scalar("val/mAP_keypoints", metrics["keypoints"]["AP"], self.trainer.iter)
            print(f"\n📝 Validation metrics at iter {self.trainer.iter}: {metrics}\n")

class IterativeCheckpointHook(HookBase):
    def __init__(self, save_dir, save_every=2000):
        self.save_dir = save_dir
        self.save_every = save_every
    def after_step(self):
        if self.trainer.iter % self.save_every == 0 and self.trainer.iter > 0:
            path = os.path.join(self.save_dir, f"model_iter_{self.trainer.iter}.pth")
            self.trainer.checkpointer.save(path)
            print(f"\n💾 Saved checkpoint: {path}\n")

# ================================
# TRAINER
# ================================
class DF2Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

# ================================
# CONFIGURATION
# ================================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("df2_train",)
cfg.DATASETS.TEST  = ("df2_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.MAX_ITER = 300000
cfg.SOLVER.STEPS = []  # no LR decay
cfg.SOLVER.CHECKPOINT_PERIOD = 50000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.KEYPOINT_ON = True  # landmarks

cfg.OUTPUT_DIR = OUTPUT_DIR

# ================================
# TRAINING
# ================================
writer = SummaryWriter(TENSORBOARD_DIR)
trainer = DF2Trainer(cfg)
trainer.register_hooks([TensorboardHook(writer)])
trainer.register_hooks([ValidationHook(cfg, writer, val_period=5000)])
trainer.register_hooks([IterativeCheckpointHook(OUTPUT_DIR, save_every=2000)])

trainer.resume_or_load(resume=False)
trainer.train()
writer.close()
print("🎉 Training complete!")
