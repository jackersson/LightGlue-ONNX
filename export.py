from typing import List
from collections.abc import Iterable
import torch
import fire
import typing as typ
from lightglue_onnx import LightGlue, SuperPoint
from lightglue_onnx.end2end import normalize_keypoints
from lightglue_onnx.utils import load_image, rgb_to_grayscale


def export_onnx(
    img_size: typ.Union[typ.Tuple[int, int], int]=512,
    extractor_type="superpoint",
    extractor_path=None,
    lightglue_path=None,
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
    dynamic=False,
    max_num_keypoints=None,
    detection_threshold=0.0005,
    rknn=False,
    num_keypoints_hq: int = 2048,
    num_keypoints_lq: int = 512,
):
    # Handle args
    if isinstance(img_size, Iterable) and len(img_size) == 1:
        img_size = img_size[0]
    
    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)
    
    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)
        
    # Models
    extractor_type = extractor_type.lower()
  
    if isinstance(extractor_path, str):
        
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints, detection_threshold=detection_threshold, rknn=rknn).eval()
        
        # Export Extractor
        dynamic_axes = {
            "keypoints": {1: "num_keypoints"},
            "scores": {1: "num_keypoints"},
            "descriptors": {1: "num_keypoints"},
        }
        
        if dynamic:
            dynamic_axes.update({"image": {2: "height", 3: "width"}})
        else:
            print(
                f"WARNING: Exporting without --dynamic implies that the {extractor_type} extractor's input image size will be locked to {image0.shape[-2:]}"
            )
            extractor_path = extractor_path.replace(
                ".onnx", f"_{image0.shape[-2]}x{image0.shape[-1]}.onnx"
            )
        
        output_names = ["keypoints", "scores", "descriptors"]
        if rknn:
            output_names.remove("keypoints")
            
        torch.onnx.export(
            extractor,
            image0[None],
            extractor_path,
            input_names=["image"],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            # dynamic_axes=dynamic_axes,
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(lightglue_path, str):

        lightglue = LightGlue(extractor_type, rknn=rknn).eval().to(device)
        # Export LightGlue
        # feats0, feats1 = extractor(image0[None]), extractor(image1[None])
        # kpts0, scores0, desc0 = feats0
        # kpts1, scores1, desc1 = feats1

        sp_descriptor_size = 256
        num_points = 2 # x, y
        
        kpts1 = torch.zeros(1, num_keypoints_hq, num_points).to(device)
        scores1 = torch.zeros(1, num_keypoints_hq, 1).to(device)
        desc1 = torch.zeros(1, num_keypoints_hq, sp_descriptor_size).to(device)

        kpts0 = torch.zeros(1, num_keypoints_lq, num_points).to(device)
        scores0 = torch.zeros(1, num_keypoints_lq, 1).to(device)
        desc0 = torch.zeros(1, num_keypoints_lq, sp_descriptor_size).to(device)

        # print(kpts0.shape, scores0.shape, desc0.shape)

        kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
        kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])

        output_names = ["matches0", "mscores0"]
        if rknn:
            output_names.remove("matches0")
        
        torch.onnx.export(
            lightglue,
            (kpts0, kpts1, desc0, desc1),
            lightglue_path,
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=output_names,
            do_constant_folding=True,
            opset_version=17,
            # dynamic_axes={
            #     "kpts0": {1: "num_keypoints0"},
            #     "kpts1": {1: "num_keypoints1"},
            #     "desc0": {1: "num_keypoints0"},
            #     "desc1": {1: "num_keypoints1"},
            #     "matches0": {0: "num_matches0"},
            #     "mscores0": {0: "num_matches0"},
            # },
        )


if __name__ == "__main__":
    fire.Fire(export_onnx)
