import torch
from .utils import *


class DetectFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'min_size': ('INT', {'default': 64, 'max': 512, 'step': 8}),
                'max_size': ('INT', {'default': 512, 'min': 512, 'step': 8}),
            },
            'optional': {
                'mask': ('MASK',),
            }
        }

    RETURN_TYPES = ('FACE', 'BOOLEAN')
    RETURN_NAMES = ('faces', 'has_face')
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    def run(self, image, threshold, min_size, max_size, mask=None):
        faces = []
        masked = image
        if mask is not None:
            masked = image * tv.transforms.functional.resize(1-mask, image.shape[1:3])[..., None]
        masked = (masked * 255).type(torch.uint8)
        for i, img in enumerate(masked):
            unfiltered_faces = detect_faces(img, threshold)
            for face in unfiltered_faces:
                a, b, c, d = face.bbox
                h = abs(d-b)
                w = abs(c-a)
                if (h <= max_size or w <= max_size) and (min_size <= h or min_size <= w):
                    face.image_idx = i
                    face.img = image[i]
                    faces.append(face)

        # 只返回面积最大的人脸，如果没有人脸则返回空列表

        if faces:
            largest_face = max(faces, key=lambda f: abs(f.bbox[2] - f.bbox[0]) * abs(f.bbox[3] - f.bbox[1]))
            faces = [largest_face]
        else:
            faces = []

        has_face = len(faces) > 0
        return (faces, has_face)

class CropFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'faces': ('FACE',),
                'crop_size': ('INT', {'default': 512, 'min': 512, 'max': 1024, 'step': 128}),
                'crop_factor': ('FLOAT', {'default': 1.5, 'min': 1.0, 'max': 3, 'step': 0.1}),
                'mask_type': (mask_types,)
            }
        }

    RETURN_TYPES = ('IMAGE', 'MASK', 'WARP')
    RETURN_NAMES = ('crops', 'masks', 'warps')
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    def run(self, faces, crop_size, crop_factor, mask_type):
        if len(faces) == 0:
            empty_crop = torch.zeros((1,512,512,3))
            empty_mask = torch.zeros((1,512,512))
            empty_warp = np.array([
                [1,0,-512],
                [0,1,-512],
            ], dtype=np.float32)
            return (empty_crop, empty_mask, [empty_warp])

        crops = []
        masks = []
        warps = []
        for face in faces:
            M, crop = face.crop(crop_size, crop_factor)
            mask = mask_crop(face, M, crop, mask_type)
            crops.append(np.array(crop[0]))
            masks.append(np.array(mask[0]))
            warps.append(M)
        crops = torch.from_numpy(np.array(crops)).type(torch.float32)
        masks = torch.from_numpy(np.array(masks)).type(torch.float32)
        return (crops, masks, warps)

class WarpFaceBack:
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    CATEGORY = 'facetools'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE',),
                'face': ('FACE',),
                'crop': ('IMAGE',),
                'mask': ('MASK',),
                'warp': ('WARP',),
            },
            'optional': {
                'has_face': ('BOOLEAN',),
            }
        }

    def run(self, images, face, crop, mask, warp, has_face=True):
        # 如果has_face为False，直接返回原图像
        if not has_face:
            return (images,)

        # 处理单个人脸
        if len(face) == 0:
            return (images,)

        single_face = face[0]
        single_crop = crop[0]
        single_mask = mask[0]
        single_warp = warp[0]

        results = []
        for i, image in enumerate(images):
            if i != single_face.image_idx:
                result = image
            else:
                warped_mask = np.clip(cv2.warpAffine(single_mask.numpy(),
                                cv2.invertAffineTransform(single_warp),
                                image.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4), 0, 1)

                swapped = np.clip(cv2.warpAffine(single_crop.cpu().numpy(),
                                cv2.invertAffineTransform(single_warp),
                                image.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4), 0, 1)

                result = (swapped * warped_mask[..., None] +
                         (1 - warped_mask[..., None]) * image.numpy())
                result = torch.from_numpy(result)
            results.append(result)

        results = torch.stack(results)
        return (results, )


class SelectFloatByBool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("BOOLEAN",),
                "true_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "false_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "Flow/Values"

    def run(self, cond, true_value, false_value):
        return (true_value if cond else false_value,)


