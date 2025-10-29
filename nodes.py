import torch
from torch import Tensor
from torchvision.transforms import functional

from PIL import Image
import numpy as np
import comfy.utils
import time
from io import BytesIO


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
    CATEGORY = 'sunxAI_facetools'

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


class DetectFaceByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'tooltip': '人脸检测置信度阈值，越高越严格'}),
                'min_size': ('INT', {'default': 64, 'max': 512, 'step': 8, 'tooltip': '最小人脸尺寸，过滤掉太小的检测结果'}),
                'max_size': ('INT', {'default': 512, 'min': 512, 'step': 8, 'tooltip': '最大人脸尺寸，过滤掉太大的检测结果'}),
                'face_index': ('INT', {'default': 0, 'min': 0, 'max': 10, 'step': 1, 'tooltip': '人脸索引：0=最左边第一个，1=第二个，以此类推'}),
                'gender_filter': ('INT', {'default': 0, 'min': 0, 'max': 2, 'step': 1, 'tooltip': '性别筛选：0=任意性别，1=只检测男性(man)，2=只检测女性(woman)'}),
                'priority_mode': ('INT', {'default': 0, 'min': 0, 'max': 1, 'step': 1, 'tooltip': '优先级模式：0=下标优先（先选位置再检查性别），1=性别优先（先筛选性别再选位置）'}),
            },
            'optional': {
                'mask': ('MASK',),
            }
        }

    RETURN_TYPES = ('FACE', 'BOOLEAN')
    RETURN_NAMES = ('faces', 'has_face')
    FUNCTION = 'run'
    CATEGORY = 'sunxAI_facetools'

    def run(self, image, threshold, min_size, max_size, face_index, gender_filter, priority_mode, mask=None):
        faces = []
        masked = image
        if mask is not None:
            masked = image * tv.transforms.functional.resize(1-mask, image.shape[1:3])[..., None]
        masked = (masked * 255).type(torch.uint8)

        # 先进行 YOLO 人脸检测
        for i, img in enumerate(masked):
            unfiltered_faces = detect_faces(img, threshold, detect_gender=False)  # 不检测性别
            for face in unfiltered_faces:
                a, b, c, d = face.bbox
                h = abs(d-b)
                w = abs(c-a)
                if (h <= max_size or w <= max_size) and (min_size <= h or min_size <= w):
                    face.image_idx = i
                    face.img = image[i]
                    faces.append(face)

        print(f"[DetectFaceByIndex] YOLO detected {len(faces)} faces")

        # 直接调用 Models.gender 进行性别检测
        insightface_genders = Models.gender(image[0])  # 在原始图像上检测性别
        print(f"[DetectFaceByIndex] InsightFace detected {len(insightface_genders)} faces with genders")

        # 判断人头数量和性别数量是否一致
        if len(faces) != len(insightface_genders):
            print(f"[DetectFaceByIndex] Warning: Face count mismatch! YOLO={len(faces)}, InsightFace={len(insightface_genders)}")
            # 如果数量不一致，使用启发式方法
            faces.sort(key=lambda f: f.bbox[0])
            for i, face in enumerate(faces):
                face.gender = "man" if i == 0 else "woman"  # 左边男性，右边女性
                print(f"[DetectFaceByIndex] Face {i}: Using heuristic gender={face.gender}")
        else:
            # 数量一致，按顺序分配性别
            faces.sort(key=lambda f: f.bbox[0])  # 按 x 坐标排序
            for i, face in enumerate(faces):
                face.gender = insightface_genders[i]['gender']
                print(f"[DetectFaceByIndex] Face {i}: YOLO bbox={face.bbox}, assigned gender={face.gender}")

        # 先按 x 坐标排序（从左到右）
        faces.sort(key=lambda f: f.bbox[0])

        # 打印所有人脸的性别信息
        print(f"[DetectFaceByIndex] Priority mode: {priority_mode} ({'下标优先' if priority_mode == 0 else '性别优先'})")
        print(f"[DetectFaceByIndex] Gender filter: {gender_filter}")
        print(f"[DetectFaceByIndex] Face index: {face_index}")
        print(f"[DetectFaceByIndex] Total faces before filtering: {len(faces)}")
        for i, face in enumerate(faces):
            print(f"[DetectFaceByIndex] Face {i}: bbox={face.bbox}, gender={face.gender}")

        if priority_mode == 0:  # 下标优先：先选位置再检查性别
            print(f"[DetectFaceByIndex] Using index priority mode")
            if faces and face_index < len(faces):
                selected_face = faces[face_index]
                print(f"[DetectFaceByIndex] Selected face {face_index}: bbox={selected_face.bbox}, gender={selected_face.gender}")

                # 检查性别是否符合要求
                if gender_filter == 1:  # 必须是男性
                    if selected_face.gender == "man":
                        faces = [selected_face]
                        print(f"[DetectFaceByIndex] Face {face_index} is male, keeping it")
                    else:
                        faces = []
                        print(f"[DetectFaceByIndex] Face {face_index} is not male, returning empty")
                elif gender_filter == 2:  # 必须是女性
                    if selected_face.gender == "woman":
                        faces = [selected_face]
                        print(f"[DetectFaceByIndex] Face {face_index} is female, keeping it")
                    else:
                        faces = []
                        print(f"[DetectFaceByIndex] Face {face_index} is not female, returning empty")
                else:  # gender_filter == 0，不检查性别
                    faces = [selected_face]
                    print(f"[DetectFaceByIndex] No gender filter, keeping face {face_index}")
            else:
                faces = []  # 下标超出范围，返回空列表
                print(f"[DetectFaceByIndex] Face index {face_index} out of range, returning empty")

        else:  # 性别优先：先筛选性别再选位置
            print(f"[DetectFaceByIndex] Using gender priority mode")

            # 先根据性别筛选
            if gender_filter == 1:  # 只选择男性
                filtered_faces = [face for face in faces if face.gender == "man"]
                print(f"[DetectFaceByIndex] After filtering for men: {len(filtered_faces)} faces")
            elif gender_filter == 2:  # 只选择女性
                filtered_faces = [face for face in faces if face.gender == "woman"]
                print(f"[DetectFaceByIndex] After filtering for women: {len(filtered_faces)} faces")
            else:  # gender_filter == 0，不筛选性别
                filtered_faces = faces
                print(f"[DetectFaceByIndex] No gender filtering, keeping all faces")

            # 再根据 face_index 选择
            if filtered_faces and face_index < len(filtered_faces):
                selected_face = filtered_faces[face_index]
                faces = [selected_face]
                print(f"[DetectFaceByIndex] Selected face {face_index} from filtered faces: bbox={selected_face.bbox}, gender={selected_face.gender}")
            else:
                faces = []
                print(f"[DetectFaceByIndex] Face index {face_index} out of range in filtered faces, returning empty")

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
    CATEGORY = 'sunxAI_facetools'

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
    CATEGORY = 'sunxAI_facetools'

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

class VAEDecodeNew:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
            },
            'optional': {
                'has_face': ('BOOLEAN',),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"

    CATEGORY = "sunxAI_facetools"
    DESCRIPTION = "Decodes latent images back into pixel space images."

    def decode(self, vae, samples, has_face=True):
        # 如果has_face为False，返回空白画布，节省VAE解码时间
        if not has_face:
            # 创建512x512的黑色空白画布
            blank_canvas = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank_canvas,)
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )


class VAEEncodeNew:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "pixels": ("IMAGE", ), "vae": ("VAE", )},
            'optional': {
                'has_face': ('BOOLEAN',),
            }
            }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "sunxAI_facetools"

    def encode(self, vae, pixels, has_face=True):
        # 如果has_face为False，返回空白latent，节省VAE编码时间
        if not has_face:
            # 创建512x512对应的空白latent (512//8 = 64)
            blank_latent = torch.zeros([1, 4, 64, 64])
            return ({"samples": blank_latent},)

        t = vae.encode(pixels[:,:,:,:3])
        return ({"samples":t}, )


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
    CATEGORY = "sunxAI_facetools"

    def run(self, cond, true_value, false_value):
        return (true_value if cond else false_value,)



class ColorAdjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": -255,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "hue": ("FLOAT", {
                    "default": 0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
            },
            'optional': {
                'has_face': ('BOOLEAN',),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "sunxAI_facetools"

    def main(self,
             image: Tensor,
             contrast: float = 1,
             brightness: float = 1,
             saturation: float = 1,
             hue: float = 0,
             gamma: float = 1,
             has_face: bool = True):

        if not has_face:
            return (image,)

        permutedImage = image.permute(0, 3, 1, 2)

        if (contrast != 1):
            permutedImage = functional.adjust_contrast(permutedImage, contrast)

        if (brightness != 1):
            permutedImage = functional.adjust_brightness(permutedImage, brightness)

        if (saturation != 1):
            permutedImage = functional.adjust_saturation(permutedImage, saturation)

        if (hue != 0):
            permutedImage = functional.adjust_hue(permutedImage, hue)

        if (gamma != 1):
            permutedImage = functional.adjust_gamma(permutedImage, gamma)

        result = permutedImage.permute(0, 2, 3, 1)

        return (result,)


class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 95,
                        "min": 60,
                        "max": 100,
                        "step": 1,
                        "tooltip": "JPEG压缩质量（60=低质量，100=高质量）"
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "sunxAI_facetools"

    def save_images(self, images, jpeg_quality):
        pbar = comfy.utils.ProgressBar(images.shape[0])

        for idx, image in enumerate(images):
            try:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # 直接在内存中进行JPEG压缩
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=jpeg_quality)
                buffer.seek(0)
                jpg_img = Image.open(buffer).convert("RGB").copy()

                # 发送JPEG格式图像
                pbar.update_absolute(idx, images.shape[0], ("JPEG", jpg_img, None))

            except Exception as e:
                print(f"[SaveImageWebsocket] ❌ Skipped idx={idx} due to error: {e}")
                continue

        return {}

    @classmethod
    def IS_CHANGED(s, images, jpeg_quality):
        return time.time()





