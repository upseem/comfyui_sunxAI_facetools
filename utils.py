import os
import torch
import torchvision as tv
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import ConvexHull
from folder_paths import models_dir
from ultralytics import YOLO
from onnxruntime import InferenceSession, get_available_providers
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def pad_to_stride(image, stride=32):
    h, w, _ = image.shape
    pr = (stride - w % stride) % stride
    pb = (stride - h % stride) % stride
    padded_image = tv.transforms.transforms.F.pad(image.permute(2,0,1), (0, 0, pr, pb)).permute(1,2,0)
    return padded_image

def resize(img, size):
    h, w, _ = img.shape
    s = max(h, w)
    scale_factor = s / size
    ph, pw = (s - h) // 2, (s - w) // 2
    pad = tv.transforms.Pad((pw, ph))
    resize = tv.transforms.Resize(size=(size, size), antialias=True)
    img = resize(pad(img.permute(2,0,1))).permute(1,2,0)
    return img, scale_factor, ph, pw


class Models:
    @classmethod
    def yolo(cls, img, threshold):
        if '_yolo' not in cls.__dict__:
            cls._yolo = YOLO(os.path.join(models_dir,'ultralytics','bbox','face_yolov8m.pt'))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._yolo = cls._yolo.to(device)
            print(f"[YOLO] Model loaded to device: {device}")

        # 推理时打印图像和当前模型 device（用于验证）
        print(f"[YOLO] Predicting on device: {cls._yolo.device}")
        dets = cls._yolo(img, conf=threshold)[0]
        return dets

    @classmethod
    def gender(cls, crop):
        """使用 cvlib 进行性别检测"""
        try:
            import cvlib as cv
            import cv2

            # 将 torch tensor 转为 numpy (确保在 CPU)
            if isinstance(crop, torch.Tensor):
                crop_np = crop.detach().cpu().numpy()
                # [C,H,W] -> [H,W,C]
                if crop_np.ndim == 3 and crop_np.shape[0] in [1,3]:
                    crop_np = np.transpose(crop_np, (1, 2, 0))
            else:
                crop_np = crop

            # 转换为 uint8
            crop_np = np.clip(crop_np * 255, 0, 255).astype(np.uint8)

            # 检测人脸
            faces, confidences = cv.detect_face(crop_np)

            if len(faces) > 0:
                # 取第一个人脸区域
                x1, y1, x2, y2 = faces[0]
                face_crop = np.copy(crop_np[y1:y2, x1:x2])

                # 性别检测
                label, confidence = cv.detect_gender(face_crop)
                best_label = label[int(np.argmax(confidence))]  # 取置信度最高的结果
                return best_label
            else:
                return "unknown"

        except ImportError:
            print("[Gender] cvlib not installed, using fallback method")

            # fallback: 用宽高比简单猜
            crop_np = crop.detach().cpu().numpy() if isinstance(crop, torch.Tensor) else crop
            h, w = crop_np.shape[-2:]
            face_ratio = w / (h + 1e-6)  # 防止除零错误
            return "man" if face_ratio > 0.8 else "woman"

        except Exception as e:
            print(f"[Gender] Error in gender detection: {e}")
            return "unknown"
    @classmethod
    def lmk(cls, crop):
        if '_lmk' not in cls.__dict__:
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            cls._lmk = InferenceSession(os.path.join(models_dir, 'landmarks', 'fan2_68_landmark.onnx'), providers=providers)
            print("[landmark] Using ONNX providers:", cls._lmk.get_providers())
        lmk = cls._lmk.run(None, {'input': crop})[0]

        return lmk

def get_submatrix_with_padding(img, a, b, c, d):
    pl = -min(a, 0)
    pt = -min(b, 0)
    pr = -min(img.shape[1] - c, 0)
    pb = -min(img.shape[0] - d, 0)
    a, b, c, d = max(a, 0), max(b, 0), min(c, img.shape[1]), min(d, img.shape[0])

    submatrix = img[b:d, a:c].permute(2,0,1)
    pad = tv.transforms.Pad((pl, pt, pr, pb))
    submatrix = pad(submatrix).permute(1,2,0)

    return submatrix

class Face:
    def __init__(self, img, a, b, c, d, detect_gender=False) -> None:
        self.img = img
        lmk = None
        best_score = 0
        i = 0
        crop = get_submatrix_with_padding(self.img, a, b, c, d)
        for curr_i in range(4):
            rcrop, s, ph, pw = resize(crop.rot90(curr_i), 256)
            rcrop = (rcrop[None] / 255).permute(0,3,1,2).type(torch.float32).numpy()
            curr_lmk = Models.lmk(rcrop)
            score = np.mean(curr_lmk[0,:,2])
            if score > best_score:
                best_score = score
                lmk = curr_lmk
                i = curr_i

        self.bbox = (a,b,c,d)
        self.w = c - a
        self.h = d - b
        self.confidence = best_score

        self.kps = np.vstack([
            lmk[0,[37,38,40,41],:2].mean(axis=0),
            lmk[0,[43,44,46,47],:2].mean(axis=0),
            lmk[0,[30,48,54],:2]
        ]) * 4 * s

        self.T2 = np.array([[1, 0, -a], [0, 1, -b], [0, 0, 1]])
        rot = cv2.getRotationMatrix2D((128*s,128*s), 90*i, 1)
        self.R = np.vstack((rot, np.array((0,0,1))))

        # 只在需要时进行性别检测
        if detect_gender:
            self.gender = Models.gender(crop)
        else:
            self.gender = "unknown"

    def crop(self, size, crop_factor):
        S = np.array([[1/crop_factor, 0, 0], [0, 1/crop_factor, 0], [0, 0, 1]])
        M = estimate_norm(self.kps, size)
        N = M @ self.R @ self.T2
        cx, cy = np.array((size/2, size/2, 1)) @ cv2.invertAffineTransform(M @ self.R @ self.T2).T
        T3 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        T4 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        N = N @ T4 @ S @ T3
        crop = cv2.warpAffine(self.img.numpy(), N, (size, size), flags=cv2.INTER_LANCZOS4)
        crop = torch.from_numpy(crop)[None]
        crop = torch.clip(crop, 0, 1)

        return N, crop

def detect_faces(img, threshold, detect_gender=False):
    img = pad_to_stride(img, stride=32)
    original_torch_load = torch.load
    def torch_load_wrap(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = torch_load_wrap
    dets = Models.yolo((img[None] / 255).permute(0,3,1,2), threshold)
    torch.load = original_torch_load
    boxes = (dets.boxes.xyxy.reshape(-1,2,2)).reshape(-1,4)
    faces = []
    for (a,b,c,d), box in zip(boxes.type(torch.int).cpu().numpy(), dets.boxes):
        cx, cy = (a+c)/2, (b+d)/2
        r = np.sqrt((c-a)**2 + (d-b)**2) / 2

        a,b,c,d = [int(x) for x in (cx - r, cy - r, cx + r, cy + r)]
        face = Face(img, a, b, c, d, detect_gender=detect_gender)

        faces.append(face)
    return faces

def get_face_mesh(crop: torch.Tensor):
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=10) as face_mesh:
        mesh = face_mesh.process(crop.mul(255).type(torch.uint8)[0].numpy())
    _, h, w, _ = crop.shape
    if mesh.multi_face_landmarks is not None:
        all_pts = np.array([np.array([(w*l.x, h*l.y) for l in lmks.landmark]) for lmks in mesh.multi_face_landmarks], dtype=np.int32)
        idx = np.argmin(np.abs(all_pts - np.array([w/2,h/2])).sum(axis=(1,2)))
        points = all_pts[idx]
        return points
    else:
        return None

def mask_simple_square(face, M, crop):
    # rotated bbox and size
    h,w = crop.shape[1:3]
    a,b,c,d = face.bbox
    rect = np.array([
        [a,b,1],
        [a,d,1],
        [c,b,1],
        [c,d,1],
    ]) @ M.T
    lx, ly = [int(x) for x in np.min(rect, axis=0)]
    hx, hy = [int(x) for x in np.max(rect, axis=0)]
    mask = np.zeros((h,w), dtype=np.float32)
    mask = cv2.rectangle(mask, (lx,ly), (hx,hy), 1, -1)
    mask = torch.from_numpy(mask)[None]
    return mask

def mask_convex_hull(face, M, crop):
    h,w = crop.shape[1:3]
    points = get_face_mesh(crop)
    if points is None: return mask_simple_square(face, M, crop)
    hull = ConvexHull(points)
    mask = np.zeros((h,w), dtype=np.int32)
    cv2.fillPoly(mask, [points[hull.vertices,:]], color=1)
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask[None])
    return mask


def mask_jonathandinu(crop, skin=True, nose=True, eye_g=True, l_eye=True, r_eye=True, l_brow=True, r_brow=True,
                    l_ear=True, r_ear=True, mouth=True, u_lip=True, l_lip=True,
                    hair=False, hat=False, ear_r=False, neck_l=False, neck=False, cloth=False):
    global jonathandinu_image_processor, jonathandinu_model

    device = (
        "cuda"
        # Device for NVIDIA or AMD GPUs
        if torch.cuda.is_available()
        else "mps"
        # Device for Apple Silicon (Metal Performance Shaders)
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if 'jonathandinu_image_processor' not in globals():
        jonathandinu_image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        jonathandinu_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        jonathandinu_model.to(device)

    inputs = jonathandinu_image_processor(images=crop.mul(255).type(torch.uint8), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = jonathandinu_model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # resize output to match input image dimensions
    upsampled_logits = tv.transforms.functional.resize(logits, crop.shape[1:3], antialias=True)

    labels = upsampled_logits.argmax(dim=1)

    ids = {
        'skin': 1,
        'nose': 2,
        'eye_g': 3,
        'l_eye': 4,
        'r_eye': 5,
        'l_brow': 6,
        'r_brow': 7,
        'l_ear': 8,
        'r_ear': 9,
        'mouth': 10,
        'u_lip': 11,
        'l_lip': 12,
        'hair': 13,
        'hat': 14,
        'ear_r': 15,
        'neck_l': 16,
        'neck': 17,
        'cloth': 18,
    }
    keep = []
    for k, v in locals().items():
        if k in ids and v:
            keep.append(ids[k])
    face_part_ids = torch.tensor(keep).cuda()

    mask = torch.sum(labels.repeat(len(face_part_ids), 1,1,1) == face_part_ids[...,None,None,None], axis=0).float().cpu()

    return mask

mask_types = [
    'simple_square',
    'convex_hull',
    'jonathandinu',
]

mask_funs = {
    'simple_square': mask_simple_square,
    'convex_hull': mask_convex_hull,
    'jonathandinu': lambda face, M, crop: mask_jonathandinu(crop),
}

def mask_crop(face, M, crop, mask_type):
    mask = mask_funs[mask_type](face, M, crop)
    return mask




## InstantID start

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor

## InstantID end
