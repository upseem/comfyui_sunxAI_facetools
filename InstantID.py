import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import time
import PIL.Image
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention
from .utils import tensor_to_image

from insightface.app import FaceAnalysis

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F

MODELS_DIR = os.path.join(folder_paths.models_dir, "instantid")

if "instantid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["instantid"]

folder_paths.folder_names_and_paths["instantid"] = (current_paths, folder_paths.supported_pt_extensions)

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class InstantID(torch.nn.Module):
    def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        self.ip_layers = To_KV(instantid_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        #image_prompt_embeds = clip_embed.clone().detach()
        image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = clip_embed_zeroed.clone().detach()
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

        return image_prompt_embeds, uncond_image_prompt_embeds

class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = torch.nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value

def _set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()

    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(instantid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(instantid_attention, **patch_kwargs)

class InstantIDModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "instantid_file": (folder_paths.get_filename_list("instantid"), )}}

    RETURN_TYPES = ("INSTANTID",)
    FUNCTION = "load_model"
    CATEGORY = "sunxAI_facetools"

    def load_model(self, instantid_file):
        ckpt_path = folder_paths.get_full_path("instantid", instantid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        model = InstantID(
            model,
            cross_attention_dim=1280,
            output_cross_attention_dim=model["ip_adapter"]["1.to_k_ip.weight"].shape[1],
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        )

        return (model,)

def extractFeatures(insightface, image, extract_kps=False):
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(face_img[i])
            if face:
                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

                if extract_kps:
                    out.append(draw_kps(face_img[i], face['kps']))
                else:
                    out.append(torch.from_numpy(face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        else:
            out = torch.stack(out, dim=0)
    else:
        out = None

    return out

class InstantIDFaceAnalysis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM", "CoreML"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insight_face"
    CATEGORY = "sunxAI_facetools"

    def load_insight_face(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)


def add_noise(image, factor):
    seed = int(torch.sum(image).item()) % 1000000007
    torch.manual_seed(seed)
    mask = (torch.rand_like(image) < factor).float()
    noise = torch.rand_like(image)
    noise = torch.zeros_like(image) * (1-mask) + noise * mask

    return factor*noise

class ApplyInstantID:
    """
    应用InstantID的主要节点
    功能：将人脸身份信息应用到模型中，实现换脸效果

    参数说明：
    - image: 参考人脸图像，用于提取人脸特征嵌入
    - image_kps: 可选的关键点图像，用于生成ControlNet的控制信号
      * 如果提供：直接使用该图像提取人脸关键点
      * 如果不提供：使用参考图像(image)的第一张提取关键点
    - face_embed: 可选的预计算人脸嵌入，如果提供则跳过人脸检测和特征提取
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid": ("INSTANTID", ),  # InstantID模型
                "insightface": ("FACEANALYSIS", ),  # InsightFace人脸分析模型
                "control_net": ("CONTROL_NET", ),  # ControlNet模型
                "image": ("IMAGE", ),  # 参考人脸图像
                "model": ("MODEL", ),  # 基础扩散模型
                "positive": ("CONDITIONING", ),  # 正向条件
                "negative": ("CONDITIONING", ),  # 负向条件
                "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),  # 权重
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),  # 开始时间步
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),  # 结束时间步
            },
            "optional": {
                "image_kps": ("IMAGE",),  # 可选：关键点图像，用于ControlNet
                "mask": ("MASK",),  # 可选：遮罩
                "face_embed": ("FACE_EMBEDS",),  # 可选：预计算的人脸嵌入
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "FACE_EMBEDS", "BOOLEAN")
    RETURN_NAMES = ("MODEL", "positive", "negative", "face_embed", "has_face")
    FUNCTION = "apply_instantid"
    CATEGORY = "sunxAI_facetools"

    def apply_instantid(self, instantid, insightface, control_net, image, model, positive, negative, start_at, end_at, weight=.8, ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average', face_embed=None):
        """
        应用InstantID换脸效果

        处理流程：
        1. 人脸嵌入处理：
           - 如果提供face_embed：直接使用预计算的嵌入，跳过人脸检测
           - 如果未提供：从image中检测人脸并提取特征嵌入

        2. 关键点处理：
           - 如果提供image_kps：使用该图像提取人脸关键点用于ControlNet
           - 如果未提供：使用参考图像(image)的第一张图片提取关键点

        3. 模型修补：将人脸嵌入注入到模型的注意力层中
        4. ControlNet应用：使用人脸关键点控制生成过程
        """
        # 如果end_at为0，直接返回原始数据，跳过所有处理
        if end_at == 0 or weight == 0:
            print(f"\033[33mINFO: end_at=0 or weight=0，跳过InstantID处理\033[0m")
            return (model, positive, negative, None, False)


        start_total = time.time()
        print(f"\033[36m=== InstantID 处理开始 ===\033[0m")

        # 设置数据类型和设备
        start_setup = time.time()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 设置权重参数
        ip_weight = weight if ip_weight is None else ip_weight  # IP-Adapter权重
        cn_strength = weight if cn_strength is None else cn_strength  # ControlNet强度
        print(f"\033[36m初始化设置耗时: {time.time() - start_setup:.3f}s\033[0m")

        # === 人脸嵌入处理 ===
        start_embed = time.time()
        # 如果提供了预计算的face_embed，直接使用；否则从图像提取
        if face_embed is not None:
            # 检查是否为无人脸标记
            if face_embed.get('no_face', False):
                print(f"\033[33mINFO: 使用预计算的无人脸标记，跳过InstantID处理\033[0m")
                return (model, positive, negative, face_embed, False)

            print(f"\033[32mINFO: 使用预计算的人脸嵌入（已在GPU）\033[0m")
            # 数据已在LoadFaceEmbeds中加载到GPU，直接使用
            image_prompt_embeds = face_embed['cond']
            uncond_image_prompt_embeds = face_embed['uncond']
            output_face_embed = None  # 已有embed，不需要输出新的
        else:
            print(f"\033[32mINFO: 从参考图像提取人脸特征\033[0m")
            # 从参考图像中提取人脸特征
            start_face_detect = time.time()
            face_embed_raw = extractFeatures(insightface, image)
            print(f"\033[36m人脸检测耗时: {time.time() - start_face_detect:.3f}s\033[0m")

            if face_embed_raw is None:
                print(f"\033[33mWARNING: 参考图像中未检测到人脸, 创建无人脸标记并返回\033[0m")
                # 创建无人脸标记，避免下次重复检测
                no_face_embed = {
                    'no_face': True,
                    'timestamp': int(time.time())
                }
                return (model, positive, negative, no_face_embed, False)

            start_embed_process = time.time()
            clip_embed = face_embed_raw
            # InstantID使用平均嵌入效果更好
            if clip_embed.shape[0] > 1:
                if combine_embeds == 'average':
                    clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
                elif combine_embeds == 'norm average':
                    clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

            # 添加噪声到负向嵌入
            if noise > 0:
                seed = int(torch.sum(clip_embed).item()) % 1000000007
                torch.manual_seed(seed)
                clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)
            print(f"\033[36m嵌入预处理耗时: {time.time() - start_embed_process:.3f}s\033[0m")

            # 使用InstantID模型处理嵌入
            start_instantid = time.time()
            instantid_model = instantid
            instantid_model.to(device, dtype=dtype)

            image_prompt_embeds, uncond_image_prompt_embeds = instantid_model.get_image_embeds(clip_embed.to(device, dtype=dtype), clip_embed_zeroed.to(device, dtype=dtype))

            image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)
            print(f"\033[36mInstantID模型处理耗时: {time.time() - start_instantid:.3f}s\033[0m")

            # 保存生成的face_embed用于下次使用
            output_face_embed = { "cond": image_prompt_embeds, "uncond": uncond_image_prompt_embeds }

        print(f"\033[36m人脸嵌入总耗时: {time.time() - start_embed:.3f}s\033[0m")

        # === 关键点处理 ===
        start_kps = time.time()
        # 如果没有提供关键点图像，使用参考图像的第一张提取关键点
        # image_kps用于ControlNet控制生成的人脸姿态和表情
        if image_kps is not None:
            print(f"\033[32mINFO: 使用提供的关键点图像\033[0m")
            face_kps = extractFeatures(insightface, image_kps, extract_kps=True)
        else:
            print(f"\033[32mINFO: 从参考图像提取关键点\033[0m")
            face_kps = extractFeatures(insightface, image[0].unsqueeze(0), extract_kps=True)

        # 如果关键点提取失败，使用零张量占位
        if face_kps is None:
            face_kps = torch.zeros_like(image) if image_kps is None else image_kps
            print(f"\033[33mWARNING: 关键点图像中未检测到人脸，可能影响控制效果\033[0m")
        print(f"\033[36m关键点提取耗时: {time.time() - start_kps:.3f}s\033[0m")

        # === 模型修补 ===
        start_patch = time.time()
        # 克隆模型以避免影响原始模型
        work_model = model.clone()

        # 计算时间步范围
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        if mask is not None:
            mask = mask.to(device)

        # 准备修补参数
        patch_kwargs = {
            "ipadapter": instantid,
            "weight": ip_weight,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        # 修补模型的注意力层
        number = 0
        # 输入块
        for id in [4,5,7,8]:
            block_indices = range(2) if id in [4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        # 输出块
        for id in range(6):
            block_indices = range(2) if id in [3, 4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        # 中间块
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
            number += 1
        print(f"\033[36m模型修补耗时: {time.time() - start_patch:.3f}s\033[0m")

        # === ControlNet应用 ===
        start_controlnet = time.time()
        # 处理遮罩维度
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        # 为正向和负向条件分别应用ControlNet
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                # 获取或创建ControlNet
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    # 使用人脸关键点作为ControlNet的控制信号
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                # 设置跨注意力ControlNet嵌入
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                # 应用遮罩（仅对正向条件）
                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False
        print(f"\033[36mControlNet应用耗时: {time.time() - start_controlnet:.3f}s\033[0m")

        total_time = time.time() - start_total
        print(f"\033[36m=== InstantID 处理完成，总耗时: {total_time:.3f}s ===\033[0m")

        # 优化建议
        if output_face_embed is not None:
            print(f"\033[33m💡 优化建议: 保存生成的face_embed可节省 {time.time() - start_embed:.3f}s 的人脸处理时间\033[0m")

        return(work_model, cond_uncond[0], cond_uncond[1], output_face_embed , True)


class SaveFaceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_embed": ("FACE_EMBEDS",),
                "name": ("STRING", {"default": "face_embed"}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_face_embed"
    CATEGORY = "sunxAI_facetools"
    OUTPUT_NODE = True

    def save_face_embed(self, face_embed, name):
        # 检查face_embed是否为None或空
        if face_embed is None:
            print(f"\033[33mWARNING: Face embed is None, skipping save.\033[0m")
            return {}

        # 创建保存目录
        save_dir = os.path.join(folder_paths.models_dir, "face_embeds")
        os.makedirs(save_dir, exist_ok=True)

        # 确保文件名以.pt结尾
        filename = name + '.pt'
        filepath = os.path.join(save_dir, filename)

        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"\033[33mWARNING: File {filepath} already exists, skipping save.\033[0m")
            return {}

        # 处理不同类型的face_embed数据
        save_data = {}

        # 检查是否为无人脸标记
        if face_embed.get('no_face', False):
            print(f"\033[32mINFO: Saving no-face marker to {filepath}\033[0m")
            save_data = {
                "no_face": True,
                "timestamp": face_embed.get('timestamp', int(time.time()))
            }
        else:
            # 正常的人脸嵌入数据 - 保存到磁盘，立即释放内存
            if 'cond' in face_embed and 'uncond' in face_embed:
                print(f"\033[32mINFO: Saving face embeddings to {filepath}\033[0m")
                save_data = {
                    "cond": face_embed["cond"].cpu(),
                    "uncond": face_embed["uncond"].cpu(),
                    "timestamp": int(time.time())
                }
            else:
                print(f"\033[33mWARNING: Invalid face_embed format, missing 'cond' or 'uncond' fields\033[0m")
                return {}

        # 保存到磁盘
        torch.save(save_data, filepath)

        # 立即清理内存
        del save_data
        print(f"\033[32mINFO: Face embed data saved successfully and memory cleared\033[0m")

        return {}

class LoadFaceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "face_embed"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("FACE_EMBEDS",)
    RETURN_NAMES = ("face_embed",)
    FUNCTION = "load_face_embed"
    CATEGORY = "sunxAI_facetools"

    def load_face_embed(self, name, seed=0):
        face_embeds_dir = os.path.join(folder_paths.models_dir, "face_embeds")

        # 确保文件名以.pt结尾
        filename = name + '.pt'
        filepath = os.path.join(face_embeds_dir, filename)

        if not os.path.exists(filepath):
            print(f"\033[33mWARNING: Face embed file not found: {filepath}, returning None\033[0m")
            return (None,)

        try:
            print(f"\033[36mINFO: Loading face embed (reload={seed}): {filename}\033[0m")

            # 加载人脸嵌入数据到CPU
            save_data = torch.load(filepath, map_location="cpu")

            # 检查是否为无人脸标记
            if save_data.get('no_face', False):
                print(f"\033[32mINFO: Loaded no-face marker from {filepath}\033[0m")
                face_embed = {
                    "no_face": True,
                    "timestamp": save_data.get('timestamp', 0)
                }
                del save_data
                return (face_embed,)

            # 正常的人脸嵌入数据 - 直接加载到GPU
            if 'cond' in save_data and 'uncond' in save_data:
                print(f"\033[32mINFO: Loading face embeddings to GPU from {filepath}\033[0m")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                face_embed = {
                    "cond": save_data["cond"].to(device, dtype=dtype),
                    "uncond": save_data["uncond"].to(device, dtype=dtype),
                    "timestamp": save_data.get('timestamp', 0)
                }

                # 清理CPU数据
                del save_data
                print(f"\033[32mINFO: Face embeddings loaded to {device} and CPU cache cleared\033[0m")

                return (face_embed,)
            else:
                print(f"\033[33mWARNING: Invalid saved data format, missing 'cond' or 'uncond' fields\033[0m")
                del save_data
                return (None,)

        except Exception as e:
            print(f"\033[31mERROR: Failed to load face embed from {filepath}: {e}\033[0m")
            return (None,)



