# comfyui_sunxAI_facetools

Face detection & restoration tools for ComfyUI by Sunx.ai

> [!NOTE]
> This projected was created with a [cookiecutter](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template. It helps you start writing custom nodes without worrying about the Python setup.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features

- **DetectFaces**: 检测图像中的人脸，返回面积最大的人脸
- **DetectFaceByIndex**: 根据索引检测人脸，支持从左到右选择特定人脸（0=最左边，1=第二个，以此类推），支持性别筛选（0=任意性别，1=男性，2=女性）。性别检测只在需要时启用，提高性能。
- **DetectFaceByGender**: 根据性别和索引检测人脸，支持筛选男性/女性人脸，并按从左到右顺序选择
- **CropFaces**: 裁剪检测到的人脸区域
- **WarpFaceBack**: 将处理后的脸贴回原图
- **InstantID**: 人脸身份保持功能
- **ColorAdjust**: 人脸颜色调整
- **SaveImageWebsocket**: 通过 WebSocket 保存图像

## 安装依赖

本插件使用 InsightFace 进行高精度性别检测，首次使用时会自动下载模型：

```bash
pip install insightface
```

**注意**: InsightFace 需要额外的模型文件，首次运行时会自动下载。

### 性别检测特性
- 使用 InsightFace 进行高精度性别识别
- 支持 GPU 和 CPU 模式，自动选择最佳设备
- 提供年龄检测和置信度信息
- 支持 fallback 方案（基于面部宽高比）
- 防止除零错误，确保稳定运行

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd comfyui_sunxAI_facetools
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name.
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
```

## Writing custom nodes

An example custom node is located in [node.py](src/comfyui_sunxAI_facetools/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile).
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

