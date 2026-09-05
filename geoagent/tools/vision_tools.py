"""视觉自查工具：让 agent 能"看"工作区里的图（多模态模型）。

工作方式：`see_image` 把图片编码为 base64 data URL 发给当前提供商的
多模态模型，模型的视觉分析作为工具结果回传——于是 agent 在生成图件后
可以自动调用它检查配色、标签、数据合理性，形成"出图 → 看图 → 修正"闭环。

默认使用当前会话的模型；可用环境变量 GEOAGENT_VISION_MODEL 指定专门的
视觉模型（如 gpt-4o、glm-4v-plus）。需要 API 密钥，offline 模式不可用。
"""

from __future__ import annotations

import base64
import os

from .base import registry

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _encode_image(path: str) -> tuple[str, str]:
    """返回 (mime, base64 data URL)。"""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}[ext]
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return mime, f"data:image/{mime};base64,{b64}"


@registry.register(category="vision")
def see_image(path: str, question: str = "请描述并检查这张图") -> str:
    """让多模态模型查看工作区内的一张图片并回答检查问题（看图自查）。

    path 是工作区内的图片文件（plot_* 工具返回的 PNG 路径可直接传入）；
    question 描述要检查的内容，如"坐标轴标签是否清晰、数据是否有异常跳变"。
    """
    from ..config import AgentConfig

    cfg = AgentConfig()
    if not cfg.api_key:
        return (
            "ERROR: 视觉自查需要 API 密钥（offline 模式无法看图）。"
            "配置密钥后即可让 agent 检查生成的图件。"
        )
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, path))
    if os.path.commonpath([root, full]) != root:
        return f"ERROR: 路径越出工作目录: {path}"
    if not os.path.isfile(full):
        return f"ERROR: 图片不存在: {path}（先用 plot_* 工具生成）"
    if os.path.splitext(full)[1].lower() not in IMAGE_EXTS:
        return f"ERROR: 不支持的图片格式 {os.path.splitext(full)[1]}，支持: {', '.join(sorted(IMAGE_EXTS))}"
    if os.path.getsize(full) > 15 * 1024 * 1024:
        return "ERROR: 图片超过 15MB，过大"

    try:
        from openai import OpenAI
    except ImportError:
        return "ERROR: 需要 openai 库。请先安装: pip install openai"

    mime, data_url = _encode_image(full)
    model = os.environ.get("GEOAGENT_VISION_MODEL", cfg.model)
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "你是科研图件质检员。请针对这张地球物理/数据图件回答：\n"
                                f"{question}\n"
                                "检查维度：坐标轴标签与单位、图例、配色可辨识度、数据异常"
                                "（跳变/空洞/全空白）、整体是否达到论文插图水准。"
                                "先一句话结论（合格/需修正），再列出具体问题。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=800,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"ERROR: 视觉模型调用失败: {exc}\n"
            "提示: 当前模型可能不支持图像输入，可设置环境变量 GEOAGENT_VISION_MODEL "
            "指定多模态模型（如 gpt-4o）。"
        )
    answer = resp.choices[0].message.content or "（模型未返回内容）"
    return f"视觉检查（模型 {model}，图片 {os.path.basename(path)}）:\n{answer}"
