"""Agent 记忆系统：研究笔记的持久化保存与检索。

让 agent 在跨会话时记住研究结论。笔记以 JSON 存放在
``~/.geoagent/notes.json``，全部工具可离线使用。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .base import registry

NOTES_PATH = Path.home() / ".geoagent" / "notes.json"


def _load() -> list:
    if NOTES_PATH.exists():
        try:
            return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save(notes: list) -> None:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@registry.register(category="memory")
def save_note(title: str, content: str, tags: str = "") -> str:
    """保存一条研究笔记（标题+正文，逗号分隔的标签可选），持久化到本地。"""
    notes = _load()
    notes.append(
        {
            "id": (max((n["id"] for n in notes), default=0)) + 1,
            "title": title,
            "content": content,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _save(notes)
    return f"已保存笔记 #{notes[-1]['id']}《{title}》（共 {len(notes)} 条）→ {NOTES_PATH}"


@registry.register(category="memory")
def list_notes(tag: str = "") -> str:
    """列出全部研究笔记；给定 tag 时只列出含该标签的笔记。"""
    notes = _load()
    if tag:
        notes = [n for n in notes if tag in n["tags"]]
    if not notes:
        return "（还没有符合条件的笔记）"
    return "\n".join(
        f"#{n['id']} [{n['time']}] ({','.join(n['tags']) or '无标签'}) {n['title']}"
        for n in notes
    )


@registry.register(category="memory")
def read_note(note_id: int) -> str:
    """按 id 读取一条研究笔记的完整内容。"""
    for n in _load():
        if n["id"] == note_id:
            return f"#{n['id']}《{n['title']}》 {n['time']}\n{n['content']}"
    return f"ERROR: 不存在 id={note_id} 的笔记"


@registry.register(category="memory")
def search_notes(keyword: str) -> str:
    """在笔记标题与正文里按关键词模糊检索。"""
    hits = [
        n for n in _load()
        if keyword.lower() in n["title"].lower() or keyword.lower() in n["content"].lower()
    ]
    if not hits:
        return f"没有包含 {keyword!r} 的笔记"
    return "\n".join(f"#{n['id']} {n['title']}: {n['content'][:80]}…" for n in hits)
