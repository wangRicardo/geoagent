"""文献工具：通过 Crossref API 查询文献并生成 BibTeX 条目。

无需 API key；.polite pool 的 User-Agent 能拿到更稳定的服务。
"""

from __future__ import annotations

import os
import re
import urllib.parse

import requests

from .base import registry

UA = {"User-Agent": "RicardoAgent/0.4 (research agent; geophysics; mailto:ricardo@example.com)"}
API = "https://api.crossref.org/works"


def _bibkey(meta: dict) -> str:
    first = (meta.get("author") or [{}])[0].get("family", "anon")
    year = (meta.get("published", {}).get("date-parts") or [["0000"]])[0][0]
    title = meta.get("title") or ["untitled"]
    words = title[0].split() if isinstance(title[0], str) else ["untitled"]
    word = re.sub(r"[^A-Za-z]", "", words[0]) or "title"
    return f"{first.lower()}{year}{word.lower()}"


def _to_bibtex(meta: dict) -> str:
    key = _bibkey(meta)
    authors = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}" for a in meta.get("author", [])[:8])
    year = (meta.get("published", {}).get("date-parts") or [["0000"]])[0][0]
    lines = [
        f"@article{{{key},",
        f"  title = {{{meta.get('title', [''])[0]}}}",
        f"  author = {{{authors}}}",
        f"  year = {{{year}}}",
    ]
    if meta.get("container-title"):
        lines.append(f"  journal = {{{meta['container-title'][0]}}}")
    if meta.get("volume"):
        lines.append(f"  volume = {{{meta['volume']}}}")
    if meta.get("issue"):
        lines.append(f"  number = {{{meta['issue']}}}")
    if meta.get("page"):
        lines.append(f"  pages = {{{meta['page']}}}")
    lines.append(f"  doi = {{{meta.get('DOI', '')}}}")
    lines.append("}")
    return "\n".join(lines)


@registry.register(category="citation")
def citation_lookup(query: str, rows: int = 5, save_bibtex: str = "") -> str:
    """按关键词（或直接给 DOI）在 Crossref 查询文献，返回标题/作者/年份/期刊/DOI 及 BibTeX。

    query 传 DOI（如 10.1029/2023JB026123）则精确取单篇；否则按相关度检索前 rows 条。
    save_bibtex 给出文件名（如 refs.bib）时把所有条目追加保存到该文件。
    """
    try:
        if re.match(r"^10\.\d{4,}/\S+$", query.strip()):
            r = requests.get(f"{API}/{urllib.parse.quote(query.strip())}", headers=UA, timeout=30)
            items = [r.json()["message"]] if r.ok else []
        else:
            r = requests.get(API, params={"query.bibliographic": query, "rows": rows}, headers=UA, timeout=30)
            items = r.json()["message"]["items"] if r.ok else []
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Crossref 查询失败: {exc}"
    if not items:
        return f"没有找到与 {query!r} 匹配的文献。"

    blocks, bibs = [], []
    for i, meta in enumerate(items, 1):
        authors = (
            ", ".join(f"{a.get('family', '')} {a.get('given', '')[:1]}." for a in meta.get("author", [])[:4])
            or "Unknown"
        )
        year = (meta.get("published", {}).get("date-parts") or [[""]])[0][0]
        journal = (meta.get("container-title") or ["?"])[0]
        cited = meta.get("is-referenced-by-count", 0)
        blocks.append(
            f"[{i}] {meta.get('title', ['?'])[0]}\n"
            f"    {authors} ({year}) {journal}, 被引 {cited}\n"
            f"    DOI: {meta.get('DOI', '')}"
        )
        bibs.append(_to_bibtex(meta))

    out = "\n\n".join(blocks) + "\n\n--- BibTeX ---\n" + "\n\n".join(bibs)
    if save_bibtex:
        root = os.getcwd()
        path = os.path.abspath(os.path.join(root, save_bibtex))
        if os.path.commonpath([root, path]) == root:
            mode = "a" if os.path.exists(path) else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write("\n\n".join(bibs) + "\n\n")
            out += f"\n\n已{'追加' if mode == 'a' else '保存'} {len(bibs)} 条 BibTeX → {save_bibtex}"
    return out
