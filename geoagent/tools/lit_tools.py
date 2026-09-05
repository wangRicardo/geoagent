"""文献 RAG：arXiv 检索下载 → PDF/TXT 解析入库 → 本地检索 → 带引用的问答。

索引完全本地（TF-IDF 余弦相似度，纯 numpy 实现，无外部 API、离线可用），
存放在工作区 ``.ricardo_lit/`` 目录，随工作区隔离——每个课题一套文献库。

典型闭环：
  arxiv_search 检索 → arxiv_download 下载 PDF → lit_ingest 入库
  → lit_search 精确定位原文 → lit_ask 基于文献回答（带来源引用）
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from datetime import datetime
from typing import List, Optional

import numpy as np
import requests

from .base import registry

UA = {"User-Agent": "RicardoAgent/0.6 (research agent; geophysics)"}
LIT_DIR = ".ricardo_lit"          # 工作区内的索引目录
CHUNK_SIZE = 1200                  # 字符
CHUNK_OVERLAP = 200
ARXIV_API = "http://export.arxiv.org/api/query"


# ---------------------------------------------------------------------------
# 分词与 TF-IDF（本地实现，中文按字 bigram，英文按单词）
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{2,}|\d+\.?\d*", text.lower())
    # 中文：字 bigram（覆盖 0x4e00-0x9fff）
    for m in re.finditer(r"[\u4e00-\u9fff]+", text):
        seg = m.group()
        tokens += [seg[i:i + 2] for i in range(len(seg) - 1)]
    return tokens


class TfidfIndex:
    """轻量 TF-IDF 向量索引：文档 → 稀疏向量（dict 词 -> 权重），余弦检索。"""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.doc_vecs: List[dict[str, float]] = []
        self.doc_norms: np.ndarray | None = None

    def _fit_idf(self, docs_tokens: List[List[str]]) -> None:
        n = len(docs_tokens)
        df: dict[str, int] = {}
        for toks in docs_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    def _vec(self, toks: List[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        return {w: (1 + np.log(c)) * self.idf.get(w, np.log(2.0) + 1.0)
                for w, c in tf.items()}

    def fit(self, docs_tokens: List[List[str]]) -> None:
        self._fit_idf(docs_tokens)
        self.doc_vecs = [self._vec(t) for t in docs_tokens]
        self.doc_norms = np.array([np.sqrt(sum(v * v for v in d.values())) or 1.0
                                   for d in self.doc_vecs])

    def transform(self, toks: List[str]) -> dict[str, float]:
        return self._vec(toks)

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        na = np.sqrt(sum(v * v for v in a.values())) or 1.0
        nb = np.sqrt(sum(v * v for v in b.values())) or 1.0
        dot = sum(v * b.get(w, 0.0) for w, v in a.items())
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# 索引存取（工作区 .ricardo_lit/）
# ---------------------------------------------------------------------------

def _lit_dir() -> str:
    d = os.path.join(os.getcwd(), LIT_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def _load_db() -> dict:
    path = os.path.join(_lit_dir(), "db.json")
    if os.path.exists(path):
        try:
            return json.loads(open(path, encoding="utf-8").read())
        except (json.JSONDecodeError, OSError):
            pass
    return {"chunks": [], "sources": {}}


def _save_db(db: dict) -> None:
    with open(os.path.join(_lit_dir(), "db.json"), "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)


def _build_index(db: dict) -> TfidfIndex:
    idx = TfidfIndex()
    idx.fit([_tokenize(c["text"]) for c in db["chunks"]])
    return idx


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("需要 pypdf 库: pip install pypdf")
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - 个别页损坏时跳过
            pages.append("")
    return "\n".join(f"[page {i + 1}] {p}" for i, p in enumerate(pages))


def _hard_split(text: str) -> List[str]:
    """把超长文本按句子边界切成 CHUNK_SIZE 窗口，窗口间保留 overlap。"""
    sentences = re.split(r"(?<=[.。!?？！])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        while len(s) > CHUNK_SIZE:  # 无句子边界的超长串，硬切
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:CHUNK_SIZE])
            s = s[CHUNK_SIZE - CHUNK_OVERLAP:]
        if len(cur) + len(s) + 1 > CHUNK_SIZE:
            chunks.append(cur)
            cur = cur[-CHUNK_OVERLAP:] + " " + s
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if len(c.strip()) > 50]


def _chunk_text(text: str) -> List[str]:
    """段落聚合分块；PDF 提取的文本常缺段落分隔，超长段落自动按句切窗。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return _hard_split(text)
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(p) > CHUNK_SIZE:
            if cur.strip():
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(p))
            continue
        if len(cur) + len(p) + 2 > CHUNK_SIZE:
            if cur:
                chunks.append(cur)
            cur = (chunks[-1][-CHUNK_OVERLAP:] + "\n" + p) if chunks else p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

@registry.register(category="rag")
def lit_ingest(paths: List[str]) -> str:
    """把工作区内的 PDF / TXT / Markdown 文献解析入库并重建检索索引。

    paths 支持具体文件或目录（目录会递归收集 .pdf/.txt/.md）。
    重复文件会跳过（按路径判重）。入库后可用 lit_search / lit_ask 检索问答。
    """
    root = os.getcwd()
    files: List[str] = []
    for p in paths:
        full = os.path.abspath(os.path.join(root, p))
        if os.path.commonpath([root, full]) != root:
            return f"ERROR: 路径越出工作目录: {p}"
        if os.path.isdir(full):
            for dirpath, _, names in os.walk(full):
                files += [os.path.join(dirpath, n) for n in names
                          if n.lower().endswith((".pdf", ".txt", ".md"))]
        elif os.path.isfile(full):
            files.append(full)
        else:
            return f"ERROR: 路径不存在: {p}"
    if not files:
        return "ERROR: 没有找到 .pdf/.txt/.md 文件"

    db = _load_db()
    known = {c["source"] for c in db["chunks"]}
    added, skipped = 0, 0
    for f in files:
        rel = os.path.relpath(f, root)
        if rel in known:
            skipped += 1
            continue
        try:
            if f.lower().endswith(".pdf"):
                text = _extract_pdf(f)
            else:
                text = open(f, encoding="utf-8", errors="replace").read()
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: 解析 {rel} 失败: {exc}"
        if not text.strip():
            skipped += 1
            continue
        for chunk in _chunk_text(text):
            db["chunks"].append({"source": rel, "text": chunk})
        db["sources"][rel] = {"pages": text.count("[page"), "chars": len(text),
                              "ingested": datetime.now().isoformat(timespec="seconds")}
        added += 1
    _save_db(db)
    n = len(db["chunks"])
    return (f"入库完成: 新增 {added} 篇（跳过 {skipped}），文献库共 {n} 个文本块、"
            f"{len(db['sources'])} 篇文献 → {LIT_DIR}/db.json")


@registry.register(category="rag")
def lit_search(query: str, k: int = 5) -> str:
    """在已入库的文献库中检索最相关的文本块（TF-IDF 余弦，本地离线）。"""
    db = _load_db()
    if not db["chunks"]:
        return "文献库为空。先用 lit_ingest 导入 PDF/TXT，或 arxiv_download 下载文献。"
    idx = _build_index(db)
    qv = idx.transform(_tokenize(query))
    scores = [TfidfIndex.cosine(qv, dv) for dv in idx.doc_vecs]
    order = np.argsort(scores)[::-1][:max(1, min(k, 8))]
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        c = db["chunks"][int(i)]
        snippet = c["text"][:300].replace("\n", " ")
        out.append(f"[{scores[i]:.3f}] {c['source']}\n    {snippet}…")
    if not out:
        return f"没有检索到与 {query!r} 相关的内容。"
    return f"检索结果（{query!r}）:\n" + "\n\n".join(out)


@registry.register(category="rag")
def lit_ask(question: str, k: int = 5) -> str:
    """基于文献库回答问题（RAG）：先检索相关段落，再让模型综合并标注来源。

    需要配置 API 密钥；检索部分始终本地离线。
    """
    db = _load_db()
    if not db["chunks"]:
        return "文献库为空。先用 lit_ingest 导入文献。"
    idx = _build_index(db)
    qv = idx.transform(_tokenize(question))
    scores = [TfidfIndex.cosine(qv, dv) for dv in idx.doc_vecs]
    order = np.argsort(scores)[::-1][:max(1, min(k, 8))]
    ctx, refs = [], []
    for rank, i in enumerate(order, 1):
        if scores[i] <= 0:
            break
        c = db["chunks"][int(i)]
        ctx.append(f"[{rank}] ({c['source']}) {c['text']}")
        if c["source"] not in refs:
            refs.append(c["source"])
    if not ctx:
        return f"文献库中没有与 {question!r} 相关的内容。"

    from ..config import AgentConfig
    cfg = AgentConfig()
    if not cfg.api_key:
        return (
            "ERROR: lit_ask 的综合回答需要 API 密钥；纯检索请用 lit_search（离线可用）。"
            f"\n已检索到 {len(ctx)} 个相关段落，来源: {', '.join(refs)}"
        )
    try:
        from openai import OpenAI
    except ImportError:
        return "ERROR: 需要 openai 库: pip install openai"
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    prompt = (
        "你是文献研究助手。仅依据下面提供的文献片段回答问题；"
        "引用时在句末标注来源编号如 [1]；片段不足以回答时明确说明。\n\n"
        f"问题: {question}\n\n文献片段:\n" + "\n\n".join(ctx)
    )
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: LLM 调用失败: {exc}"
    return f"{answer}\n\n参考来源: {', '.join(refs)}"


@registry.register(category="rag")
def lit_status() -> str:
    """查看文献库状态：文献数、文本块数、各文献的入库时间与规模。"""
    db = _load_db()
    if not db["sources"]:
        return f"文献库为空（索引目录 {LIT_DIR}/）。用 lit_ingest 或 arxiv_download 开始。"
    lines = [f"文献 {len(db['sources'])} 篇 / 文本块 {len(db['chunks'])} 个:"]
    for src, meta in db["sources"].items():
        lines.append(f"  {src} — {meta['chars']:,} 字符, {meta['pages']} 页, {meta['ingested']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# arXiv 检索与下载
# ---------------------------------------------------------------------------

def _parse_arxiv_feed(xml: str) -> List[dict]:
    entries = []
    for m in re.finditer(r"<entry>(.*?)</entry>", xml, re.S):
        e = m.group(1)
        get = lambda tag: (re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S).group(1).strip()
                           if re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S) else "")
        title = re.sub(r"\s+", " ", get("title"))
        summary = re.sub(r"\s+", " ", get("summary"))
        authors = re.findall(r"<name>(.*?)</name>", e)
        link = get("id")
        entries.append({"id": link.split("/abs/")[-1], "title": title,
                        "authors": authors, "summary": summary, "url": link})
    return entries


@registry.register(category="rag")
def arxiv_search(query: str, max_results: int = 8, sort_by: str = "relevance") -> str:
    """检索 arXiv 文献（地球物理常用分类自动包含），返回标题/作者/摘要/编号。

    query 支持 arXiv 语法，如 'all:ambient noise tomography' 或
    'cat:physics.geo-ph AND all:machine learning'。sort_by: relevance|submittedDate。
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max(1, min(max_results, 20)),
        "sortBy": sort_by,
    }
    try:
        r = requests.get(ARXIV_API, params=params, headers=UA, timeout=30)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: arXiv 检索失败: {exc}"
    entries = _parse_arxiv_feed(r.text)
    if not entries:
        return f"arXiv 没有返回与 {query!r} 匹配的文献。"
    out = [f"arXiv 检索结果 {len(entries)} 条:"]
    for e in entries:
        out.append(
            f"  [{e['id']}] {e['title']}\n"
            f"      {', '.join(e['authors'][:4])}{'等' if len(e['authors']) > 4 else ''}\n"
            f"      {e['summary'][:180]}…"
        )
    out.append("用 arxiv_download 下载编号对应的 PDF。")
    return "\n".join(out)


@registry.register(category="rag")
def arxiv_download(arxiv_id: str, ingest: bool = True) -> str:
    """下载 arXiv 论文 PDF 到工作区；默认随后自动 lit_ingest 入库。

    arxiv_id 形如 '2401.12345' 或 '2401.12345v2'。
    """
    arxiv_id = arxiv_id.strip()
    if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id) and "/" not in arxiv_id:
        return f"ERROR: arXiv 编号格式不正确: {arxiv_id}（示例 2401.12345）"
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            return f"ERROR: 返回内容不是 PDF（可能被限流），稍后重试"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: 下载失败: {exc}"
    fname = f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
    root = os.getcwd()
    full = os.path.abspath(os.path.join(root, fname))
    if os.path.commonpath([root, full]) != root:
        return f"ERROR: 路径越出工作目录"
    with open(full, "wb") as f:
        f.write(r.content)
    out = f"已下载: {fname}（{len(r.content):,} 字节）"
    if ingest:
        out += "\n" + lit_ingest([fname])
    return out
