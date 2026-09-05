"""文献 RAG：arXiv 检索下载 → PDF/TXT 解析入库 → 语义检索 → 带引用的问答。

检索三级后端（按可用性自动选择，lit_reindex 可显式切换）：
  1. provider   —— 当前提供商的 /embeddings 接口（可用 GEOAGENT_EMBED_MODEL 指定模型）
  2. local      —— 本地 sentence-transformers 模型（RICARDO_LOCAL_EMBED_MODEL，默认多语 MiniLM）
  3. tfidf      —— 本地 TF-IDF 余弦（纯 numpy，永远可用，兜底）
索引存放在工作区 ``.ricardo_lit/``（chunks 在 db.json，向量在 embeddings.npy），
随工作区隔离——每个课题一套文献库。

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
EMBED_BATCH = 32

# 各提供商默认的 embedding 模型（None 表示该提供商没有 embeddings 端点）
PROVIDER_EMBED_MODELS = {
    "openai": "text-embedding-3-small",
    "zhipu": "embedding-3",
    "qwen": "text-embedding-v3",
    "siliconflow": "BAAI/bge-m3",
    "ollama": "nomic-embed-text",
    "deepseek": None,
    "moonshot": None,
}
LOCAL_EMBED_MODEL = os.environ.get(
    "RICARDO_LOCAL_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)


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
# 嵌入后端
# ---------------------------------------------------------------------------

def _embed_provider(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    """调用当前提供商的 /embeddings 接口，返回 (n, dim) 归一化向量。"""
    from ..config import AgentConfig
    cfg = AgentConfig()
    if not cfg.api_key:
        raise RuntimeError("未配置 API 密钥，provider 嵌入不可用")
    model = model or os.environ.get("GEOAGENT_EMBED_MODEL") \
        or PROVIDER_EMBED_MODELS.get(cfg.provider)
    if not model:
        raise RuntimeError(
            f"提供商 {cfg.provider} 没有 embeddings 端点；"
            "可用 GEOAGENT_EMBED_MODEL 显式指定兼容端点的模型，"
            "或 lit_reindex backend=local / tfidf"
        )
    from openai import OpenAI
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    vecs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [t[:6000] for t in texts[i:i + EMBED_BATCH]]
        resp = client.embeddings.create(model=model, input=batch)
        vecs += [d.embedding for d in resp.data]
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def _embed_local(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    """本地 sentence-transformers 嵌入（首次使用会下载模型）。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "本地嵌入需要 sentence-transformers: pip install sentence-transformers"
        )
    st = SentenceTransformer(model or LOCAL_EMBED_MODEL)
    arr = np.asarray(st.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=False),
                     dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def _get_embedder(backend: str):
    """backend -> embed 函数；不可用抛 RuntimeError。"""
    if backend == "provider":
        return _embed_provider
    if backend == "local":
        return _embed_local
    raise RuntimeError(f"未知嵌入后端: {backend}")


def _current_provider() -> str:
    from ..config import AgentConfig
    return AgentConfig().provider


def _active_backend(db: dict) -> str:
    """数据库当前向量索引的后端名（tfidf 或 provider/local）。"""
    return db.get("embed", {}).get("backend", "tfidf")


def _embed_texts(db: dict, texts: List[str], backend: Optional[str] = None) -> np.ndarray:
    """按指定后端嵌入；backend=None 时沿用库中现有后端（缺省 tfidf）。"""
    backend = backend or _active_backend(db)
    if backend == "tfidf":
        raise RuntimeError("tfidf 后端不产生向量")
    return _get_embedder(backend)(texts)


def _semantic_scores(db: dict, query: str, backend: Optional[str] = None):
    """语义检索打分；后端不可用时返回 None（调用方回退 TF-IDF）。"""
    if "embed" not in db:
        return None
    vec_path = os.path.join(_lit_dir(), "embeddings.npy")
    if not os.path.exists(vec_path):
        return None
    mat = np.load(vec_path)
    if len(mat) != len(db["chunks"]):
        return None
    try:
        qv = _embed_texts(db, [query], backend)[0]
    except RuntimeError:
        return None
    return mat @ qv  # 已归一化，点积即余弦


def _tfidf_scores(db: dict, query: str) -> np.ndarray:
    idx = _build_index(db)
    qv = idx.transform(_tokenize(query))
    return np.array([TfidfIndex.cosine(qv, dv) for dv in idx.doc_vecs])


def _save_vectors(vectors: np.ndarray) -> None:
    np.save(os.path.join(_lit_dir(), "embeddings.npy"), vectors)


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
    new_chunks: List[dict] = []
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
            new_chunks.append({"source": rel, "text": chunk})
        db["sources"][rel] = {"pages": text.count("[page"), "chars": len(text),
                              "ingested": datetime.now().isoformat(timespec="seconds")}
        added += 1
    db["chunks"] += new_chunks
    # 已有语义索引时，为新块增量补向量；失败则降级为 TF-IDF 库
    if "embed" in db and new_chunks:
        try:
            vecs = _embed_texts(db, [c["text"] for c in new_chunks])
            vec_path = os.path.join(_lit_dir(), "embeddings.npy")
            old = np.load(vec_path) if os.path.exists(vec_path) else np.zeros((0, vecs.shape[1]), np.float32)
            _save_vectors(np.vstack([old, vecs]))
        except RuntimeError as exc:
            db.pop("embed", None)
            if os.path.exists(os.path.join(_lit_dir(), "embeddings.npy")):
                os.remove(os.path.join(_lit_dir(), "embeddings.npy"))
            _save_db(db)
            return (f"ERROR: 语义索引增量更新失败（{exc}），已回退 TF-IDF。"
                    f"可稍后运行 lit_reindex 重建语义索引。")
    _save_db(db)
    n = len(db["chunks"])
    backend = _active_backend(db)
    return (f"入库完成: 新增 {added} 篇（跳过 {skipped}），文献库共 {n} 个文本块、"
            f"{len(db['sources'])} 篇文献（检索后端: {backend}）→ {LIT_DIR}/db.json")


@registry.register(category="rag")
def lit_search(query: str, k: int = 5, backend: str = "") -> str:
    """在文献库中检索最相关的文本块。

    默认用库中已有的索引后端（语义向量优先，TF-IDF 兜底）；backend 可显式
    指定 provider / local / tfidf。语义后端不可用时自动回退 TF-IDF 并说明。
    """
    db = _load_db()
    if not db["chunks"]:
        return "文献库为空。先用 lit_ingest 导入 PDF/TXT，或 arxiv_download 下载文献。"
    wanted = (backend or _active_backend(db)).lower()
    scores, used, note = None, wanted, ""
    if wanted != "tfidf" and "embed" not in db:
        # 懒升级：库还是 TF-IDF 时，显式请求语义后端会现场构建向量索引
        try:
            vecs = _get_embedder(wanted)([c["text"] for c in db["chunks"]], None)
            if wanted == "provider":
                model_name = os.environ.get("GEOAGENT_EMBED_MODEL") \
                    or PROVIDER_EMBED_MODELS.get(_current_provider()) or wanted
            else:
                model_name = os.environ.get("RICARDO_LOCAL_EMBED_MODEL", LOCAL_EMBED_MODEL)
            db["embed"] = {"backend": wanted, "model": model_name}
            _save_vectors(vecs)
            _save_db(db)
        except RuntimeError:
            pass
    if wanted != "tfidf":
        scores = _semantic_scores(db, query, wanted or None)
        if scores is None:
            used, note = "tfidf", f"（{wanted} 后端不可用，已回退 TF-IDF）"
    if scores is None:
        used = "tfidf"
        scores = _tfidf_scores(db, query)
    order = np.argsort(scores)[::-1][:max(1, min(k, 8))]
    out = [f"检索结果（{query!r}，后端 {used}{note}）:"]
    hit = False
    for i in order:
        if scores[i] <= 0:
            break
        hit = True
        c = db["chunks"][int(i)]
        snippet = c["text"][:300].replace("\n", " ")
        out.append(f"[{scores[i]:.3f}] {c['source']}\n    {snippet}…")
    if not hit:
        return f"没有检索到与 {query!r} 相关的内容。"
    return "\n\n".join(out)


@registry.register(category="rag")
def lit_ask(question: str, k: int = 5) -> str:
    """基于文献库回答问题（RAG）：先检索相关段落，再让模型综合并标注来源。

    需要配置 API 密钥；检索部分始终本地离线。
    """
    db = _load_db()
    if not db["chunks"]:
        return "文献库为空。先用 lit_ingest 导入文献。"
    scores = _semantic_scores(db, question)  # 语义优先，不可用自动回退 TF-IDF
    if scores is None:
        scores = _tfidf_scores(db, question)
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
    lines = [f"文献 {len(db['sources'])} 篇 / 文本块 {len(db['chunks'])} 个 / "
             f"检索后端 {_active_backend(db)}:"]
    for src, meta in db["sources"].items():
        lines.append(f"  {src} — {meta['chars']:,} 字符, {meta['pages']} 页, {meta['ingested']}")
    return "\n".join(lines)


@registry.register(category="rag")
def lit_reindex(backend: str = "provider", model: str = "") -> str:
    """重建文献库的语义检索索引。backend: provider | local | tfidf。

    provider: 用当前提供商的 /embeddings 接口（GEOAGENT_EMBED_MODEL 或按提供商默认）；
    local: 本地 sentence-transformers（RICARDO_LOCAL_EMBED_MODEL）；
    tfidf: 纯本地词频索引（兜底，离线永远可用）。
    语义后端不可用时返回明确原因，库保持原状。
    """
    db = _load_db()
    if not db["chunks"]:
        return "文献库为空，无需重建索引。"
    backend = backend.lower()
    if backend == "tfidf":
        db.pop("embed", None)
        if os.path.exists(os.path.join(_lit_dir(), "embeddings.npy")):
            os.remove(os.path.join(_lit_dir(), "embeddings.npy"))
        _save_db(db)
        return f"已切换为 TF-IDF 索引（{len(db['chunks'])} 个文本块）。"
    if backend not in ("provider", "local"):
        return f"ERROR: backend 可选 provider / local / tfidf，收到 {backend!r}"
    try:
        vecs = _get_embedder(backend)([c["text"] for c in db["chunks"]], model or None)
    except RuntimeError as exc:
        return f"ERROR: {exc}"
    if backend == "provider":
        default_model = model or os.environ.get("GEOAGENT_EMBED_MODEL") \
            or PROVIDER_EMBED_MODELS.get(_current_provider())
    else:
        default_model = model or LOCAL_EMBED_MODEL
    db["embed"] = {"backend": backend, "model": default_model}
    _save_vectors(vecs)
    _save_db(db)
    return (f"语义索引重建完成: 后端 {backend}（{default_model}），"
            f"{vecs.shape[0]} 个文本块 x {vecs.shape[1]} 维。现在同义改述也能命中。")


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
