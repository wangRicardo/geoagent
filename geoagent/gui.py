"""Ricardo Agent 桌面版（Tkinter，零额外依赖）。

与 CLI 共用同一套核心（GeoAgent + AgentConfig）：
- 工作区：启动时选择任意文件夹，运行中可随时更换（类似 codex 选工作区）
- 提供商/模型/思考强度：下拉框热切换，配置持久化到 ~/.geoagent/config.json
- LLM 调用在后台线程执行，界面不卡顿；工具调用过程实时显示
- 未配置密钥时进入 offline 模式，仍可离线使用全部工具

入口：ricardo gui  /  python -m geoagent gui
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from . import GeoAgent, registry
from .config import PROVIDERS, THINKING_LEVELS

BANNER_LINE = "◈ Ricardo Agent — 地球物理 × 机器学习个人研究助手"


class ChatWindow:
    def __init__(self, workdir: str | None = None) -> None:
        self.agent = GeoAgent(workdir=workdir)
        self.root = tk.Tk()
        self.root.title("Ricardo Agent")
        self.root.geometry("980x680")
        self.root.minsize(720, 520)
        self._build_style()
        self._build_ui()
        self._queue: queue.Queue = queue.Queue()
        self.root.after(120, self._poll_queue)
        self._log_system(BANNER_LINE)
        self._log_system(f"工作区: {self.agent.workdir}")
        self._refresh_status()

    # -- 样式 ----------------------------------------------------------------

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg, card = "#0a0e1a", "#141c2e"
        fg, muted, accent = "#e8ecf4", "#9aa7ba", "#22d3ee"
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, fieldbackground=card)
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("Muted.TLabel", background=bg, foreground=muted, font=("Microsoft YaHei UI", 9))
        style.configure("TButton", background=card, foreground=fg, padding=6)
        style.map("TButton", background=[("active", "#1d2942")])
        style.configure("Accent.TButton", background=accent, foreground="#0a0e1a")
        style.map("Accent.TButton", background=[("active", "#67e8f9")])
        style.configure("TCombobox", fieldbackground=card, background=card, foreground=fg)
        self.root.option_add("*TCombobox*Listbox.background", card)
        self.root.option_add("*TCombobox*Listbox.foreground", fg)

    # -- 界面 ------------------------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")

        ttk.Button(top, text="📂 工作区", command=self._pick_workdir).pack(side="left")
        self.lbl_dir = ttk.Label(top, text=self.agent.workdir, style="Muted.TLabel")
        self.lbl_dir.pack(side="left", padx=(6, 14))
        self.lbl_dir.bind("<Button-1>", lambda e: self._pick_workdir())

        ttk.Label(top, text="提供商").pack(side="left", padx=(0, 4))
        self.cmb_provider = ttk.Combobox(top, width=10, state="readonly",
                                         values=list(PROVIDERS))
        self.cmb_provider.set(self.agent.config.provider)
        self.cmb_provider.pack(side="left", padx=(0, 10))
        self.cmb_provider.bind("<<ComboboxSelected>>", lambda e: self._on_provider_change())

        ttk.Label(top, text="模型").pack(side="left", padx=(0, 4))
        self.cmb_model = ttk.Combobox(top, width=22)
        self._sync_model_list()
        self.cmb_model.pack(side="left", padx=(0, 10))
        self.cmb_model.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        self.cmb_model.bind("<Return>", lambda e: self._on_model_change())

        ttk.Label(top, text="思考").pack(side="left", padx=(0, 4))
        self.cmb_think = ttk.Combobox(top, width=8, state="readonly",
                                      values=THINKING_LEVELS)
        self.cmb_think.set(self.agent.config.thinking)
        self.cmb_think.pack(side="left", padx=(0, 12))
        self.cmb_think.bind("<<ComboboxSelected>>", lambda e: self._on_thinking_change())

        self.lbl_status = ttk.Label(top, text="", style="Muted.TLabel")
        self.lbl_status.pack(side="right")

        self.txt = tk.Text(self.root, wrap="word", bd=0, padx=14, pady=12,
                           bg="#0a0e1a", fg="#e8ecf4", insertbackground="#e8ecf4",
                           font=("Microsoft YaHei UI", 10), state="disabled", cursor="arrow")
        self.txt.pack(fill="both", expand=True, padx=10, pady=(4, 0))
        for tag, conf in [
            ("system", {"foreground": "#64748b", "font": ("Microsoft YaHei UI", 9)}),
            ("user", {"foreground": "#22d3ee", "font": ("Microsoft YaHei UI", 10, "bold")}),
            ("assistant", {"foreground": "#e8ecf4", "font": ("Microsoft YaHei UI", 10)}),
            ("tool", {"foreground": "#facc15", "font": ("Consolas", 9)}),
            ("error", {"foreground": "#f87171"}),
        ]:
            self.txt.tag_configure(tag, **conf)

        bottom = ttk.Frame(self.root, padding=(10, 8))
        bottom.pack(fill="x")
        self.input = tk.Text(bottom, height=3, wrap="word", bd=0,
                             bg="#141c2e", fg="#e8ecf4", insertbackground="#e8ecf4",
                             font=("Microsoft YaHei UI", 10))
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)  # 换行
        self.btn_send = ttk.Button(bottom, text="发送 ⏎", style="Accent.TButton",
                                   command=self._send, state="disabled")
        self.btn_send.pack(side="left", fill="y", padx=(8, 0))
        if not self.agent.online:
            self._log_system(
                "未检测到 API 密钥（offline 模式）：界面与工具可用，但无法对话。\n"
                "设置提供商对应的环境变量（如 DEEPSEEK_API_KEY）或通用的 RICARDO_API_KEY 后重启。"
            )
        self.input.focus_set()

    # -- 日志 ------------------------------------------------------------------

    def _append(self, text: str, tag: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", text + "\n\n", tag)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _log_system(self, msg: str) -> None:
        self._append(msg, "system")

    def _refresh_status(self) -> None:
        c = self.agent.config
        key = "✓ 密钥" if c.api_key else "✗ 无密钥"
        self.lbl_status.configure(text=f"{c.provider} · {c.model} · 思考:{c.thinking} · {key}")
        self.lbl_dir.configure(text=self.agent.workdir)

    # -- 配置热切换 ---------------------------------------------------------------

    def _sync_model_list(self) -> None:
        self.cmb_model["values"] = self.agent.config.models()
        self.cmb_model.set(self.agent.config.model)

    def _on_provider_change(self) -> None:
        msg = self.agent.config.set_provider(self.cmb_provider.get())
        self._sync_model_list()
        self._log_system(msg)
        self._refresh_status()

    def _on_model_change(self) -> None:
        msg = self.agent.config.set_model(self.cmb_model.get().strip())
        self._log_system(msg)
        self._refresh_status()

    def _on_thinking_change(self) -> None:
        msg = self.agent.config.set_thinking(self.cmb_think.get())
        self._log_system(msg)
        self._refresh_status()

    def _pick_workdir(self) -> None:
        d = filedialog.askdirectory(title="选择工作文件夹", initialdir=self.agent.workdir)
        if d:
            self._log_system(self.agent.set_workdir(d))
            self._refresh_status()

    # -- 发送与后台线程 -------------------------------------------------------------

    def _on_enter(self, event) -> str:
        if not (event.state & 0x0001):  # 无 Shift 时回车发送
            self._send()
            return "break"
        return None

    def _send(self) -> None:
        text = self.input.get("1.0", "end").strip()
        if not text or self.btn_send["state"] == "disabled":
            return
        self.input.delete("1.0", "end")
        self._append("你: " + text, "user")
        if text.startswith("/"):
            from .cli import _handle_slash
            self._append(_handle_slash(text, self.agent) or "", "system")
            return
        self.btn_send.configure(state="disabled")
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _worker(self, text: str) -> None:
        try:
            reply = self.agent.chat(text)
            self._queue.put(("assistant", reply))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"ERROR: {exc}"))

    def _poll_queue(self) -> None:
        try:
            while True:
                tag, msg = self._queue.get_nowait()
                self._append(("Ricardo: " if tag == "assistant" else "") + msg, tag)
        except queue.Empty:
            pass
        self.btn_send.configure(state="normal")
        self._refresh_status()
        self.root.after(150, self._poll_queue)

    def run(self) -> None:
        self.root.mainloop()


def launch(workdir: str | None = None) -> None:
    try:
        ChatWindow(workdir).run()
    except tk.TclError as exc:
        raise SystemExit(f"无法启动图形界面（无显示环境?）: {exc}")
