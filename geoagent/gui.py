"""Ricardo Agent 桌面版（Tkinter，零额外依赖）——现代聊天界面。

布局对齐主流 AI 客户端（ZCode / ChatGPT 风格）：
┌─────────┬──────────────────────────────┐
│ 侧边栏   │  顶栏（工作区/提供商/模型/思考）  │
│ logo     ├──────────────────────────────┤
│ 新对话   │  聊天区（用户右/助手左 气泡）     │
│ 会话列表 │                              │
│ 状态     │  输入区（多行输入 + 发送按钮）    │
└─────────┴──────────────────────────────┘
与 CLI 共用同一套核心（GeoAgent + AgentConfig）。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import GeoAgent
from .config import PERMISSION_LABELS, PERMISSION_MODES, PROVIDERS, THINKING_LEVELS
from .observability import gui_crash_hook, setup_logging

# -- 配色（与官网一致的深色科技风） -------------------------------------------
BG = "#0a0e1a"  # 主背景
SIDEBAR = "#0d1424"  # 侧边栏
CARD = "#141c2e"  # 卡片/输入框
CARD_HI = "#1a2540"  # 卡片悬停
BUBBLE_USER = "#155e75"  # 用户气泡（青色暗调）
BUBBLE_AI = "#161f33"  # 助手气泡
BORDER = "#232f4b"
TEXT = "#e8ecf4"
MUTED = "#8b99af"
ACCENT = "#22d3ee"
GREEN = "#34d399"
RED = "#f87171"
YELLOW = "#facc15"
FONT = ("Microsoft YaHei UI", 10)
FONT_S = ("Microsoft YaHei UI", 9)
FONT_XS = ("Microsoft YaHei UI", 8)
FONT_MONO = ("Consolas", 9)
FONT_LOGO = ("Segoe UI", 15, "bold")


def _asset(name: str) -> str | None:
    """兼容源码运行与 PyInstaller 打包的资源定位。"""
    base = getattr(sys, "_MEIPASS", None)
    for root in (base, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if root:
            p = os.path.join(root, "assets", name)
            if os.path.exists(p):
                return p
    return None


class ChatWindow:
    def __init__(self, workdir: str | None = None) -> None:
        self.agent = GeoAgent(workdir=workdir)
        self._log_path = setup_logging()
        sys.excepthook = gui_crash_hook
        self.root = tk.Tk()
        self.root.report_callback_exception = lambda et, ev, tb: gui_crash_hook(et, ev, tb)
        self.root.title("Ricardo Agent")
        self.root.geometry("1180x760")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)
        icon = _asset("ricardo.ico")
        if icon:
            try:
                self.root.iconbitmap(icon)
            except tk.TclError:
                pass
        self._queue: queue.Queue = queue.Queue()
        self._live_open = False
        self._reply_started = False
        self._build_ui()
        self.root.after(120, self._poll_queue)
        self._sys("◈ Ricardo Agent — 地球物理 × 机器学习个人研究助手")
        self._sys(f"工作区: {self.agent.workdir} ｜ 输入 /help 查看命令")
        if not self.agent.online:
            self._sys(
                "未检测到 API 密钥（离线简单模式）：设置 DEEPSEEK_API_KEY 等环境变量后重启即可完整对话。"
            )
        self._refresh_sessions()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ---------- 侧边栏 ----------
        side = tk.Frame(self.root, width=224, bg=SIDEBAR)
        side.grid(row=0, column=0, sticky="nsw")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)

        tk.Label(side, text="◈ Ricardo", font=FONT_LOGO, bg=SIDEBAR, fg=ACCENT).pack(
            anchor="w", padx=18, pady=(18, 0)
        )
        tk.Label(side, text="Agent", font=FONT_LOGO, bg=SIDEBAR, fg=TEXT).pack(anchor="w", padx=18)

        def side_btn(text, cmd):
            b = tk.Label(
                side, text=text, font=FONT, bg=SIDEBAR, fg=TEXT, cursor="hand2", padx=18, pady=8, anchor="w"
            )
            b.bind("<Button-1>", lambda e: cmd())
            b.bind("<Enter>", lambda e: b.configure(bg=CARD))
            b.bind("<Leave>", lambda e: b.configure(bg=SIDEBAR))
            b.pack(fill="x", pady=2)
            return b

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", pady=10)
        side_btn("＋ 新对话", self._new_chat)
        side_btn("📂 选择工作区", self._pick_workdir)
        side_btn("💾 保存会话", lambda: self._slash(f"/save {self._session_name()}"))
        side_btn("🚀 运行评测 bench", self._run_bench)

        tk.Label(side, text="会话", font=FONT_S, bg=SIDEBAR, fg=MUTED, anchor="w").pack(
            fill="x", padx=18, pady=(14, 2)
        )
        self.sess_box = tk.Frame(side, bg=SIDEBAR)
        self.sess_box.pack(fill="x")

        # 侧边栏底部状态
        foot = tk.Frame(side, bg=SIDEBAR)
        foot.pack(side="bottom", fill="x", padx=14, pady=12)
        self.lbl_dot = tk.Label(foot, text="", font=FONT_S, bg=SIDEBAR, fg=GREEN, anchor="w")
        self.lbl_dot.pack(fill="x")
        self.lbl_prov = tk.Label(
            foot, text="", font=FONT_XS, bg=SIDEBAR, fg=MUTED, anchor="w", justify="left", wraplength=190
        )
        self.lbl_prov.pack(fill="x")

        # ---------- 右侧主区（ZCode 风格：聊天区 / 输入框 / 底部状态栏） ----------
        main = tk.Frame(self.root, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # 聊天区（可滚动画布 + 消息气泡容器）
        wrap = tk.Frame(main, bg=BG)
        wrap.grid(row=0, column=0, sticky="nsew", padx=16, pady=(12, 0))
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(wrap, bg=BG, bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sb = tk.Scrollbar(wrap, command=self.canvas.yview, width=12)
        sb.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=sb.set)
        self.msgs = tk.Frame(self.canvas, bg=BG)
        self._win = self.canvas.create_window((0, 0), window=self.msgs, anchor="nw")
        self.msgs.columnconfigure(0, weight=1)
        self._tail = tk.Frame(self.msgs, bg=BG)
        self._tail.pack(side="bottom", fill="x")
        self.msgs.bind("<Configure>", self._on_msgs_configure)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        for seq in ("<MouseWheel>",):
            self.canvas.bind_all(seq, self._on_wheel)

        # 输入区（发送按钮悬浮右下角）
        comp = tk.Frame(main, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        comp.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 6))
        self.input = tk.Text(
            comp,
            height=3,
            wrap="word",
            bd=0,
            bg=CARD,
            fg=TEXT,
            insertbackground=ACCENT,
            font=FONT,
            padx=14,
            pady=10,
        )
        self.input.pack(fill="both", expand=True, padx=(4, 0), pady=(4, 0))
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)
        self.btn_send = tk.Label(
            comp,
            text="发送 ⏎",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg=ACCENT,
            fg="#0a0e1a",
            padx=18,
            pady=8,
            cursor="hand2",
        )
        self.btn_send.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor="se")
        self.input.focus_set()

        # 底部状态栏：工作区 / 设置 / 模型 / 思考 / 权限 / 密钥 / 用量 全部集中于此
        status = tk.Frame(main, bg=SIDEBAR, highlightbackground=BORDER, highlightthickness=1)
        status.grid(row=2, column=0, sticky="ew")

        def sep():
            tk.Label(status, text="│", font=FONT_XS, bg=SIDEBAR, fg=BORDER).pack(side="left")

        def chip(text, fg=MUTED, cmd=None):
            c = tk.Label(
                status,
                text=text,
                font=FONT_XS,
                bg=SIDEBAR,
                fg=fg,
                padx=10,
                pady=7,
                cursor="hand2" if cmd else "arrow",
            )
            c.pack(side="left")
            if cmd:
                c.bind("<Button-1>", lambda e: cmd())
                c.bind("<Enter>", lambda e: c.configure(fg=TEXT))
                c.bind("<Leave>", lambda e: c.configure(fg=fg))
            return c

        wd = self.agent.workdir
        self.lbl_dir = chip("📂 " + (wd[:30] + "…" if len(wd) > 30 else wd), cmd=self._pick_workdir)
        chip("⚙ 设置", fg=ACCENT, cmd=self._open_settings)
        sep()
        tk.Label(status, text="模型", font=FONT_XS, bg=SIDEBAR, fg=MUTED).pack(side="left", padx=(8, 2))
        self.cmb_model = ttk.Combobox(
            status,
            width=20,
            state="readonly",
            font=FONT_XS,
            values=self.agent.config.models(),
        )
        self.cmb_model.set(self.agent.config.model)
        self.cmb_model.pack(side="left")
        self.cmb_model.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        self.cmb_model.bind("<Return>", lambda e: self._on_model_change())
        tk.Label(status, text="思考", font=FONT_XS, bg=SIDEBAR, fg=MUTED).pack(side="left", padx=(10, 2))
        self.cmb_think = ttk.Combobox(status, width=6, state="readonly", font=FONT_XS, values=THINKING_LEVELS)
        self.cmb_think.set(self.agent.config.thinking)
        self.cmb_think.pack(side="left")
        self.cmb_think.bind("<<ComboboxSelected>>", lambda e: self._on_thinking_change())
        tk.Label(status, text="权限", font=FONT_XS, bg=SIDEBAR, fg=MUTED).pack(side="left", padx=(10, 2))
        self.cmb_perm = ttk.Combobox(
            status, width=9, state="readonly", font=FONT_XS, values=list(PERMISSION_MODES)
        )
        self.cmb_perm.set(self.agent.config.permission_mode)
        self.cmb_perm.pack(side="left")
        self.cmb_perm.bind("<<ComboboxSelected>>", lambda e: self._on_perm_change())
        self.lbl_perm = tk.Label(status, text="", font=FONT_XS, bg=SIDEBAR, fg=MUTED)
        self.lbl_perm.pack(side="left", padx=(4, 0))
        self.lbl_key = tk.Label(
            status, text="🔑", font=FONT_XS, bg=SIDEBAR, fg=MUTED, padx=10, cursor="hand2"
        )
        self.lbl_key.pack(side="right")
        self.lbl_key.bind("<Button-1>", lambda e: self._open_settings())
        self.lbl_usage = tk.Label(status, text="", font=FONT_XS, bg=SIDEBAR, fg=MUTED, padx=8)
        self.lbl_usage.pack(side="right")
        self._refresh_status()

    # ------------------------------------------------------------ 气泡渲染

    def _on_msgs_configure(self, _e=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_wheel(self, e) -> None:
        self.canvas.yview_scroll(int(-e.delta / 120), "units")

    def _bubble(
        self, text: str, side: str, tag_bg: str, tag_fg: str, font=FONT, max_ratio: float = 0.78
    ) -> None:
        """在聊天区末尾追加一条消息气泡。"""
        row = tk.Frame(self.msgs, bg=BG)
        row.pack(fill="x", pady=4, before=self._tail)
        anchor = "e" if side == "user" else "w"
        inner = tk.Frame(row, bg=tag_bg, highlightbackground=BORDER, highlightthickness=1)
        inner.pack(anchor=anchor, padx=(80 if anchor == "e" else 8, 8 if anchor == "e" else 80))
        maxw = max(360, int(self.root.winfo_width() * max_ratio) - 60)
        lbl = tk.Label(
            inner, text=text, font=font, bg=tag_bg, fg=tag_fg, wraplength=maxw, justify="left", anchor="w"
        )
        lbl.pack(padx=14, pady=10)
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _user(self, text: str) -> None:
        self._bubble(text, "user", BUBBLE_USER, "#e0faff")

    def _ai(self, text: str) -> None:
        self._bubble(text, "ai", BUBBLE_AI, TEXT)

    def _tool(self, text: str) -> None:
        self._bubble("🔧 " + text, "ai", "#101826", YELLOW, font=FONT_MONO, max_ratio=0.9)

    def _sys(self, text: str) -> None:
        row = tk.Frame(self.msgs, bg=BG)
        row.pack(fill="x", pady=2, before=self._tail)
        tk.Label(row, text="— " + text + " —", font=FONT_XS, bg=BG, fg=MUTED).pack()
        self.canvas.yview_moveto(1.0)

    def _err(self, text: str) -> None:
        self._bubble(text, "ai", "#2a1215", RED)

    # ------------------------------------------------------------ 状态/配置

    def _refresh_status(self) -> None:
        c = self.agent.config
        ok = bool(c.api_key)
        self.lbl_dot.configure(text="● 在线" if ok else "○ 离线（简单模式）", fg=GREEN if ok else MUTED)
        self.lbl_prov.configure(text=f"{c.provider} · {c.model}")
        wd = self.agent.workdir
        self.lbl_dir.configure(text="📂 " + (wd[:30] + "…" if len(wd) > 30 else wd))
        self.lbl_perm.configure(text=PERMISSION_LABELS.get(c.permission_mode, c.permission_mode))
        self.cmb_perm.set(c.permission_mode)
        self.lbl_key.configure(text=f"🔑 {c.key_source()}", fg=GREEN if ok else MUTED)
        u = self.agent.usage
        if u["calls"]:
            self.lbl_usage.configure(text=f"{u['total_tokens']:,} tok")

    def _refresh_sessions(self) -> None:
        for w in self.sess_box.winfo_children():
            w.destroy()
        names = [s for s in self.agent.list_sessions().split(", ") if s and "没有" not in s]
        for n in names[:8]:
            lbl = tk.Label(
                self.sess_box,
                text="· " + n,
                font=FONT_S,
                bg=SIDEBAR,
                fg=MUTED,
                anchor="w",
                padx=18,
                cursor="hand2",
            )
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda e, n=n: self._slash(f"/load {n}"))
            lbl.bind("<Enter>", lambda e, lab=lbl: lab.configure(fg=ACCENT))
            lbl.bind("<Leave>", lambda e, lab=lbl: lab.configure(fg=MUTED))

    def _session_name(self) -> str:
        from datetime import datetime

        return "chat_" + datetime.now().strftime("%m%d_%H%M")

    def _on_provider_change(self) -> None:
        msg = self.agent.config.set_provider(self.cmb_provider.get())
        self.cmb_model["values"] = self.agent.config.models()
        self.cmb_model.set(self.agent.config.model)
        self._sys(msg)
        self._refresh_status()

        def _fetch():
            remote = self.agent.config.fetch_models()
            if remote:
                self.root.after(0, lambda: self.cmb_model.configure(values=remote[:50]))

        threading.Thread(target=_fetch, daemon=True).start()

    def _on_model_change(self) -> None:
        self._sys(self.agent.config.set_model(self.cmb_model.get().strip()))
        self._refresh_status()

    def _on_thinking_change(self) -> None:
        self._sys(self.agent.config.set_thinking(self.cmb_think.get()))
        self._refresh_status()

    def _sync_perm_combo(self) -> None:
        m = self.agent.config.permission_mode
        self.cmb_perm.set(f"{m} · {PERMISSION_LABELS[m]}")

    def _on_perm_change(self) -> None:
        sel = self.cmb_perm.get().split(" · ")[0]
        if sel == "ask":
            self._sys("谨慎模式：写入/执行/联网将在弹窗中逐次确认。")
        out = self.agent.set_permission_mode(sel, confirm=self._confirm_tool)
        self._sys(out)
        self._refresh_status()

    def _confirm_tool(self, tool_name: str, risk: str) -> bool:
        """谨慎模式的逐次确认回调：后台线程等待，主线程弹窗。"""
        import threading as _th

        box = {"yes": False}
        done = _th.Event()

        def _ask_main() -> None:
            box["yes"] = messagebox.askyesno(
                "权限确认",
                f"权限模式为「谨慎」，工具 {tool_name} 需要{risk}权限。\n\n允许本次调用吗？",
            )
            done.set()

        self.root.after(0, _ask_main)
        while not done.wait(timeout=0.1):
            pass
        return box["yes"]

    def _pick_workdir(self) -> None:
        d = filedialog.askdirectory(title="选择工作文件夹", initialdir=self.agent.workdir)
        if d:
            self._sys(self.agent.set_workdir(d))
            self._refresh_status()

    def _new_chat(self) -> None:
        for w in self.msgs.winfo_children():
            w.destroy()
        self.agent.history = []
        self._sys("新对话已开始（历史已清空）")

    def _run_bench(self) -> None:
        if not messagebox.askyesno("评测", "运行 7 个场景的行为评测？\n会真实调用 LLM，耗时约 1-2 分钟。"):
            return
        self._user("🚀 /bench")
        self.btn_send.configure(state="disabled")
        threading.Thread(target=self._bench_worker, daemon=True).start()

    def _bench_worker(self) -> None:
        from .bench import load_scenarios, run_benchmark

        try:
            report = run_benchmark(
                self.agent,
                load_scenarios(),
                on_progress=lambda m: self._queue.put(("ev", {"type": "tool_end", "result": m})),
            )
            self._queue.put(("raw", report))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"ERROR: {exc}"))

    # ------------------------------------------------------------ 发送

    def _slash(self, text: str) -> None:
        self._user(text)
        from .cli import _handle_slash

        out = _handle_slash(text, self.agent)
        if out:
            self._bubble(out, "ai", BUBBLE_AI, MUTED, font=FONT_S)
        self._refresh_sessions()
        self._refresh_status()

    def _on_enter(self, event) -> str:
        if not (event.state & 0x0001):
            self._send()
            return "break"
        return None

    def _send(self) -> None:
        text = self.input.get("1.0", "end").strip()
        if not text or str(self.btn_send["state"]) == "disabled":
            return
        self.input.delete("1.0", "end")
        self._user(text)
        if text.startswith("/"):
            self._slash(text)
            return
        if not self.agent.online:
            from .cli import _demo_reply

            self._ai("（离线简单模式）\n" + _demo_reply(text, self.agent))
            return
        self.btn_send.configure(state="disabled", bg="#0e7490")
        self._live_open = False
        self._reply_started = False
        self._live_widgets: list[tk.Widget] = []
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _worker(self, text: str) -> None:
        def on_event(ev: dict) -> None:
            self._queue.put(("ev", ev))

        try:
            self.agent.chat(text, on_event=on_event)
            self._queue.put(("done", self.agent.usage_report()))
        except Exception as exc:  # noqa: BLE001
            self._queue.put(("error", f"ERROR: {exc}"))

    def _handle_event(self, ev: dict) -> None:
        if ev["type"] == "delta":
            if not self._live_open:
                self._live_text = tk.Text(
                    self._tail,
                    wrap="word",
                    bd=0,
                    bg=BUBBLE_AI,
                    fg=TEXT,
                    font=FONT,
                    padx=14,
                    pady=10,
                    height=1,
                )
                inner = tk.Frame(self._tail, bg=BUBBLE_AI, highlightbackground=BORDER, highlightthickness=1)
                inner.pack(anchor="w", padx=(8, 80), pady=4)
                self._live_text.pack(in_=inner, padx=1, pady=1)
                self._live_text.bind("<Key>", lambda e: "break")
                self._live_open = True
                self._reply_started = True
            self._live_text.insert("end", ev["text"])
            self._live_text.configure(height=max(1, int(self._live_text.index("end-1c").split(".")[0])))
            self.canvas.yview_moveto(1.0)
        elif ev["type"] == "tool_start":
            import json as _json

            self._live_open = False
            args_s = _json.dumps(ev["args"], ensure_ascii=False)[:110]
            self._tool(f"{ev['name']}({args_s})")
        elif ev["type"] == "tool_end":
            r = ev["result"].replace("\n", " ")
            self._tool("↳ " + (r[:150] + ("…" if len(r) > 150 else "")))

    def _tool(self, text: str) -> None:
        inner = tk.Frame(self._tail, bg="#101826", highlightbackground=BORDER, highlightthickness=1)
        inner.pack(anchor="w", padx=(8, 80), pady=1)
        tk.Label(
            inner, text="🔧 " + text, font=FONT_MONO, bg="#101826", fg=YELLOW, wraplength=760, justify="left"
        ).pack(padx=12, pady=6)
        self.canvas.yview_moveto(1.0)

    def _flush_tail(self) -> None:
        """把流式区内容定格为正式气泡。"""
        text = ""
        if self._live_open:
            text = self._live_text.get("1.0", "end").strip()
        for w in self._tail.winfo_children():
            w.destroy()
        self._live_open = False
        if text:
            self._ai(text)

    def _poll_queue(self) -> None:
        try:
            while True:
                tag, msg = self._queue.get_nowait()
                if tag == "ev":
                    self._handle_event(msg)
                    continue
                if tag in ("done", "raw"):
                    if tag == "raw":
                        self._ai(msg)
                    elif not self._reply_started:
                        self._ai("（模型未返回文本）")
                    self._sys(msg if tag == "done" else "")
                    self._live_open = False
                    self._reply_started = False
                else:
                    self._live_open = False
                    self._reply_started = False
                    self._err(msg)
        except queue.Empty:
            pass
        self.btn_send.configure(state="normal", bg=ACCENT)
        self._refresh_status()
        self.root.after(120, self._poll_queue)

    def run(self) -> None:
        self.root.mainloop()


def launch(workdir: str | None = None) -> None:
    try:
        ChatWindow(workdir).run()
    except tk.TclError as exc:
        raise SystemExit(f"无法启动图形界面（无显示环境?）: {exc}") from exc


class SettingsDialog(tk.Toplevel):
    """设置窗口：厂商管理（内置+自定义）、密钥保存、连接测试。"""

    def __init__(self, parent, agent: GeoAgent, on_change) -> None:
        super().__init__(parent)
        self.agent = agent
        self.on_change = on_change  # 变更后刷新主界面
        self.title("设置 — 提供商与密钥")
        self.geometry("640x520")
        self.configure(bg=BG)
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="提供商", font=("Microsoft YaHei UI", 11, "bold"), bg=BG, fg=TEXT).pack(
            anchor="w", padx=16, pady=(14, 4)
        )
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x", padx=16)
        self.lst = tk.Listbox(row, height=8, bg=CARD, fg=TEXT, bd=0, highlightbackground=BORDER, font=FONT)
        self.lst.pack(side="left", fill="both", expand=True)
        self.lst.bind("<<ListboxSelect>>", lambda e: self._load_selected())
        ops = tk.Frame(row, bg=BG)
        ops.pack(side="left", fill="y", padx=(8, 0))
        for text in ["设为当前", "删除"]:
            tk.Label(ops, text=text, font=FONT_S, bg=CARD, fg=TEXT, padx=10, pady=6, cursor="hand2").pack(
                fill="x", pady=3
            )
        # 手动绑定点击
        for w in ops.winfo_children():
            w.bind("<Button-1>", lambda e, w=w: self._op(w["text"]))
        self._reload_list()

        form = tk.Frame(self, bg=BG)
        form.pack(fill="x", padx=16, pady=(10, 0))
        self.fields: dict[str, tk.Entry] = {}
        for i, (key, label, width) in enumerate(
            [
                ("name", "厂商名", 16),
                ("base_url", "API 地址", 44),
                ("key", "API 密钥", 44),
                ("default_model", "默认模型", 24),
            ]
        ):
            tk.Label(form, text=label, font=FONT_XS, bg=BG, fg=MUTED).grid(
                row=i, column=0, sticky="w", pady=3
            )
            e = tk.Entry(
                form,
                width=width,
                bg=CARD,
                fg=TEXT,
                bd=0,
                insertbackground=ACCENT,
                font=FONT_S,
                show="•" if key == "key" else "",
            )
            e.grid(row=i, column=1, sticky="w", padx=8, pady=3)
            self.fields[key] = e

        btns = tk.Frame(self, bg=BG)
        btns.pack(fill="x", padx=16, pady=12)
        for text, cmd in [
            ("💾 保存厂商", self._save),
            ("🔑 保存密钥", self._save_key),
            ("🧪 测试连接", self._test),
            ("✖ 关闭", self.destroy),
        ]:
            lbl = tk.Label(btns, text=text, font=FONT_S, bg=CARD, fg=ACCENT, padx=12, pady=6, cursor="hand2")
            lbl.pack(side="left", padx=(0, 8))
            lbl.bind("<Button-1>", lambda e, c=cmd: c())
        tk.Label(
            self,
            text="提示：密钥保存在本地 ~/.geoagent/config.json（明文），也可继续使用环境变量方式。",
            font=FONT_XS,
            bg=BG,
            fg=MUTED,
            wraplength=600,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 12))

    def _reload_list(self) -> None:
        cfg = self.agent.config
        self.lst.delete(0, "end")
        for n in cfg.provider_names():
            mark = "●" if cfg.provider == n else "○"
            custom = "（自定义）" if n in cfg.custom_providers else ""
            has_key = "🔑" if (cfg.saved_keys.get(n) or cfg.custom_providers.get(n, {}).get("key")) else ""
            self.lst.insert("end", f"{mark} {n} {custom}{has_key}")

    def _selected_name(self) -> str | None:
        sel = self.lst.curselection()
        if not sel:
            return None
        return self.agent.config.provider_names()[sel[0]]

    def _load_selected(self) -> None:
        name = self._selected_name()
        if not name:
            return
        cfg = self.agent.config
        preset = cfg.custom_providers.get(name) or PROVIDERS.get(name, {})
        self.fields["name"].delete(0, "end")
        self.fields["name"].insert(0, name)
        self.fields["base_url"].delete(0, "end")
        self.fields["base_url"].insert(0, preset.get("base_url", ""))
        self.fields["key"].delete(0, "end")
        self.fields["key"].insert(0, cfg.saved_keys.get(name) or preset.get("key", ""))
        self.fields["default_model"].delete(0, "end")
        self.fields["default_model"].insert(0, preset.get("default_model", ""))

    def _op(self, text: str) -> None:
        if text == "设为当前":
            name = self._selected_name()
            if name:
                self.agent.config.set_provider(name)
                self._reload_list()
                self.on_change()
        elif text == "删除":
            name = self._selected_name()
            if name:
                self._sysmsg(self.agent.config.delete_provider(name))
                self._reload_list()
                self.on_change()

    def _save(self) -> None:
        msg = self.agent.config.upsert_provider(
            self.fields["name"].get(),
            self.fields["base_url"].get(),
            self.fields["key"].get(),
            self.fields["default_model"].get(),
        )
        self._sysmsg(msg)
        self._reload_list()
        self.on_change()

    def _save_key(self) -> None:
        name = self.fields["name"].get().strip().lower()
        if not name:
            self._sysmsg("ERROR: 请填写厂商名")
            return
        self._sysmsg(self.agent.config.set_key(name, self.fields["key"].get()))
        self._reload_list()

    def _test(self) -> None:
        cfg = self.agent.config
        base = self.fields["base_url"].get().strip() or cfg.base_url
        key = self.fields["key"].get().strip() or (cfg.api_key or "")
        import requests

        try:
            r = requests.get(
                base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=8
            )
            n = len(r.json().get("data", []))
            self._sysmsg(f"✅ 连接成功，发现 {n} 个模型")
        except Exception as exc:  # noqa: BLE001
            self._sysmsg(f"❌ 连接失败: {exc}")

    def _sysmsg(self, msg: str) -> None:
        messagebox.showinfo("设置", msg)


def _open_settings(self) -> None:
    SettingsDialog(self.root, self.agent, on_change=self._refresh_status)


# 动态挂载方法
ChatWindow._open_settings = _open_settings
