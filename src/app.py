#!/usr/bin/env python3
"""护理教材 AI 助手"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from retriever import Retriever
from llm import generate_response
from ingest import ingest_all
from indexer import build_index

INDEX_DIR = PROJECT_ROOT / "data" / "index"

EXAMPLES = [
    "肺炎患者的护理措施有哪些？",
    "糖尿病病人的饮食护理要点",
    "高血压患者的健康教育内容",
    "手术前病人的护理评估包括哪些？",
    "术后如何预防切口感染？",
]

st.set_page_config(page_title="护理教材 AI", page_icon="", layout="wide")


@st.cache_resource
def ensure_index():
    if not (INDEX_DIR / "invoices.pkl").exists():
        with st.spinner("首次运行，正在构建索引..."):
            docs = ingest_all()
            build_index(docs)
        return len(docs)
    return 0


@st.cache_resource
def get_retriever():
    return Retriever()


def main():
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    retriever = get_retriever()

    # 侧边栏
    st.sidebar.title("护理教材 AI")
    textbook_val = st.sidebar.selectbox(
        "教材范围",
        ["全部", "内科护理学", "外科护理学", "新编护理学基础"],
        label_visibility="collapsed",
    )
    top_k = st.sidebar.slider("引用条数", 3, 10, 5)

    has_index = (INDEX_DIR / "invoices.pkl").exists()
    st.sidebar.metric("索引状态", "就绪" if has_index else "未构建")
    if st.sidebar.button("重建索引", use_container_width=True):
        with st.spinner("重建中..."):
            get_retriever.clear()
            ensure_index.clear()
            build_index(ingest_all())
        st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("清空对话", use_container_width=True):
        st.session_state.msgs = []
        st.rerun()

    textbook = None if textbook_val == "全部" else textbook_val

    st.title("护理教材 AI 助手")

    # 快捷提问
    cols = st.columns(len(EXAMPLES))
    quick = None
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            quick = ex

    # 历史
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                with st.expander(f"参考来源 ({len(m['citations'])}条)", expanded=False):
                    for c in m["citations"]:
                        st.markdown(
                            f"**[{c['index']}] {c['textbook']} / {c['chapter']} / {c['title']}**"
                        )
                        st.text(c["content"][:200] + "...")

    q = st.chat_input("输入护理学问题...") or quick
    if not q:
        return

    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            try:
                ctx, cites = retriever.get_context(q, top_k, textbook)
                if not cites:
                    resp = "未找到相关资料，请尝试更换关键词或放宽教材范围。"
                else:
                    srcs = " | ".join([f"[{c['index']}] {c['chapter']}" for c in cites])
                    st.caption(f"引用: {srcs}")
                    resp = generate_response(q, ctx)

                st.markdown(resp)
                st.session_state.msgs.append({
                    "role": "assistant",
                    "content": resp,
                    "citations": cites,
                })
            except Exception as e:
                st.error(f"出错了: {e}")
                st.session_state.msgs.append({"role": "assistant", "content": str(e)})


if __name__ == "__main__":
    ensure_index()
    main()
