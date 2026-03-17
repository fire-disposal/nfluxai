#!/usr/bin/env python3
"""
护理教材 AI 问答系统 - Streamlit 界面
支持完整的引用溯源和原文来源列出
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import NursingRetriever, create_prompt


# 页面配置
st.set_page_config(
    page_title="护理教材 AI 助手",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 10px 15px;
    }
    .citation-box {
        background-color: #f0f4f8;
        border-left: 4px solid #3182ce;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 6px 6px 0;
    }
    .reference-item {
        background-color: #e8f4fd;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .source-detail {
        background-color: #f8f9fa;
        padding: 10px;
        margin: 8px 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever():
    """缓存加载检索器"""
    with st.spinner("🔄 正在加载检索引擎..."):
        retriever = NursingRetriever()
        retriever.initialize()
    return retriever


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("🩺 护理教材 AI")
        st.markdown("---")
        
        # 教材过滤
        st.subheader("📖 教材范围")
        textbook_filter = st.radio(
            "选择教材:",
            options=["全部", "内科护理学", "外科护理学", "新编护理学基础"],
            index=0,
            help="限制检索范围到特定教材"
        )
        
        # 高级选项
        with st.expander("⚙️ 高级设置"):
            top_k = st.slider(
                "引用数量",
                min_value=3,
                max_value=10,
                value=5,
                help="每次回答引用的资料数量"
            )
            show_sources = st.checkbox(
                "显示详细来源",
                value=True,
                help="在回答后显示完整的引用来源"
            )
            show_content = st.checkbox(
                "显示原文内容",
                value=False,
                help="在来源中显示原文片段"
            )
        
        st.markdown("---")
        
        # 系统状态
        st.subheader("📊 系统状态")
        try:
            retriever = load_retriever()
            if retriever.vectorstore:
                collection = retriever.vectorstore._collection
                count = collection.count()
                st.metric("向量数量", f"{count:,}")
                st.success("✅ 向量库已加载")
        except Exception as e:
            st.error(f"❌ 加载失败：{e}")
        
        # 使用说明
        st.markdown("---")
        st.markdown("""
        ### 💡 使用提示
        
        **示例问题:**
        - 呼吸系统疾病病人的护理评估要点
        - 护理程序的基本步骤有哪些
        - 慢性阻塞性肺疾病的护理措施
        - 手术前后病人的护理注意事项
        
        **功能说明:**
        - AI 基于三本护理教材回答
        - 回答中标注 [1][2] 等引用
        - 可在下方查看完整来源信息
        - 支持按教材筛选
        """)
        
        # 清空对话
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # 转换过滤器
    if textbook_filter == "全部":
        textbook_filter = None
    
    return textbook_filter, top_k, show_sources, show_content


def render_sources_panel(citations: List[Dict], retriever: NursingRetriever, show_content: bool = False):
    """渲染来源信息面板"""
    if not citations:
        return
    
    st.markdown("### 📑 参考来源")
    
    for cite in citations:
        full_source = cite.get("full_source", {})
        meta = cite.get("metadata", {})
        
        with st.expander(f"[{cite['index']}] {cite['source']} (相似度：{cite['score']:.4f})"):
            # 来源详情
            st.markdown("**来源信息:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"- **教材**: {full_source.get('textbook', 'N/A')}")
                st.markdown(f"- **章节**: {full_source.get('chapter_title', 'N/A')}")
            with col2:
                st.markdown(f"- **文件**: {full_source.get('filename', 'N/A')}")
                st.markdown(f"- **行号**: {full_source.get('line_start', 0)}-{full_source.get('line_end', 0)}")
            
            st.markdown(f"**小节**: {full_source.get('section_header', 'N/A')} / {full_source.get('subsection_header', 'N/A')}")
            
            # 显示原文内容
            if show_content:
                st.markdown("**原文内容:**")
                content = cite.get("content", "")
                if content:
                    st.markdown(f"> {content[:500]}..." if len(content) > 500 else f"> {content}")
                
                # 尝试读取完整原文
                if full_source.get("filepath") and full_source.get("line_start") and full_source.get("line_end"):
                    with st.spinner("读取原文..."):
                        original = retriever.get_source_content(
                            textbook=full_source.get("textbook", ""),
                            filename=full_source.get("filename", ""),
                            line_start=full_source.get("line_start", 0),
                            line_end=full_source.get("line_end", 0),
                        )
                        if original:
                            st.markdown("**完整原文片段:**")
                            st.text(original[:1000] + "..." if len(original) > 1000 else original)


def render_chat_interface():
    """渲染聊天界面"""
    st.title("🩺 护理教材 AI 问答系统")
    st.markdown("**知识来源**: 内科护理学 | 外科护理学 | 新编护理学基础")
    st.markdown("---")
    
    # 获取配置
    textbook_filter, top_k, show_sources, show_content = render_sidebar()
    
    # 加载检索器
    retriever = load_retriever()
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 显示上次的来源信息
    if st.session_state.get("last_citations") and show_sources:
        render_sources_panel(st.session_state.last_citations, retriever, show_content)
        st.markdown("---")
    
    # 聊天输入
    prompt = st.chat_input("请输入护理学相关问题...")
    
    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在检索教材资料..."):
                # 检索
                context, citations = retriever.get_context_for_llm(
                    query=prompt,
                    top_k=top_k,
                    textbook_filter=textbook_filter,
                )
                
                # 保存引用供后续显示
                st.session_state.last_citations = citations
                
                # 显示引用预览
                if citations:
                    st.markdown("📚 **参考资料:**")
                    cite_cols = st.columns(min(len(citations), 3))
                    for i, cite in enumerate(citations):
                        with cite_cols[i % len(cite_cols)]:
                            st.markdown(f"<div class='reference-item'>[{cite['index']}] {cite['source']}</div>", 
                                      unsafe_allow_html=True)
                    st.markdown("---")
                
                # 调用 LLM 生成回答
                full_prompt = create_prompt(prompt, context)
                response = call_llm(full_prompt, prompt, citations)
                
                st.markdown(response)
            
            # 保存助手消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })
        
        # 显示来源面板
        if show_sources and citations:
            render_sources_panel(citations, retriever, show_content)
        
        st.rerun()


def call_llm(prompt: str, query: str, citations: List[Dict]) -> str:
    """
    调用 LLM 生成回答
    """
    from llm import generate_response
    
    # 使用 llm 模块生成回答
    context = prompt  # prompt 已经包含完整的上下文
    return generate_response(query, context, citations)


def main():
    """主函数"""
    init_session_state()
    render_chat_interface()


if __name__ == "__main__":
    main()
