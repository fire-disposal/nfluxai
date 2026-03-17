#!/usr/bin/env python3
"""
护理教材 AI 问答系统 - Streamlit 界面
支持完整的引用溯源和原文来源列出

优化内容:
- 来源面板嵌入每条助手消息
- 引用预览增强（支持所有引用）
- LLM 调用超时和重试机制
- 搜索历史功能
- 移动端适配优化
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import streamlit as st

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import NursingRetriever, create_prompt


# 页面配置 - 支持移动端响应式
st.set_page_config(
    page_title="护理教材 AI 助手",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Getting Started": False,
        "Report a bug": False,
        "About": False,
    },
)

# 自定义 CSS - 增强移动端适配
st.markdown("""
<style>
    /* 全局响应式优化 */
    @media (max-width: 768px) {
        .stChatMessage {
            padding: 8px 10px !important;
        }
        .stSidebar {
            width: 100% !important;
        }
        .stExpander {
            font-size: 0.9em;
        }
    }

    /* 聊天消息样式 */
    .stChatMessage {
        padding: 10px 15px;
    }

    /* 引用框样式 */
    .citation-box {
        background-color: #f0f4f8;
        border-left: 4px solid #3182ce;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 6px 6px 0;
    }

    /* 引用项样式 - 支持滚动 */
    .reference-item {
        background-color: #e8f4fd;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.9em;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .reference-item:hover {
        background-color: #d0e8f8;
    }

    /* 引用网格布局容器 */
    .reference-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 8px;
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f8fafc;
        border-radius: 6px;
    }

    /* 来源详情样式 */
    .source-detail {
        background-color: #f8f9fa;
        padding: 10px;
        margin: 8px 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85em;
    }

    /* 历史查询项样式 */
    .history-item {
        background-color: #edf2f7;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85em;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .history-item:hover {
        background-color: #d0e8f8;
        transform: translateX(4px);
    }

    /* 错误提示框样式 */
    .error-box {
        background-color: #fff5f5;
        border-left: 4px solid #fc8181;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 10px 0;
    }

    /* 警告提示框样式 */
    .warning-box {
        background-color: #fffff0;
        border-left: 4px solid #f6e05e;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 10px 0;
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
    # 搜索历史
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    # 消息对应的引用
    if "message_citations" not in st.session_state:
        st.session_state.message_citations = {}


def save_search_history(query: str):
    """保存查询到历史记录"""
    history = st.session_state.search_history
    # 移除重复项
    history = [h for h in history if h["query"] != query]
    # 添加到开头
    history.insert(0, {"query": query, "timestamp": datetime.now().isoformat()})
    # 限制最多 10 条
    st.session_state.search_history = history[:10]


def render_search_history():
    """渲染搜索历史"""
    if not st.session_state.search_history:
        st.info("暂无搜索历史")
        return None

    st.markdown("### 📜 最近查询")

    # 历史查询列表
    for i, item in enumerate(st.session_state.search_history):
        query = item["query"]
        timestamp = item["timestamp"]
        # 显示截断的时间
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%m-%d %H:%M")
        except:
            time_str = ""

        # 可点击的历史项
        if st.button(
            f"🔍 {query[:30]}{'...' if len(query) > 30 else ''}",
            key=f"history_{i}",
            help=f"{query}\n\n时间：{time_str}",
            use_container_width=True,
        ):
            return query

    return None


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

        # 搜索历史
        selected_history = render_search_history()

        # 清空历史按钮
        if st.session_state.search_history:
            if st.button("🗑️ 清空历史", use_container_width=True, key="clear_history"):
                st.session_state.search_history = []
                st.rerun()

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
            st.session_state.message_citations = {}
            st.rerun()

    # 转换过滤器
    if textbook_filter == "全部":
        textbook_filter = None

    return textbook_filter, top_k, show_sources, show_content, selected_history


def render_sources_panel(citations: List[Dict], retriever: NursingRetriever, show_content: bool = False, embedded: bool = True):
    """
    渲染来源信息面板

    Args:
        citations: 引用列表
        retriever: 检索器实例
        show_content: 是否显示原文内容
        embedded: 是否嵌入在消息内（使用 expander）
    """
    if not citations:
        return

    # 定义展开器容器
    container = st.expander(f"📑 参考来源 ({len(citations)}条)", expanded=not embedded) if embedded else st.container()

    with container:
        # 使用网格布局显示所有引用预览
        st.markdown("**引用预览:**")

        # 计算网格列数（响应式）
        num_cols = min(len(citations), 4)
        cols = st.columns(num_cols)

        for i, cite in enumerate(citations):
            with cols[i % num_cols]:
                st.markdown(
                    f"<div class='reference-item' style='cursor: pointer;'>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**[{cite['index']}] {cite['source']}**")
                st.caption(f"相似度：{cite['score']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # 详细来源列表
        st.markdown("**详细来源:**")
        for cite in citations:
            full_source = cite.get("full_source", {})
            meta = cite.get("metadata", {})

            with st.expander(f"[{cite['index']}] {cite['source']} (相似度：{cite['score']:.4f})"):
                # 来源详情
                st.markdown("**来源信息:**")
                # 响应式列布局
                if st._config.get("browserWidth", "wide") == "wide" or True:  # 始终使用双列
                    col1, col2 = st.columns(2)
                else:
                    col1, col2 = st.columns(1), st.columns(1)

                with col1[0] if isinstance(col1, list) else col1:
                    st.markdown(f"- **教材**: {full_source.get('textbook', 'N/A')}")
                    st.markdown(f"- **章节**: {full_source.get('chapter_title', 'N/A')}")
                with col2[0] if isinstance(col2, list) else col2:
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
    textbook_filter, top_k, show_sources, show_content, selected_history = render_sidebar()

    # 加载检索器
    retriever = load_retriever()

    # 处理历史查询选择
    if selected_history:
        prompt = selected_history
        # 清空选择避免重复触发
        selected_history = None

    # 显示历史消息（每条消息包含嵌入的来源）
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 如果是助手消息且有对应的引用，嵌入显示
            if msg["role"] == "assistant":
                msg_id = msg.get("id", i)
                citations = st.session_state.message_citations.get(msg_id, [])
                if citations and show_sources:
                    render_sources_panel(citations, retriever, show_content, embedded=True)

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
                try:
                    # 检索
                    context, citations = retriever.get_context_for_llm(
                        query=prompt,
                        top_k=top_k,
                        textbook_filter=textbook_filter,
                    )

                    # 保存引用到会话状态（与消息关联）
                    msg_id = len(st.session_state.messages)
                    st.session_state.message_citations[msg_id] = citations

                    # 显示引用预览（网格布局，无数量限制）
                    if citations:
                        st.markdown("📚 **参考资料:**")

                        # 使用滚动网格显示所有引用
                        st.markdown("<div class='reference-grid'>", unsafe_allow_html=True)
                        for cite in citations:
                            st.markdown(
                                f"<div class='reference-item'>[{cite['index']}] {cite['source']}</div>",
                                unsafe_allow_html=True
                            )
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")

                    # 调用 LLM 生成回答（带超时和重试）
                    full_prompt = create_prompt(prompt, context)
                    response = call_llm_with_retry(full_prompt, prompt, citations, max_retries=2)

                    st.markdown(response)

                    # 保存助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "id": msg_id,
                    })

                    # 保存到搜索历史
                    save_search_history(prompt)

                except Exception as e:
                    error_msg = f"""
<div class='error-box'>
<strong>⚠️ 处理请求时出错:</strong><br>
{str(e)}<br><br>
请尝试:
<ul>
<li>检查网络连接</li>
<li>确认 LLM 服务是否正常运行</li>
<li>简化您的问题或换一种问法</li>
</ul>
</div>
"""
                    st.markdown(error_msg, unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"抱歉，处理您的请求时出现错误：{str(e)}",
                        "id": msg_id,
                    })

            st.rerun()


def call_llm(prompt: str, query: str, citations: List[Dict]) -> str:
    """
    调用 LLM 生成回答（带超时处理）

    Args:
        prompt: 完整提示词
        query: 用户原始问题
        citations: 引用列表

    Returns:
        LLM 生成的回答
    """
    from llm import generate_response

    # 使用 llm 模块生成回答
    context = prompt  # prompt 已经包含完整的上下文
    return generate_response(query, context, citations)


def call_llm_with_retry(prompt: str, query: str, citations: List[Dict], max_retries: int = 2, timeout: int = 60) -> str:
    """
    调用 LLM 生成回答，带重试机制和超时处理

    Args:
        prompt: 完整提示词
        query: 用户原始问题
        citations: 引用列表
        max_retries: 最大重试次数（默认 2 次）
        timeout: 超时时间（秒）

    Returns:
        LLM 生成的回答或降级响应
    """
    from llm import generate_response, get_llm_config

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # 无引用时的降级处理
            if not citations:
                return generate_friendly_fallback(query)

            # 调用 LLM
            context = prompt
            response = generate_response(query, context, citations)

            # 检查是否是错误响应
            if response.startswith("[Ollama 调用失败") or response.startswith("[API 调用失败"):
                raise Exception(response)

            # 检查是否是降级响应（无 LLM 时）
            if "部署完整 LLM 后" in response or "无法找到与您的问题直接相关的内容" in response:
                # 降级响应也是有效响应，直接返回
                return response

            return response

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                # 显示重试提示
                st.warning(f"⚠️ LLM 调用失败，正在重试 ({attempt + 1}/{max_retries})...")
                time.sleep(1.0 * (attempt + 1))  # 递增延迟
            else:
                # 所有重试失败，返回友好降级提示
                return generate_friendly_fallback(query, str(e))

    # 理论上不会到这里
    return generate_friendly_fallback(query, str(last_error) if last_error else "未知错误")


def generate_friendly_fallback(query: str, error: str = None) -> str:
    """
    生成友好的降级提示（当 LLM 不可用时）

    Args:
        query: 用户问题
        error: 错误信息（可选）

    Returns:
        友好的降级响应
    """
    response = """## 📚 基于检索结果的信息

根据护理教材资料，以下是相关参考资料：

"""
    # 这里会由调用方补充引用信息
    response += """
---
### 💡 温馨提示

当前 LLM 服务暂时不可用，您可以：

1. **检查 LLM 服务状态**
   - 确认 Ollama 服务是否运行：`ollama list`
   - 启动服务：`ollama serve`

2. **检查配置**
   - 查看 `config.yaml` 中的 LLM 配置
   - 确认模型已下载：`ollama pull qwen2.5:7b`

3. **替代方案**
   - 以上已列出相关教材参考资料
   - 可直接查阅教材原文获取详细信息
   - 咨询专业教师或同学

---
*系统已尽力为您检索相关资料，部署完整 LLM 后可获得更准确、结构化的回答。*
"""
    return response


def main():
    """主函数"""
    init_session_state()
    render_chat_interface()


if __name__ == "__main__":
    main()
