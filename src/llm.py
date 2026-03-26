#!/usr/bin/env python3
"""LLM 集成模块（统一 API 服务模式）。"""

from typing import Dict, Any, List

from model_clients import get_llm_service_config, call_chat_completion


def get_llm_config() -> Dict[str, Any]:
    """获取 LLM API 配置（兼容新旧配置结构）。"""
    return get_llm_service_config()


def generate_response(prompt: str, context: str, citations: List[Dict]) -> str:
    """生成 LLM 回答。"""
    if not citations:
        return """抱歉，根据现有教材资料，我无法找到与您问题直接相关的内容。

请尝试：
1. 换一种问法
2. 使用更专业的医学术语
3. 咨询专业教师或参考教材原文"""

    config = get_llm_config()
    full_prompt = f"""你是一名专业的护理学助教，基于提供的参考资料回答用户问题。

## 参考资料
{context}

## 回答要求
1. 仅基于上述资料回答，不要编造信息
2. 在引用处标注 [1], [2] 等编号
3. 如资料不足，请说明"根据现有资料无法完整回答"
4. 使用专业但易懂的中文
5. 结构清晰，分点叙述

## 用户问题
{prompt}

## 回答
"""
    return call_chat_completion(full_prompt, config)


def _fallback_response(prompt: str, context: str, citations: List[Dict]) -> str:
    response = "根据护理教材资料，以下是相关信息：\n\n"
    for i, cite in enumerate(citations[:3], 1):
        title = cite.get("metadata", {}).get("title", cite.get("metadata", {}).get("header", "相关内容"))
        source = cite.get("source", "")
        response += f"**{source}** [{i}]\n{title}\n\n"
    response += "---\n*注：部署完整 LLM API 服务后可获得更准确的回答。*"
    return response
