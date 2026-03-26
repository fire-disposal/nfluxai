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
        raise ValueError("未检索到可引用的教材片段，无法生成可靠回答")

    config = get_llm_config()
    context_limit = int(config.get("context_max_chars", 12000))
    safe_context = context[:context_limit]
    citation_text = ", ".join(f"[{c['index']}]{c['source']}" for c in citations)

    full_prompt = f"""你是一名专业的护理学助教，基于提供的参考资料回答用户问题。

## 参考资料
{safe_context}

## 可用引用索引
{citation_text}

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
