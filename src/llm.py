#!/usr/bin/env python3
"""LLM 模块"""

from model_clients import call_deepseek


def generate_response(query: str, context: str) -> str:
    safe = context[:8000]
    prompt = f"""参考资料：
{safe}

问题：{query}

要求：仅依据以上资料回答，标注来源编号（如[1]）。资料不足则说明。"""
    return call_deepseek(prompt)
