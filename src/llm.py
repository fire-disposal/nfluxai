#!/usr/bin/env python3
"""
LLM 集成模块
支持多种 LLM 后端：DeepSeek, Ollama, 智谱, 通义千问等
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List


def get_llm_config() -> Dict[str, Any]:
    """从配置或环境变量获取 LLM 配置"""
    config_path = Path(__file__).parent.parent / "config.yaml"

    # 默认配置
    config = {
        "provider": os.getenv("LLM_PROVIDER", "deepseek"),
        "model": os.getenv("LLM_MODEL", "deepseek-chat"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "")),
        "api_base": os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1/chat/completions"),
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    if config_path.exists():
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config and "llm" in yaml_config:
                config.update(yaml_config["llm"])

    # 环境变量优先级最高
    if os.getenv("DEEPSEEK_API_KEY"):
        config["api_key"] = os.getenv("DEEPSEEK_API_KEY")
    if os.getenv("ZHIPU_API_KEY"):
        config["api_key"] = os.getenv("ZHIPU_API_KEY")
        config["provider"] = "zhipu"
    if os.getenv("DASHSCOPE_API_KEY"):
        config["api_key"] = os.getenv("DASHSCOPE_API_KEY")
        config["provider"] = "qwen"

    return config


def call_deepseek(prompt: str, config: Dict[str, Any]) -> str:
    """调用 DeepSeek API"""
    import requests

    api_key = config.get("api_key", "")
    if not api_key:
        return "[错误：未配置 DeepSeek API Key，请设置 DEEPSEEK_API_KEY 环境变量]"

    api_base = config.get("api_base", "https://api.deepseek.com/v1/chat/completions")
    model = config.get("model", "deepseek-chat")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 2048),
    }

    try:
        response = requests.post(api_base, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "[错误：API 请求超时，请稍后重试]"
    except requests.exceptions.HTTPError as e:
        return f"[错误：API 请求失败 - {e}]"
    except Exception as e:
        return f"[错误：{e}]"


def call_zhipu(prompt: str, config: Dict[str, Any]) -> str:
    """调用智谱 GLM API"""
    import requests

    api_key = config.get("api_key", "")
    if not api_key:
        return "[错误：未配置智谱 API Key，请设置 ZHIPU_API_KEY 环境变量]"

    api_base = config.get("api_base", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    model = config.get("model", "glm-4-flash")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 2048),
    }

    try:
        response = requests.post(api_base, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[错误：{e}]"


def call_qwen(prompt: str, config: Dict[str, Any]) -> str:
    """调用通义千问 API"""
    import requests

    api_key = config.get("api_key", "")
    if not api_key:
        return "[错误：未配置 DashScope API Key，请设置 DASHSCOPE_API_KEY 环境变量]"

    api_base = config.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    model = config.get("model", "qwen-turbo")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 2048),
    }

    try:
        response = requests.post(api_base, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[错误：{e}]"


def call_ollama(prompt: str, config: Dict[str, Any]) -> str:
    """调用本地 Ollama"""
    import requests

    model = config.get("model", "qwen2.5:7b")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.get("temperature", 0.7),
                    "num_predict": config.get("max_tokens", 2048),
                }
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "[错误：无法连接 Ollama 服务，请确保已运行 'ollama serve']"
    except Exception as e:
        return f"[错误：{e}]"


def generate_response(prompt: str, context: str, citations: List[Dict]) -> str:
    """
    生成 LLM 回答

    Args:
        prompt: 用户问题
        context: 检索到的上下文
        citations: 引用列表

    Returns:
        LLM 生成的回答
    """
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

    provider = config.get("provider", "deepseek")

    if provider == "deepseek":
        return call_deepseek(full_prompt, config)
    elif provider == "zhipu":
        return call_zhipu(full_prompt, config)
    elif provider == "qwen":
        return call_qwen(full_prompt, config)
    elif provider == "ollama":
        return call_ollama(full_prompt, config)
    else:
        return _fallback_response(prompt, context, citations)


def _fallback_response(prompt: str, context: str, citations: List[Dict]) -> str:
    """降级回答生成（无 LLM 时）"""
    response = "根据护理教材资料，以下是相关信息：\n\n"

    for i, cite in enumerate(citations[:3], 1):
        title = cite.get('metadata', {}).get('title', cite.get('metadata', {}).get('header', '相关内容'))
        source = cite.get('source', '')
        response += f"**{source}** [{i}]\n{title}\n\n"

    response += "---\n*注：部署完整 LLM 后可获得更准确的回答。*"
    return response


if __name__ == "__main__":
    # 测试
    print("LLM 配置:", get_llm_config())

    test_context = """[1] 内科护理学/第二章/呼吸系统
呼吸系统疾病病人护理评估包括病史采集、体格检查等。

[2] 新编护理学基础/第八章/护理程序
护理程序包括评估、诊断、计划、实施、评价五个步骤。"""

    test_citations = [
        {"index": 1, "source": "[内科护理学/第二章]", "metadata": {"title": "呼吸系统疾病"}},
        {"index": 2, "source": "[新编护理学基础/第八章]", "metadata": {"title": "护理程序"}},
    ]

    response = generate_response("护理程序有哪些步骤？", test_context, test_citations)
    print("\n回答:\n", response)