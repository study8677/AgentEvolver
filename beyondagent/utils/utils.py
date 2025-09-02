from typing import Any, List, Dict
from loguru import logger

# apply chat_template to a message, and then convert back to message
def convert_tool_to_user_message(tool_message, format="qwen"):
    assert format == "qwen"

    if tool_message["role"] == "user":
        return tool_message
    elif tool_message["role"] == "tool" and len(tool_message["tool_calls"])>0:
        assert len(tool_message["tool_calls"])==1
        return {
            "role": "user",
            "content": str(tool_message["tool_calls"][0]['result'])
        }


def clip_state_content_correctly(tokenizer, state_content: str, max_env_len: int) -> str:
    """
    正确地截断state_content，确保不会破坏token边界
    
    Args:
        tokenizer: 分词器
        state_content: 要截断的内容
        max_env_len: 最大允许的token长度
    
    Returns:
        截断后的内容字符串
    """
    # 先tokenize检查长度
    tokens = tokenizer(state_content, return_tensors="pt", padding=False)["input_ids"][0]
    
    if len(tokens) <= max_env_len:
        return state_content
    
    # 如果超长，截断到max_env_len长度的token
    truncated_tokens = tokens[:max_env_len]
    
    # 更安全的方式：使用tokenizer的内置方法
    # 大多数tokenizer都有更好的处理方式
    if hasattr(tokenizer, 'decode'):
        # 首先尝试保留special tokens
        try:
            truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            return truncated_content
        except:
            # 如果失败，可能是截断位置不当，尝试移除special tokens
            try:
                truncated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                return truncated_content
            except:
                # 最后的fallback：手动处理
                pass
    
    # 如果所有decode方法都失败，使用更保守的方法
    # 逐步减少token数量直到成功decode
    for i in range(min(10, max_env_len)):  # 最多尝试10次
        try:
            test_tokens = tokens[:max_env_len - i]
            truncated_content = tokenizer.decode(test_tokens, skip_special_tokens=False)
            logger.warning(f"Had to reduce token count by {i} to successfully decode")
            return truncated_content
        except:
            continue
    
    # 最终fallback：使用原始的字符截断方法
    logger.error("All token-based truncation methods failed, falling back to character truncation")
    return state_content[:max_env_len]