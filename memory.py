import json
import os

MEMORY_FILE = "user_profile.json"

def load_profile(user_id: str = "default_user") -> dict:
    """加载用户画像，如果不存在则返回空字典"""
    if not os.path.exists(MEMORY_FILE):
        return {}
    
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(user_id, {})
    except Exception as e:
        print(f"[Memory] Load Error: {e}")
        return {}

def save_profile(key: str, value: str, user_id: str = "default_user"):
    """更新用户画像的一个字段"""
    # 1. 读取现有数据
    all_data = {}
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except: pass
    
    # 2. 更新特定用户的字段
    if user_id not in all_data:
        all_data[user_id] = {}
    
    all_data[user_id][key] = value
    
    # 3. 写回文件
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"--- [记忆] 已记住: {key} = {value} ---")

def get_profile_str(user_id: str = "default_user") -> str:
    """将画像转换为 Prompt 友好的字符串"""
    profile = load_profile(user_id)
    if not profile:
        return ""
    
    lines = ["【用户长期记忆/偏好】(必须遵守):"]
    for k, v in profile.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"