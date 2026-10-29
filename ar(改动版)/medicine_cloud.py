"""
云端药品查询 API（基于 DashScope 通义千问）
当本地数据库找不到时，自动调用云端API获取药品信息并缓存到本地
ELF+2 开发板 (RK3588) 药物识别系统
"""

import os
import json
import hashlib
from typing import Optional, Dict, List
from medicine_db import get_db

# 尝试导入 dashscope
try:
    import dashscope
    from dashscope import Generation
    _HAS_DASHSCOPE = True
except ImportError:
    _HAS_DASHSCOPE = False

# 结果缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medicine_data", "cloud_cache")


class MedicineCloudAPI:
    """
    云端药品查询
    使用 DashScope (通义千问) 作为兜底查询
    """

    def __init__(self, api_key: Optional[str] = None):
        os.makedirs(CACHE_DIR, exist_ok=True)

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get('DASHSCOPE_API_KEY', '')

        if _HAS_DASHSCOPE and self.api_key:
            dashscope.api_key = self.api_key
            self.available = True
        else:
            self.available = False
            if not _HAS_DASHSCOPE:
                print("[云端药品] dashscope 库未安装，云端查询不可用")
            elif not self.api_key:
                print("[云端药品] DASHSCOPE_API_KEY 未设置，云端查询不可用")

    def query_medicine(self, medicine_name: str) -> Optional[Dict]:
        """
        查询药品信息
        先查本地缓存 → 再查云端 → 自动存入数据库
        """
        # 1. 查缓存
        cached = self._load_cache(medicine_name)
        if cached:
            print(f"[云端药品] 命中缓存: {medicine_name}")
            return cached

        # 2. 云端查询
        if not self.available:
            return None

        result = self._query_dashscope(medicine_name)
        if result:
            # 3. 写入缓存 & 数据库
            self._save_cache(medicine_name, result)
            self._save_to_db(result)
            return result

        return None

    def _query_dashscope(self, medicine_name: str) -> Optional[Dict]:
        """调用通义千问查询药品信息"""
        prompt = f"""你是一个药品信息查询助手。请提供以下药品的详细信息，以JSON格式返回：

药品名称：{medicine_name}

请返回如下格式的JSON（不要包含其他文字）：
{{
    "name": "药品通用名",
    "category": "药品分类",
    "dosage_form": "剂型",
    "specification": "常见规格",
    "indications": "适应症",
    "usage_dosage": "用法用量",
    "side_effects": "不良反应",
    "contraindications": "禁忌",
    "precautions": "注意事项",
    "drug_interactions": "药物相互作用",
    "manufacturer": "常见生产厂家",
    "storage_condition": "储存条件",
    "aliases": ["别名1", "别名2"]
}}

如果这个不是药品名称，返回一个空JSON对象。
仅返回JSON，不要有任何其他文字。"""

        try:
            response = Generation.call(
                model='qwen-turbo',
                messages=[{'role': 'user', 'content': prompt}],
                result_format='message',
                temperature=0.1,
                max_tokens=2000
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                # 尝试解析JSON
                # 清理markdown代码块标记
                content = content.strip()
                if content.startswith('```'):
                    content = content.split('\n', 1)[-1]
                    content = content.rsplit('```', 1)[0]
                content = content.strip()

                data = json.loads(content)
                if data and data.get('name'):
                    print(f"[云端药品] 成功查询到: {data['name']}")
                    return data
                else:
                    print(f"[云端药品] 返回数据为空")
                    return None
            else:
                print(f"[云端药品] API调用失败: {response.status_code}")
                return None

        except json.JSONDecodeError as e:
            print(f"[云端药品] JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"[云端药品] 查询异常: {e}")
            return None

    def _load_cache(self, name: str) -> Optional[Dict]:
        """加载本地缓存"""
        cache_file = self._get_cache_path(name)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, name: str, data: Dict):
        """保存到缓存"""
        try:
            cache_file = self._get_cache_path(name)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[云端药品] 缓存写入失败: {e}")

    def _save_to_db(self, data: Dict):
        """将云端结果存入本地数据库"""
        try:
            db = get_db()

            # 检查是否已存在
            existing = db.search_by_name(data['name'])
            if existing:
                return  # 已存在，跳过

            # 转换为数据库格式
            med_data = {
                'name': data.get('name', ''),
                'category': data.get('category', ''),
                'dosage_form': data.get('dosage_form', ''),
                'specification': data.get('specification', ''),
                'indications': data.get('indications', ''),
                'usage_dosage': data.get('usage_dosage', ''),
                'side_effects': data.get('side_effects', ''),
                'contraindications': data.get('contraindications', ''),
                'precautions': data.get('precautions', ''),
                'drug_interactions': data.get('drug_interactions', ''),
                'manufacturer': data.get('manufacturer', ''),
                'storage_condition': data.get('storage_condition', ''),
                'aliases': data.get('aliases', []),
                'ingredients': data.get('ingredients', [])
            }
            db.add_medicine(med_data)
            print(f"[云端药品] 已缓存到本地数据库: {data['name']}")
        except Exception as e:
            print(f"[云端药品] 本地存储失败: {e}")

    def _get_cache_path(self, name: str) -> str:
        """获取缓存文件路径"""
        hash_name = hashlib.md5(name.encode('utf-8')).hexdigest()
        return os.path.join(CACHE_DIR, f"{hash_name}.json")


# ===== 快捷函数 =====
_cloud_api_instance = None


def get_cloud_api() -> MedicineCloudAPI:
    global _cloud_api_instance
    if _cloud_api_instance is None:
        _cloud_api_instance = MedicineCloudAPI()
    return _cloud_api_instance


if __name__ == "__main__":
    # 测试云端查询
    api = MedicineCloudAPI(api_key="your-api-key-here")
    if api.available:
        result = api.query_medicine("阿莫西林")
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("查询失败或API不可用")
    else:
        print("云端API未配置，请设置DASHSCOPE_API_KEY环境变量")
