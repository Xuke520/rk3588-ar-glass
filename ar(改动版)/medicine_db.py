"""
药物数据库模块 - 使用SQLite存储药品信息
ELF+2 开发板 (RK3588) 药物识别系统
"""

import sqlite3
import os
import json
from typing import Optional, Dict, List, Tuple


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medicine_data", "medicine.db")


class MedicineDB:
    """药物本地数据库，支持增删查改和模糊搜索"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """连接数据库（线程安全，每次操作使用新连接或复用）"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

    def _create_tables(self):
        """建表：药品主表 + 别名表 + 成分表"""
        cursor = self.conn.cursor()

        # 药品主表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medicines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,              -- 药品通用名
                pinyin TEXT DEFAULT '',                  -- 拼音助记
                category TEXT DEFAULT '',                -- 分类（如：抗生素、降压药…）
                dosage_form TEXT DEFAULT '',             -- 剂型（片剂、胶囊…）
                specification TEXT DEFAULT '',           -- 规格（如：0.25g*12片）
                usage_dosage TEXT DEFAULT '',            -- 用法用量
                side_effects TEXT DEFAULT '',            -- 不良反应
                contraindications TEXT DEFAULT '',       -- 禁忌
                precautions TEXT DEFAULT '',             -- 注意事项
                drug_interactions TEXT DEFAULT '',       -- 药物相互作用
                indications TEXT DEFAULT '',             -- 适应症
                manufacturer TEXT DEFAULT '',            -- 生产厂家
                approval_number TEXT DEFAULT '',         -- 批准文号
                storage_condition TEXT DEFAULT '',       -- 储存条件
                expiry_warning TEXT DEFAULT '',          -- 过期提醒
                image_path TEXT DEFAULT '',              -- 药盒参考图片路径
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 别名表（商品名 / 常用别名 → 药品ID）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                alias TEXT NOT NULL,                     -- 别名（商品名/品牌名）
                FOREIGN KEY (medicine_id) REFERENCES medicines(id),
                UNIQUE(alias)
            )
        """)

        # 成分表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                ingredient_name TEXT NOT NULL,            -- 成分名称
                quantity TEXT DEFAULT '',                 -- 含量
                FOREIGN KEY (medicine_id) REFERENCES medicines(id)
            )
        """)

        # 索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_medicines_name ON medicines(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_med_id ON aliases(medicine_id)")

        self.conn.commit()

    def add_medicine(self, data: Dict) -> bool:
        """添加一种药品"""
        try:
            cursor = self.conn.cursor()
            fields = {k: v for k, v in data.items() if k != 'aliases' and k != 'ingredients'}
            placeholders = ', '.join(['?' for _ in fields])
            columns = ', '.join(fields.keys())
            cursor.execute(
                f"INSERT OR REPLACE INTO medicines ({columns}) VALUES ({placeholders})",
                list(fields.values())
            )
            med_id = cursor.lastrowid

            # 添加别名
            if 'aliases' in data and data['aliases']:
                for alias in data['aliases']:
                    try:
                        cursor.execute(
                            "INSERT OR IGNORE INTO aliases (medicine_id, alias) VALUES (?, ?)",
                            (med_id, alias.strip())
                        )
                    except Exception:
                        pass

            # 添加成分
            if 'ingredients' in data and data['ingredients']:
                for ing in data['ingredients']:
                    if isinstance(ing, dict):
                        cursor.execute(
                            "INSERT INTO ingredients (medicine_id, ingredient_name, quantity) VALUES (?, ?, ?)",
                            (med_id, ing.get('name', ''), ing.get('quantity', ''))
                        )

            self.conn.commit()
            return True
        except Exception as e:
            print(f"[MedicineDB] 添加药品失败: {e}")
            return False

    def search_by_name(self, keyword: str) -> List[Dict]:
        """按药品名称或别名模糊搜索（全字段搜索）"""
        cursor = self.conn.cursor()
        like_pattern = f"%{keyword}%"

        # 搜索主表（名称 + 拼音 + 分类 + 适应症 + 厂家 + 注意事项）
        cursor.execute(
            """SELECT * FROM medicines
               WHERE name LIKE ? OR pinyin LIKE ? OR category LIKE ?
                  OR indications LIKE ? OR manufacturer LIKE ?
                  OR usage_dosage LIKE ? OR precautions LIKE ?
               ORDER BY name LIMIT 30""",
            (like_pattern, like_pattern, like_pattern,
             like_pattern, like_pattern, like_pattern, like_pattern)
        )
        results = [dict(row) for row in cursor.fetchall()]

        # 搜索别名表
        cursor.execute("""
            SELECT m.* FROM medicines m
            JOIN aliases a ON m.id = a.medicine_id
            WHERE a.alias LIKE ?
            ORDER BY m.name LIMIT 20
        """, (like_pattern,))
        alias_results = [dict(row) for row in cursor.fetchall()]

        # 搜索成分表
        cursor.execute("""
            SELECT DISTINCT m.* FROM medicines m
            JOIN ingredients i ON m.id = i.medicine_id
            WHERE i.ingredient_name LIKE ?
            ORDER BY m.name LIMIT 20
        """, (like_pattern,))
        ing_results = [dict(row) for row in cursor.fetchall()]

        # 合并去重
        seen_ids = {r['id'] for r in results}
        for r in alias_results:
            if r['id'] not in seen_ids:
                results.append(r)
                seen_ids.add(r['id'])
        for r in ing_results:
            if r['id'] not in seen_ids:
                results.append(r)
                seen_ids.add(r['id'])

        # 为每个结果加载别名和成分
        for r in results:
            r['aliases'] = self.get_aliases(r['id'])
            r['ingredients'] = self.get_ingredients(r['id'])

        return results

    def get_by_id(self, med_id: int) -> Optional[Dict]:
        """按ID查询"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM medicines WHERE id = ?", (med_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['aliases'] = self.get_aliases(med_id)
            result['ingredients'] = self.get_ingredients(med_id)
            return result
        return None

    def get_aliases(self, med_id: int) -> List[str]:
        """获取药品别名"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT alias FROM aliases WHERE medicine_id = ?", (med_id,))
        return [row['alias'] for row in cursor.fetchall()]

    def get_ingredients(self, med_id: int) -> List[Dict]:
        """获取药品成分"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT ingredient_name, quantity FROM ingredients WHERE medicine_id = ?",
            (med_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_categories(self) -> List[str]:
        """获取所有药品分类"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM medicines WHERE category != '' ORDER BY category")
        return [row['category'] for row in cursor.fetchall()]

    def count(self) -> int:
        """获取药品总数"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM medicines")
        return cursor.fetchone()['cnt']

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# ===== 快捷函数 =====
_db_instance = None


def get_db() -> MedicineDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = MedicineDB()
    return _db_instance
