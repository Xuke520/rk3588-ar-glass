"""
药物识别系统 - 快速测试脚本
测试数据库功能和OCR文本提取（不需要摄像头、cv2和PP-OCR）
"""


def test_all():
    print("=" * 50)
    print("  1. 测试数据库初始化")
    print("=" * 50)
    from medicine_data_init import init_database
    count = init_database()

    print("\n" + "=" * 50)
    print("  2. 测试药品搜索")
    print("=" * 50)
    from medicine_db import get_db
    db = get_db()

    test_keywords = ["阿莫", "布洛芬", "感冒", "降压", "维生素", "连花"]
    for kw in test_keywords:
        results = db.search_by_name(kw)
        names = [r['name'] for r in results[:3]]
        print(f"  搜索「{kw}」-> {len(results)} 个: {names}")

    print("\n" + "=" * 50)
    print("  3. 测试OCR文本提取")
    print("=" * 50)
    from medicine_recognizer import extract_medicine_names, search_medicine

    mock_ocr_texts = [
        "阿莫西林胶囊",
        "国药准字H20020343",
        "0.25g*24粒",
    ]
    candidates = extract_medicine_names(mock_ocr_texts)
    print(f"  OCR文本: {mock_ocr_texts}")
    print(f"  提取候选药名: {candidates}")

    best, all_results = search_medicine(mock_ocr_texts)
    if best:
        print(f"\n  最佳匹配: {best['name']}")
        print(f"  适应症: {best['indications'][:50]}...")

    print("\n" + "=" * 50)
    print("  4. 数据库统计")
    print("=" * 50)
    print(f"  药品总数: {db.count()} 种")
    print(f"  分类列表: {db.get_all_categories()}")

    db.close()
    print("\n" + "=" * 50)
    print("  所有测试通过！")
    print("=" * 50)


if __name__ == "__main__":
    print()
    print("药物识别系统 - 功能测试")
    print()
    test_all()
    print()
    input("按回车键退出...")
