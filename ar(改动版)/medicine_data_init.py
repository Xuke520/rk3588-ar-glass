"""
常用药品数据库初始化脚本
首次运行时自动填充常见药品数据
"""

from medicine_db import MedicineDB, get_db

# ===== 常见药品数据 =====
COMMON_MEDICINES = [
    {
        "name": "阿莫西林胶囊",
        "pinyin": "Amoxilin",
        "category": "抗生素",
        "dosage_form": "胶囊",
        "specification": "0.25g*24粒",
        "indications": "适用于敏感菌所致的呼吸道感染、泌尿生殖道感染、皮肤软组织感染等。",
        "usage_dosage": "口服。成人一次0.5g，每6-8小时1次。儿童一日20-40mg/kg，每8小时1次。",
        "side_effects": "恶心、呕吐、腹泻等胃肠道反应；皮疹、药物热等过敏反应。",
        "contraindications": "对青霉素类药物过敏者禁用；青霉素皮试阳性者禁用。",
        "precautions": "用前必须做青霉素皮试。肾功能严重损害者需调整剂量。",
        "drug_interactions": "与丙磺舒合用可提高血药浓度；与甲氨蝶呤合用可增加其毒性。",
        "manufacturer": "常用品牌：阿莫仙、再林",
        "approval_number": "国药准字H20020343",
        "storage_condition": "密封，在阴凉干燥处保存（不超过20℃）",
        "aliases": ["阿莫仙", "再林", "阿莫灵"],
        "ingredients": [{"name": "阿莫西林", "quantity": "0.25g"}]
    },
    {
        "name": "头孢克肟胶囊",
        "pinyin": "Toubaokewo",
        "category": "抗生素",
        "dosage_form": "胶囊",
        "specification": "50mg*6粒",
        "indications": "用于支气管炎、肺炎、肾盂肾炎、膀胱炎、淋球菌性尿道炎等。",
        "usage_dosage": "口服。成人及体重30kg以上儿童：一次50-100mg，一日2次。",
        "side_effects": "腹泻、胃部不适、恶心、皮疹等。",
        "contraindications": "对头孢菌素类过敏者禁用。",
        "precautions": "严重肾功能障碍者慎用。",
        "drug_interactions": "与利尿剂合用可增加肾毒性。",
        "manufacturer": "常见品牌：世福素",
        "approval_number": "国药准字H20020542",
        "storage_condition": "密封保存",
        "aliases": ["世福素", "头孢克肟"],
        "ingredients": [{"name": "头孢克肟", "quantity": "50mg"}]
    },
    {
        "name": "布洛芬缓释胶囊",
        "pinyin": "Buluofen",
        "category": "解热镇痛药",
        "dosage_form": "缓释胶囊",
        "specification": "0.3g*20粒",
        "indications": "用于缓解轻至中度疼痛如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经。也用于普通感冒或流行性感冒引起的发热。",
        "usage_dosage": "口服。成人一次1粒（0.3g），一日2次（早晚各一次）。",
        "side_effects": "少数病人可出现恶心、呕吐、胃烧灼感或轻度消化不良等。",
        "contraindications": "对本品过敏者禁用；活动性消化性溃疡者禁用；严重肝肾功能不全者禁用。",
        "precautions": "孕妇及哺乳期妇女慎用；不宜长期服用。",
        "drug_interactions": "与抗凝血药合用可增加出血风险；与阿司匹林合用增加胃肠道不良反应。",
        "manufacturer": "常见品牌：芬必得",
        "approval_number": "国药准字H10900089",
        "storage_condition": "密封保存",
        "aliases": ["芬必得", "布洛芬", "异丁苯丙酸"],
        "ingredients": [{"name": "布洛芬", "quantity": "0.3g"}]
    },
    {
        "name": "对乙酰氨基酚片",
        "pinyin": "Duiyixiananjifen",
        "category": "解热镇痛药",
        "dosage_form": "片剂",
        "specification": "0.5g*10片",
        "indications": "用于普通感冒或流行性感冒引起的发热，也用于缓解轻至中度疼痛。",
        "usage_dosage": "口服。成人一次0.3-0.6g，一日3-4次。一日最大量不超过2g。",
        "side_effects": "偶见皮疹、荨麻疹。过量使用可引起严重肝损伤。",
        "contraindications": "严重肝肾功能不全者禁用。",
        "precautions": "一日剂量不超过2g。服药期间不得饮酒。",
        "drug_interactions": "与华法林合用可增强抗凝作用。",
        "manufacturer": "常见品牌：泰诺林、扑热息痛",
        "approval_number": "国药准字H31020560",
        "storage_condition": "密封保存",
        "aliases": ["扑热息痛", "泰诺林", "必理通"],
        "ingredients": [{"name": "对乙酰氨基酚", "quantity": "0.5g"}]
    },
    {
        "name": "阿司匹林肠溶片",
        "pinyin": "Asipilin",
        "category": "抗血小板药",
        "dosage_form": "肠溶片",
        "specification": "100mg*30片",
        "indications": "抑制血小板聚集，用于预防心脑血管疾病。降低急性心肌梗死疑似患者的发病风险。",
        "usage_dosage": "口服。肠溶片宜饭前用适量水送服。预防心脑血管疾病：一日1片（100mg）。",
        "side_effects": "胃肠道反应如恶心、呕吐；长期服用可致胃肠道出血。",
        "contraindications": "活动性消化性溃疡者禁用；血友病或血小板减少症者禁用；对阿司匹林过敏者禁用。",
        "precautions": "手术前一周停用。饮酒后不宜服用。",
        "drug_interactions": "与抗凝血药合用增加出血风险；与甲氨蝶呤合用增加毒性。",
        "manufacturer": "常见品牌：拜阿司匹灵",
        "approval_number": "国药准字J20080078",
        "storage_condition": "密封，25℃以下保存",
        "aliases": ["拜阿司匹灵", "乙酰水杨酸"],
        "ingredients": [{"name": "阿司匹林", "quantity": "100mg"}]
    },
    {
        "name": "硝苯地平控释片",
        "pinyin": "Xiaobendiping",
        "category": "降压药",
        "dosage_form": "控释片",
        "specification": "30mg*7片",
        "indications": "治疗高血压、冠心病、慢性稳定型心绞痛。",
        "usage_dosage": "口服。一次30mg（1片），一日1次。不可掰开服用。",
        "side_effects": "头痛、面部潮红、踝部水肿、心悸、头晕等。",
        "contraindications": "心源性休克者禁用；妊娠期禁用。",
        "precautions": "不可与葡萄柚汁同服。严重低血压者慎用。",
        "drug_interactions": "与CYP3A4抑制剂合用可增加血药浓度。",
        "manufacturer": "常见品牌：拜新同",
        "approval_number": "国药准字J20080091",
        "storage_condition": "密封保存",
        "aliases": ["拜新同", "心痛定"],
        "ingredients": [{"name": "硝苯地平", "quantity": "30mg"}]
    },
    {
        "name": "盐酸二甲双胍片",
        "pinyin": "Erjiashuanggua",
        "category": "降糖药",
        "dosage_form": "片剂",
        "specification": "0.5g*20片",
        "indications": "用于2型糖尿病，尤其适用于肥胖患者。",
        "usage_dosage": "口服。成人开始一次0.25g，一日2-3次，以后根据疗效逐渐加量。",
        "side_effects": "恶心、呕吐、腹泻、口中有金属味。",
        "contraindications": "严重肾功能不全者禁用；肝功能不全者禁用；酗酒者禁用。",
        "precautions": "定期检查肾功能。用药期间避免饮酒。",
        "drug_interactions": "与胰岛素合用可增加低血糖风险。",
        "manufacturer": "常见品牌：格华止",
        "approval_number": "国药准字H20023370",
        "storage_condition": "密封保存",
        "aliases": ["格华止", "二甲双胍"],
        "ingredients": [{"name": "盐酸二甲双胍", "quantity": "0.5g"}]
    },
    {
        "name": "氯雷他定片",
        "pinyin": "Lüleitading",
        "category": "抗过敏药",
        "dosage_form": "片剂",
        "specification": "10mg*6片",
        "indications": "用于缓解过敏性鼻炎有关的症状，如喷嚏、流涕、鼻痒。也用于缓解慢性荨麻疹。",
        "usage_dosage": "口服。成人及12岁以上儿童：一次1片（10mg），一日1次。",
        "side_effects": "乏力、头痛、嗜睡、口干、胃肠道不适等。",
        "contraindications": "对本品过敏者禁用。",
        "precautions": "严重肝肾功能不全者慎用。",
        "drug_interactions": "与酮康唑、红霉素合用可增加血药浓度。",
        "manufacturer": "常见品牌：开瑞坦、息斯敏",
        "approval_number": "国药准字H10970410",
        "storage_condition": "密封保存",
        "aliases": ["开瑞坦", "氯雷他定"],
        "ingredients": [{"name": "氯雷他定", "quantity": "10mg"}]
    },
    {
        "name": "藿香正气水",
        "pinyin": "Huoxiangzhengqi",
        "category": "中成药",
        "dosage_form": "酊剂",
        "specification": "10ml*10支",
        "indications": "解表化湿，理气和中。用于外感风寒、内伤湿滞或夏伤暑湿所致的感冒。",
        "usage_dosage": "口服。一次5-10ml，一日2次，用时摇匀。",
        "side_effects": "偶有过敏反应。含酒精，服后不得驾驶。",
        "contraindications": "对本品及酒精过敏者禁用。",
        "precautions": "本品含乙醇，服后不得驾驶机、车、船。",
        "drug_interactions": "与头孢类抗生素合用可引起双硫仑样反应。",
        "manufacturer": "常见品牌：太极集团",
        "approval_number": "国药准字Z20000068",
        "storage_condition": "密封保存",
        "aliases": ["藿香正气"],
        "ingredients": [{"name": "苍术", "quantity": ""}, {"name": "陈皮", "quantity": ""}, {"name": "厚朴", "quantity": ""}, {"name": "白芷", "quantity": ""}, {"name": "茯苓", "quantity": ""}, {"name": "大腹皮", "quantity": ""}, {"name": "生半夏", "quantity": ""}]
    },
    {
        "name": "连花清瘟胶囊",
        "pinyin": "Lianhuaqingwen",
        "category": "中成药",
        "dosage_form": "胶囊",
        "specification": "0.35g*24粒",
        "indications": "清瘟解毒，宣肺泄热。用于治疗流行性感冒属热毒袭肺证。",
        "usage_dosage": "口服。一次4粒，一日3次。",
        "side_effects": "偶见胃部不适、恶心。",
        "contraindications": "对本品过敏者禁用；严重肝肾功能不全者禁用。",
        "precautions": "风寒感冒者不适用。不宜同时服用滋补性中药。",
        "drug_interactions": "",
        "manufacturer": "以岭药业",
        "approval_number": "国药准字Z20040063",
        "storage_condition": "密封，置阴凉处",
        "aliases": ["连花清瘟"],
        "ingredients": [{"name": "连翘", "quantity": ""}, {"name": "金银花", "quantity": ""}, {"name": "炙麻黄", "quantity": ""}, {"name": "杏仁", "quantity": ""}, {"name": "石膏", "quantity": ""}]
    },
    {
        "name": "板蓝根颗粒",
        "pinyin": "Banlangen",
        "category": "中成药",
        "dosage_form": "颗粒剂",
        "specification": "10g*20袋",
        "indications": "清热解毒，凉血利咽。用于肺胃热盛所致的咽喉肿痛、口咽干燥。",
        "usage_dosage": "开水冲服。一次5-10g，一日3-4次。",
        "side_effects": "尚不明确。",
        "contraindications": "对本品过敏者禁用。",
        "precautions": "糖尿病患者慎用含糖颗粒。",
        "drug_interactions": "",
        "manufacturer": "常见品牌：白云山、同仁堂",
        "approval_number": "国药准字Z44023485",
        "storage_condition": "密封保存",
        "aliases": ["板蓝根"],
        "ingredients": [{"name": "板蓝根", "quantity": ""}]
    },
    {
        "name": "奥美拉唑肠溶胶囊",
        "pinyin": "Aomeilazuo",
        "category": "消化系统药",
        "dosage_form": "肠溶胶囊",
        "specification": "20mg*14粒",
        "indications": "用于胃溃疡、十二指肠溃疡、反流性食管炎、卓-艾综合征。",
        "usage_dosage": "口服。一次20mg，一日1-2次，晨起空腹服用效果最佳。",
        "side_effects": "头痛、腹泻、便秘、腹痛、恶心等。",
        "contraindications": "对本品过敏者禁用。",
        "precautions": "长期服用（超过1年）需监测维生素B12水平。",
        "drug_interactions": "可降低氯吡格雷疗效；与地高辛合用可增加其吸收。",
        "manufacturer": "常见品牌：洛赛克、奥克",
        "approval_number": "国药准字H20033490",
        "storage_condition": "密封，阴凉干燥处保存",
        "aliases": ["洛赛克", "奥克", "奥美拉唑"],
        "ingredients": [{"name": "奥美拉唑", "quantity": "20mg"}]
    },
    {
        "name": "阿托伐他汀钙片",
        "pinyin": "Atuofatating",
        "category": "降脂药",
        "dosage_form": "片剂",
        "specification": "20mg*7片",
        "indications": "用于高胆固醇血症和冠心病的治疗。降低心肌梗死、卒中的风险。",
        "usage_dosage": "口服。起始剂量10mg，一日1次。根据血脂水平调整。",
        "side_effects": "鼻咽炎、关节痛、腹泻、肌肉痛。",
        "contraindications": "活动性肝病者禁用；妊娠期禁用。",
        "precautions": "治疗前和治疗后定期监测肝功能。出现不明原因肌肉疼痛应及时就诊。",
        "drug_interactions": "与CYP3A4抑制剂（如红霉素）合用增加肌病风险。",
        "manufacturer": "常见品牌：立普妥",
        "approval_number": "国药准字H20051408",
        "storage_condition": "密封保存",
        "aliases": ["立普妥", "阿托伐他汀"],
        "ingredients": [{"name": "阿托伐他汀钙", "quantity": "20mg"}]
    },
    {
        "name": "氨酚黄那敏颗粒",
        "pinyin": "Anfenhuanganmin",
        "category": "感冒药",
        "dosage_form": "颗粒剂",
        "specification": "每袋含对乙酰氨基酚125mg",
        "indications": "用于儿童普通感冒及流行性感冒引起的发热、头痛、鼻塞、流涕等。",
        "usage_dosage": "开水冲服。1-3岁一次半袋，4-6岁一次1袋，一日3次。",
        "side_effects": "偶见粒细胞缺乏症、皮疹。",
        "contraindications": "对本品成分过敏者禁用。",
        "precautions": "不宜与其他含对乙酰氨基酚的药物同时使用。",
        "drug_interactions": "",
        "manufacturer": "常见品牌：小快克、护彤",
        "approval_number": "国药准字H13023418",
        "storage_condition": "密封保存",
        "aliases": ["小快克", "护彤", "氨酚黄那敏"],
        "ingredients": [
            {"name": "对乙酰氨基酚", "quantity": "125mg"},
            {"name": "人工牛黄", "quantity": "5mg"},
            {"name": "马来酸氯苯那敏", "quantity": "0.5mg"}
        ]
    },
    {
        "name": "维生素C片",
        "pinyin": "WeishengC",
        "category": "维生素",
        "dosage_form": "片剂",
        "specification": "100mg*100片",
        "indications": "用于预防和治疗坏血病，也可用于各种急慢性传染疾病及紫癜的辅助治疗。",
        "usage_dosage": "口服。预防用量一日50-200mg；治疗用量一次100-200mg，一日3次。",
        "side_effects": "大量服用可引起腹泻、胃酸过多、泌尿系结石。",
        "contraindications": "高草酸盐尿症、肾结石者禁用。",
        "precautions": "不宜长期大量服用。",
        "drug_interactions": "与华法林合用可减弱抗凝作用。",
        "manufacturer": "常见品牌：东北制药",
        "approval_number": "国药准字H21020713",
        "storage_condition": "遮光、密封保存",
        "aliases": ["VC", "抗坏血酸"],
        "ingredients": [{"name": "维生素C", "quantity": "100mg"}]
    },
    {
        "name": "蒙脱石散",
        "pinyin": "Mengtuoshi",
        "category": "消化系统药",
        "dosage_form": "散剂",
        "specification": "3g*10袋",
        "indications": "用于成人及儿童急慢性腹泻。",
        "usage_dosage": "口服。成人一次1袋（3g），一日3次。两餐间服用。",
        "side_effects": "偶见便秘。",
        "contraindications": "对本品过敏者禁用。",
        "precautions": "与其他药物间隔2小时以上服用。",
        "drug_interactions": "可影响其他药物吸收，需间隔服用。",
        "manufacturer": "常见品牌：思密达、必奇",
        "approval_number": "国药准字H20073953",
        "storage_condition": "密封保存",
        "aliases": ["思密达", "必奇", "蒙脱石"],
        "ingredients": [{"name": "蒙脱石", "quantity": "3g"}]
    },
    {
        "name": "氨溴索口服液",
        "pinyin": "Anxiusuo",
        "category": "呼吸系统药",
        "dosage_form": "口服液",
        "specification": "100ml:0.3g",
        "indications": "用于急慢性支气管炎、支气管哮喘等引起的痰液粘稠、咳痰困难。",
        "usage_dosage": "口服。成人一次10ml，一日2次；儿童一次5ml，一日2次。",
        "side_effects": "偶见胃部不适、恶心。",
        "contraindications": "对本品过敏者禁用。",
        "precautions": "孕妇慎用。",
        "drug_interactions": "",
        "manufacturer": "常见品牌：沐舒坦",
        "approval_number": "国药准字H20010340",
        "storage_condition": "密封保存",
        "aliases": ["沐舒坦", "氨溴索"],
        "ingredients": [{"name": "盐酸氨溴索", "quantity": "0.3g/100ml"}]
    },
    {
        "name": "复方丹参滴丸",
        "pinyin": "Fufangdanshen",
        "category": "中成药",
        "dosage_form": "滴丸",
        "specification": "27mg*180丸",
        "indications": "活血化瘀，理气止痛。用于气滞血瘀所致的胸痹。",
        "usage_dosage": "口服或舌下含服。一次10丸，一日3次。",
        "side_effects": "偶见胃肠道不适。",
        "contraindications": "对本品过敏者禁用；出血性疾病者禁用。",
        "precautions": "孕妇慎用。",
        "drug_interactions": "与抗凝血药合用需谨慎。",
        "manufacturer": "天士力",
        "approval_number": "国药准字Z20010011",
        "storage_condition": "密封，置阴凉处",
        "aliases": ["复方丹参"],
        "ingredients": [{"name": "丹参", "quantity": ""}, {"name": "三七", "quantity": ""}, {"name": "冰片", "quantity": ""}]
    },
    {
        "name": "马来酸依那普利片",
        "pinyin": "Yinapuli",
        "category": "降压药",
        "dosage_form": "片剂",
        "specification": "10mg*16片",
        "indications": "用于治疗高血压、心力衰竭。",
        "usage_dosage": "口服。高血压：起始5mg，一日1次，可增至10-20mg。",
        "side_effects": "干咳、头晕、乏力、低血压。",
        "contraindications": "妊娠期禁用；血管神经性水肿史者禁用。",
        "precautions": "首次服药可能出现低血压。定期监测血钾和肾功能。",
        "drug_interactions": "与利尿剂合用增加降压效果；与保钾利尿剂合用可致高钾血症。",
        "manufacturer": "常见品牌：悦宁定",
        "approval_number": "国药准字H32026567",
        "storage_condition": "密封保存",
        "aliases": ["悦宁定", "依那普利"],
        "ingredients": [{"name": "马来酸依那普利", "quantity": "10mg"}]
    },
    {
        "name": "三九胃泰颗粒",
        "pinyin": "Sanjiuweitai",
        "category": "中成药",
        "dosage_form": "颗粒剂",
        "specification": "20g*6袋",
        "indications": "清热燥湿，行气活血，柔肝止痛。用于上腹隐痛、饱胀、反酸等慢性胃炎。",
        "usage_dosage": "开水冲服。一次1袋，一日2次。",
        "side_effects": "尚不明确。",
        "contraindications": "对本品过敏者禁用。",
        "precautions": "胃寒者慎用。",
        "drug_interactions": "",
        "manufacturer": "华润三九",
        "approval_number": "国药准字Z44021208",
        "storage_condition": "密封保存",
        "aliases": ["三九胃泰"],
        "ingredients": [{"name": "三叉苦", "quantity": ""}, {"name": "九里香", "quantity": ""}, {"name": "两面针", "quantity": ""}]
    },
]


def init_database():
    """初始化/更新数据库"""
    db = get_db()

    count_before = db.count()
    print(f"[药物数据库] 当前有 {count_before} 种药品")

    added = 0
    for med in COMMON_MEDICINES:
        if db.add_medicine(med):
            added += 1

    count_after = db.count()
    print(f"[药物数据库] 新增 {added} 种，现有共 {count_after} 种药品")
    return count_after


if __name__ == "__main__":
    init_database()

    # 测试搜索
    db = get_db()
    print("\n--- 测试搜索 ---")
    for keyword in ["阿莫", "布洛芬", "感冒", "降压"]:
        results = db.search_by_name(keyword)
        print(f"\n搜索「{keyword}」找到 {len(results)} 个结果:")
        for r in results:
            print(f"  - {r['name']} ({r['category']}) [{r['specification']}]")

    db.close()
