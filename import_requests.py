import requests
import json
import time

# CBDB API 基础 URL (哈佛服务器)
# 文档参考: https://projects.iq.harvard.edu/cbdb/web-api
BASE_URL = "https://cbdb.fas.harvard.edu/cbdbapi/person.php"

def search_person(name_cn):
    """
    测试功能：输入中文名，获取 CBDB 返回的 JSON 数据
    """
    print(f"📡 正在尝试连接 CBDB 搜索: {name_cn} ...")
    
    # 构造参数
    params = {
        "name": name_cn,
        "o": "json"  # 强制要求返回 JSON 格式，默认是 XML
    }
    
    try:
        # 发送请求 (设置 10 秒超时，防止卡死)
        response = requests.get(BASE_URL, params=params, timeout=10)
        
        # 检查 HTTP 状态码
        if response.status_code == 200:
            print("✅ 连接成功! (Status 200)")
            
            # 解析 JSON
            # 注意：CBDB 有时返回的 header 声明不规范，如果报错可能需要手动处理 encoding
            response.encoding = 'utf-8' 
            data = response.json()
            
            return data
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏳ 连接超时！CBDB 服务器可能响应较慢，请重试或检查网络。")
        return None
    except requests.exceptions.ConnectionError:
        print("🚫 连接错误！无法连接到哈佛服务器。")
        return None
    except json.JSONDecodeError:
        print("⚠️ 数据解析失败！返回的可能不是有效的 JSON (可能是 XML 或 HTML 报错页面)。")
        print("返回原始内容片段:", response.text[:200])
        return None

def parse_and_display(data):
    """
    简单解析并打印一些我们关心的字段，看看能不能用
    """
    if not data or 'PersonAuthority' not in data:
        print("没有找到相关人物数据。")
        return

    # CBDB 返回的数据通常包裹在 PersonAuthority -> PersonInfo 列表中
    people = data['PersonAuthority']['PersonInfo']
    
    # 如果只有一个人，API 可能返回字典而不是列表，统一转为列表处理
    if isinstance(people, dict):
        people = [people]
        
    print(f"\n🔍 搜索结果 (共找到 {len(people)} 人):")
    print("-" * 40)
    
    for idx, p in enumerate(people):
        # 提取关键信息
        person_id = p.get('PersonId', 'N/A')
        name = p.get('PersonName', {}).get('BasicInfo', {}).get('ChName', '未知')
        
        # 提取生卒年 (IndexYear) 或 生卒详细
        year_info = p.get('PersonName', {}).get('BasicInfo', {}).get('YearRange', '未知年份')
        
        # 提取籍贯 (Addresses)
        addr_info = "未知籍贯"
        if 'PersonAddresses' in p and 'AddressInfo' in p['PersonAddresses']:
            addrs = p['PersonAddresses']['AddressInfo']
            if isinstance(addrs, list):
                addr_info = addrs[0].get('AddressName', '')
            elif isinstance(addrs, dict):
                addr_info = addrs.get('AddressName', '')

        print(f"[{idx+1}] ID: {person_id} | 姓名: {name}")
        print(f"    ⏳ 年代: {year_info}")
        print(f"    📍 籍贯: {addr_info}")
        print("-" * 40)

# --- 主程序 ---
if __name__ == "__main__":
    # 测试案例：张居正
    target_name = "张居正"
    
    result_data = search_person(target_name)
    
    if result_data:
        # 1. 打印原始 JSON (为了让你看清结构)
        # print("原始数据:", json.dumps(result_data, ensure_ascii=False, indent=2))
        
        # 2. 解析展示
        parse_and_display(result_data)
        
    print("\n💡 提示：如果总是超时，说明你需要代理或者只能使用本地 Mock 数据。")