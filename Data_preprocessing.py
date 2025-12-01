import requests
from bs4 import BeautifulSoup
import os
import time
import random
from urllib.parse import unquote

# 您提供的原始 URL 列表 (保持原样，代码会自动转换为 zh-cn)
urls = [
    # 皇帝
    "https://zh.wikipedia.org/wiki/朱元璋",
    "https://zh.wikipedia.org/wiki/建文帝",
    "https://zh.wikipedia.org/wiki/明成祖",
    "https://zh.wikipedia.org/wiki/明仁宗",
    "https://zh.wikipedia.org/wiki/明宣宗",
    "https://zh.wikipedia.org/wiki/明英宗",
    "https://zh.wikipedia.org/wiki/景泰帝",
    "https://zh.wikipedia.org/wiki/明宪宗",
    "https://zh.wikipedia.org/wiki/明孝宗",
    "https://zh.wikipedia.org/wiki/明武宗",
    "https://zh.wikipedia.org/wiki/明世宗",
    "https://zh.wikipedia.org/wiki/明穆宗",
    "https://zh.wikipedia.org/wiki/明神宗",
    "https://zh.wikipedia.org/wiki/明光宗",
    "https://zh.wikipedia.org/wiki/明熹宗",
    "https://zh.wikipedia.org/wiki/崇祯帝",
    
    # 开国功臣
    "https://zh.wikipedia.org/wiki/徐达",
    "https://zh.wikipedia.org/wiki/常遇春",
    "https://zh.wikipedia.org/wiki/刘伯温",
    "https://zh.wikipedia.org/wiki/李善长",
    "https://zh.wikipedia.org/wiki/汤和",
    "https://zh.wikipedia.org/wiki/邓愈",
    "https://zh.wikipedia.org/wiki/沐英",
    "https://zh.wikipedia.org/wiki/蓝玉",
    
    # 著名文臣/内阁首辅
    "https://zh.wikipedia.org/wiki/姚广孝",
    "https://zh.wikipedia.org/wiki/于谦",
    "https://zh.wikipedia.org/wiki/王守仁",
    "https://zh.wikipedia.org/wiki/张居正",
    "https://zh.wikipedia.org/wiki/海瑞",
    "https://zh.wikipedia.org/wiki/严嵩",
    "https://zh.wikipedia.org/wiki/徐阶",
    "https://zh.wikipedia.org/wiki/高拱",
    "https://zh.wikipedia.org/wiki/杨廷和",
    "https://zh.wikipedia.org/wiki/李东阳",
    "https://zh.wikipedia.org/wiki/谢迁",
    "https://zh.wikipedia.org/wiki/刘健_(明朝)",
    "https://zh.wikipedia.org/wiki/夏言",
    "https://zh.wikipedia.org/wiki/徐光启",
    "https://zh.wikipedia.org/wiki/杨继盛",
    
    # 著名武将
    "https://zh.wikipedia.org/wiki/戚继光",
    "https://zh.wikipedia.org/wiki/俞大猷",
    "https://zh.wikipedia.org/wiki/袁崇焕",
    "https://zh.wikipedia.org/wiki/熊廷弼",
    "https://zh.wikipedia.org/wiki/孙承宗",
    "https://zh.wikipedia.org/wiki/卢象升",
    "https://zh.wikipedia.org/wiki/李成梁",
    "https://zh.wikipedia.org/wiki/李如松",
    "https://zh.wikipedia.org/wiki/祖大寿",
    "https://zh.wikipedia.org/wiki/吴三桂",
    "https://zh.wikipedia.org/wiki/史可法",
    "https://zh.wikipedia.org/wiki/秦良玉",
    
    # 宦官
    "https://zh.wikipedia.org/wiki/郑和",
    "https://zh.wikipedia.org/wiki/魏忠贤",
    "https://zh.wikipedia.org/wiki/王振_(明朝宦官)",
    "https://zh.wikipedia.org/wiki/刘瑾",
    
    # 思想家/文学家/其他
    "https://zh.wikipedia.org/wiki/唐寅",
    "https://zh.wikipedia.org/wiki/文徵明",
    "https://zh.wikipedia.org/wiki/李贽",
    "https://zh.wikipedia.org/wiki/黄宗羲",
    "https://zh.wikipedia.org/wiki/顾炎武",
    "https://zh.wikipedia.org/wiki/王夫之",
    "https://zh.wikipedia.org/wiki/李时珍",
    "https://zh.wikipedia.org/wiki/宋应星",

    # 农民起义领袖
    "https://zh.wikipedia.org/wiki/李自成",
    "https://zh.wikipedia.org/wiki/张献忠"
]

# 可选：添加历史事件页面（示例）
event_urls = [
    "https://zh.wikipedia.org/wiki/靖难之役",
    "https://zh.wikipedia.org/wiki/土木堡之变",
    "https://zh.wikipedia.org/wiki/夺门之变",
    "https://zh.wikipedia.org/wiki/万历三大征",
    "https://zh.wikipedia.org/wiki/明末农民战争",
    "https://zh.wikipedia.org/wiki/甲申之变"
        # === 建国与制度 ===
    "https://zh.wikipedia.org/wiki/洪武之治",
    "https://zh.wikipedia.org/wiki/胡惟庸案",
    "https://zh.wikipedia.org/wiki/蓝玉案",
    "https://zh.wikipedia.org/wiki/明朝废除丞相制度",
    "https://zh.wikipedia.org/wiki/大明律",
    "https://zh.wikipedia.org/wiki/卫所制度",
    "https://zh.wikipedia.org/wiki/明太祖北伐",

    # === 皇位更迭与政变 ===
    "https://zh.wikipedia.org/wiki/靖难之役",
    "https://zh.wikipedia.org/wiki/土木堡之变",
    "https://zh.wikipedia.org/wiki/北京保卫战",
    "https://zh.wikipedia.org/wiki/夺门之变",
    "https://zh.wikipedia.org/wiki/弘治中兴",
    "https://zh.wikipedia.org/wiki/正德南巡",
    "https://zh.wikipedia.org/wiki/宁王之乱",
    "https://zh.wikipedia.org/wiki/大礼议",
    "https://zh.wikipedia.org/wiki/庚戌之变",

    # === 对外关系与战争 ===
    "https://zh.wikipedia.org/wiki/郑和下西洋",
    "https://zh.wikipedia.org/wiki/嘉靖倭患",
    "https://zh.wikipedia.org/wiki/万历朝鲜之役",
    "https://zh.wikipedia.org/wiki/明缅战争",
    "https://zh.wikipedia.org/wiki/明荷战争",
    "https://zh.wikipedia.org/wiki/澎湖之战_(1622年)",
    "https://zh.wikipedia.org/wiki/料罗湾海战",

    # === 经济与财政改革 ===
    "https://zh.wikipedia.org/wiki/一条鞭法",
    "https://zh.wikipedia.org/wiki/白银货币化",
    "https://zh.wikipedia.org/wiki/三饷",

    # === 政治斗争与党争 ===
    "https://zh.wikipedia.org/wiki/东林党争",
    "https://zh.wikipedia.org/wiki/阉党",
    "https://zh.wikipedia.org/wiki/魏忠贤专政",

    # === 文化、科技与思想 ===
    "https://zh.wikipedia.org/wiki/永乐大典",
    "https://zh.wikipedia.org/wiki/本草纲目",
    "https://zh.wikipedia.org/wiki/天工开物",
    "https://zh.wikipedia.org/wiki/利玛窦",
    "https://zh.wikipedia.org/wiki/西学东渐",
    "https://zh.wikipedia.org/wiki/心学",

    # === 明末危机与灭亡 ===
    "https://zh.wikipedia.org/wiki/明末农民战争",
    "https://zh.wikipedia.org/wiki/松锦大战",
    "https://zh.wikipedia.org/wiki/甲申之变",
    "https://zh.wikipedia.org/wiki/南明",
    "https://zh.wikipedia.org/wiki/扬州十日",
    "https://zh.wikipedia.org/wiki/嘉定三屠",
    "https://zh.wikipedia.org/wiki/李自成攻占北京",
    "https://zh.wikipedia.org/wiki/吴三桂引清兵入关",

    # === 其他重要事件/制度 ===
    "https://zh.wikipedia.org/wiki/厂卫",
    "https://zh.wikipedia.org/wiki/明实录",
    "https://zh.wikipedia.org/wiki/大明会典",
    "https://zh.wikipedia.org/wiki/明长城",
    "https://zh.wikipedia.org/wiki/九边"
]


def scrape_wiki_pages(
    url_list,
    save_folder="ming_dynasty_cn",
    skip_existing=True,
    include_events=False,
    event_url_list=None
):
    """
    爬取维基百科人物或事件页面（简体中文版），保存正文 + 参考文献。
    
    参数:
        url_list (list): 人物页面 URL 列表
        save_folder (str): 保存目录
        skip_existing (bool): 是否跳过已存在的文件
        include_events (bool): 是否额外爬取历史事件
        event_url_list (list or None): 历史事件 URL 列表
    """
    # 创建保存文件夹（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"已创建文件夹: {save_folder}")
    else:
        print(f"使用现有文件夹: {save_folder}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }

    all_urls = list(url_list)
    if include_events and event_url_list:
        all_urls.extend(event_url_list)
        print(f"将额外爬取 {len(event_url_list)} 个历史事件页面。")

    print(f"开始处理 {len(all_urls)} 个页面...\n")

    for raw_url in all_urls:
        try:
            # 转换为简体中文版 URL
            target_url = raw_url.replace("/wiki/", "/zh-cn/")
            name = unquote(raw_url.split("/")[-1])
            file_path = os.path.join(save_folder, f"{name}.txt")

            # 跳过已存在的文件
            if skip_existing and os.path.exists(file_path):
                print(f"跳过（已存在）: {name}")
                continue

            print(f"正在处理: {name} ...", end="", flush=True)

            response = requests.get(target_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                content_div = soup.find('div', {'id': 'mw-content-text'})
                
                if not content_div:
                    print(" [失败: 未找到内容区域]")
                    continue

                # --- 提取正文 ---
                text_content = f"标题: {name}\n来源链接: {target_url}\n"
                text_content += "="*50 + "\n\n"
                
                paragraphs = content_div.find_all('p')
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text:
                        text_content += text + "\n\n"

                # --- 提取参考文献 ---
                text_content += "\n" + "="*20 + " 参考文献 " + "="*20 + "\n\n"
                ref_lists = soup.find_all('ol', class_='references')
                
                ref_count = 1
                found_refs = False
                if ref_lists:
                    for ref_ol in ref_lists:
                        for li in ref_ol.find_all('li'):
                            for jump_link in li.find_all('a', href=True):
                                if "#cite_ref" in jump_link['href']:
                                    jump_link.decompose()
                            ref_text = li.get_text().strip()
                            if ref_text:
                                text_content += f"[{ref_count}] {ref_text}\n"
                                ref_count += 1
                                found_refs = True
                
                if not found_refs:
                    text_content += "（未检测到参考文献列表）\n"

                # --- 保存 ---
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                print(" [成功]")

            else:
                print(f" [失败: HTTP {response.status_code}]")

            # 随机延迟，避免被限
            time.sleep(random.uniform(1.5, 3.5))

        except Exception as e:
            print(f"\n处理 {raw_url} 时发生错误: {e}")

    print("\n所有任务完成！")


if __name__ == "__main__":
    # 示例调用：包含历史事件，跳过已有文件
    scrape_wiki_pages(
        url_list=urls,
        save_folder="ming_dynasty_cn",
        skip_existing=True,
        include_events=True,
        event_url_list=event_urls
    )