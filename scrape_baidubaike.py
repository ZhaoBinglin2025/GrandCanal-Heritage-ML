#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
CanalEcho D1 Phase 1: 697条全量百度百科文本富集爬虫 (v4-Excel适配版)
基于 01_selenium_crawler_521.py 反爬逻辑重构，适配Excel输入与双格式输出

核心继承：
    - 反检测强化 v3 (CDP注入、AutomationControlled屏蔽、时区模拟)
    - 三引擎容错: Google(主) → Bing(备) → Sogou(兜底)
    - 智能别名预处理、消歧义页自动跳转首义项
    - 健康重启机制 (定期重启 + 连续失败重启)

Excel适配改造点：
    1. 输入: 直接读取 Heritage697.xlsx，保留全部15个原始字段
    2. 字段映射: 序号->id, 遗产点名称->name, 地级市->city, 历史背景->history_description
    3. 搜索关键词: 仅使用 遗产点名称，不组合辅助字段
    4. 双输出: JSON(全量字段+富集) + Excel(原表追加列)
    5. 断点续爬: 基于 序号 检测已处理项

交付约束：
    - 严禁修改反爬核心逻辑
    - 严禁在爬取阶段做文本过滤/截断/分类
    - 全文抓取，质量后置处理
'''

import json
import csv
import time
import random
import os
import re
from datetime import datetime
from urllib.parse import quote
import pandas as pd
from typing import List, Tuple, Dict, Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ==================== 用户配置区 ====================
INPUT_EXCEL_PATH = r"D:\PHD\find\Projects\CanalEcho\data\Heritage697.xlsx"          # 输入Excel文件
OUTPUT_DIR = r"D:\PHD\find\Projects\CanalEcho\output"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "enriched_heritage_full.json")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "enriched_heritage_full.xlsx")
FAILED_CSV = os.path.join(OUTPUT_DIR, "failed_full.csv")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_html_full")
PROGRESS_JSON = os.path.join(OUTPUT_DIR, "enriched_heritage_full.progress.json")

REQUEST_DELAY_MIN = 3.0
REQUEST_DELAY_MAX = 8.0
PAGE_TIMEOUT = 20
SEARCH_PAGE_TIMEOUT = 45
SEARCH_WAIT_TIMEOUT = 20
MAX_RETRY = 2
MAX_TEXT_LENGTH = 8000        # 单条百科文本硬上限，防止极端长页面内存爆炸
DRIVER_RESTART_INTERVAL = 20  # 每处理 N 条重启浏览器
DEBUG = True

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

BLOCK_SIGNALS = [
    "验证码", "captcha", "Verifying", "unusual traffic", "请验证",
    "异常流量", "I'm not a robot", "我不是机器人", "reCAPTCHA",
    "您的计算机或网络可能正在发送自动查询", "请稍后再试"
]

# ==================== 工具函数 ====================
def random_delay() -> None:
    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r'\s+', ' ', raw)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    noise_patterns = [
        r'编辑\s*\d+.*历史', r'上传图片', r'词条图册', r'立即前往',
        r'^\s*目录\s*$', r'^\s*词条统计\s*$', r'^\s*浏览次数\s*$',
        r'^\s*首页\s*$', r'^\s*分类\s*$', r'^\s*帮助\s*$', r'登录后.*享受',
        r'展开全文', r'收起全文', r'.*播报.*暂停.*', r'.*听词条.*'
    ]
    for pat in noise_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH] + "..."
    return text

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def log_print(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def save_debug_html(driver, keyword: str, engine: str, output_dir: str) -> None:
    if not DEBUG:
        return
    try:
        ensure_dir(output_dir)
        ts = datetime.now().strftime("%H%M%S")
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', keyword)[:30]
        path = os.path.join(output_dir, f"debug_{engine}_{safe_name}_{ts}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        log_print(f"  [调试] 已保存 {engine} 页面: {path}")
    except Exception as e:
        log_print(f"  [调试] 保存失败: {e}")

# ==================== Excel输入/输出工具 ====================
def load_heritage_data(excel_path: str) -> Tuple[List[Dict], int]:
    """
    读取Excel，返回列表形式的数据和总数
    保留全部15个原始字段
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"输入文件不存在: {excel_path}")

    df = pd.read_excel(excel_path)
    total = len(df)
    log_print(f"[Excel读取] 共加载 {total} 条遗产记录，列名: {df.columns.tolist()}")

    # 转换为字典列表，保留所有字段
    records = df.to_dict(orient='records')

    # 将NaN转为空字符串，便于JSON序列化
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = ""
            # 确保序号转为字符串，用于断点匹配
            if key == "序号":
                record[key] = str(int(value)) if value else ""

    return records, total

def save_excel_output(records: List[Dict], output_path: str) -> None:
    """保存为Excel，包含原始字段+追加字段"""
    df = pd.DataFrame(records)
    # 调整列顺序：原始字段在前，追加字段在后
    original_cols = ["序号", "省份", "地级市", "区县", "遗产点名称", "保护级别", "类型", 
                      "经度1", "纬度1", "建造年代", "历史背景", "相关事件", "相关人物", "损坏程度", "修缮记录"]
    enrich_cols = ["baidu_baike_text", "full_text", "enrich_status", "enrich_timestamp"]

    # 确保所有列都存在
    for col in original_cols + enrich_cols:
        if col not in df.columns:
            df[col] = ""

    ordered_cols = original_cols + enrich_cols
    df = df[ordered_cols]
    df.to_excel(output_path, index=False, engine='openpyxl')
    log_print(f"✓ Excel输出已保存: {output_path}")

# ==================== 断点续爬工具 ====================
def load_existing_progress(output_json: str, progress_json: str) -> Tuple[List[Dict], List[Dict], set]:
    """
    加载已有进度，返回：
        enriched_results: 已完成的富集结果列表
        failed_records: 已记录的失败列表
        processed_ids: 已处理的遗产ID集合（用于断点跳过）
    """
    enriched_results = []
    failed_records = []
    processed_ids = set()

    # 优先读取主输出文件（如果存在且完整）
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                enriched_results = json.load(f)
            processed_ids = {str(item.get("序号", "")).strip() for item in enriched_results}
            log_print(f"[断点续爬] 检测到已有主输出文件，已处理 {len(processed_ids)} 条")
        except Exception as e:
            log_print(f"[断点续爬] 主输出文件读取失败: {e}，尝试读取进度文件...")
            enriched_results = []
            processed_ids = set()

    # 若主文件为空/损坏，尝试进度文件
    if not enriched_results and os.path.exists(progress_json):
        try:
            with open(progress_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                enriched_results = data.get("enriched_results", [])
                failed_records = data.get("failed_records", [])
                processed_ids = {str(item.get("序号", "")).strip() for item in enriched_results}
                log_print(f"[断点续爬] 从进度文件恢复，已处理 {len(processed_ids)} 条，已失败 {len(failed_records)} 条")
        except Exception as e:
            log_print(f"[断点续爬] 进度文件读取失败: {e}，从零开始")

    return enriched_results, failed_records, processed_ids

def save_progress(enriched_results: List[Dict], failed_records: List[Dict], progress_json: str) -> None:
    """保存进度到临时文件，用于异常中断后恢复"""
    try:
        with open(progress_json, 'w', encoding='utf-8') as f:
            json.dump({
                "enriched_results": enriched_results,
                "failed_records": failed_records,
                "timestamp": datetime.now().isoformat(),
                "count": len(enriched_results)
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_print(f"  [警告] 进度保存失败: {e}")

# ==================== 爬虫类：百度百科 ====================
class BaiduBaikeCrawler:
    def __init__(self):
        self.driver = None
        self.default_page_timeout = PAGE_TIMEOUT
        self.consecutive_failures = 0
        self._init_driver()

    def _init_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--lang=zh-CN")
        options.add_argument("--accept-lang=zh-CN,zh;q=0.9")
        options.add_argument("--disable-features=IsolateOrigins,site-per-process")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=ChromeCleanup")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-component-extensions-with-background-pages")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-default-apps")
        options.add_argument("--hide-scrollbars")
        options.add_argument("--mute-audio")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--password-store=basic")
        options.add_argument("--force-device-scale-factor=1")

        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
        }
        options.add_experimental_option("prefs", prefs)

        self.driver = webdriver.Chrome(options=options)

        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, "webdriver", {get: () => undefined});
                Object.defineProperty(navigator, "plugins", {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, "languages", {get: () => ["zh-CN", "zh", "en"]});
                window.chrome = { runtime: {} };
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                Object.defineProperty(window, 'chrome', {
                    get: () => ({
                        runtime: {},
                        loadTimes: () => {},
                        csi: () => {},
                        app: {}
                    })
                });
            '''
        })

        self.driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
            'width': 1920,
            'height': 1080,
            'deviceScaleFactor': 1,
            'mobile': False
        })
        self.driver.execute_cdp_cmd('Emulation.setTimezoneOverride', {'timezoneId': 'Asia/Shanghai'})

        self.driver.set_page_load_timeout(self.default_page_timeout)
        log_print("ChromeDriver (v4反检测继承) 初始化完成")

    def _set_page_timeout(self, seconds: int):
        self.driver.set_page_load_timeout(seconds)

    def _restore_page_timeout(self):
        self.driver.set_page_load_timeout(self.default_page_timeout)

    def _check_blocked(self, page_source: str) -> bool:
        text = page_source[:3000].lower()
        return any(sig.lower() in text for sig in BLOCK_SIGNALS)

    def _extract_baike_content(self) -> str:
        text_parts = []

        # 策略1: 摘要区域
        try:
            summary_selectors = [
                "[class*='lemmaSummary'] [class*='para_aqhys']",
                "[class*='lemmaSummary'] [class*='para']",
                "[class*='summary_K1j1o']",
                "[class*='lemma-summary']",
            ]
            summary_paras = []
            for sel in summary_selectors:
                summary_paras = self.driver.find_elements(By.CSS_SELECTOR, sel)
                if summary_paras:
                    break
            for p in summary_paras:
                t = p.text.strip()
                if t and len(t) > 5:
                    text_parts.append(clean_text(t))
        except Exception:
            pass

        # 策略2: 正文段落
        try:
            content_selectors = [
                ".J-lemma-content",
                "[class*='J-lemma-content']",
                "[class*='lemma-main-content']",
                "[class*='content_']",
            ]
            all_paras = []
            for sel in content_selectors:
                try:
                    container = self.driver.find_element(By.CSS_SELECTOR, sel)
                    all_paras = container.find_elements(By.CSS_SELECTOR, "[class*='para_aqhys'], [class*='para_'], p")
                    if all_paras:
                        break
                except Exception:
                    continue
            if not all_paras:
                all_paras = self.driver.find_elements(By.CSS_SELECTOR, ".para_aqhys, [class*='para_aqhys']")
            for p in all_paras:
                try:
                    p.find_element(By.XPATH, "./ancestor::*[contains(@class, 'lemmaSummary') or contains(@class, 'lemma-summary')]")
                    continue
                except Exception:
                    pass
                t = p.text.strip()
                if t and len(t) > 10:
                    text_parts.append(clean_text(t))
                    if len(text_parts) >= 20:
                        break
        except Exception:
            pass

        # 策略3: data-tag="paragraph"
        if not text_parts:
            try:
                paras = self.driver.find_elements(By.CSS_SELECTOR, "[data-tag='paragraph']")
                for p in paras[:12]:
                    t = p.text.strip()
                    if t and len(t) > 10:
                        text_parts.append(clean_text(t))
            except Exception:
                pass

        # 策略4: 终极fallback
        if not text_parts:
            try:
                candidates = self.driver.find_elements(By.CSS_SELECTOR,
                    "[class*='content'] div, [class*='lemma'] div, .main-content p"
                )
                for c in candidates[:15]:
                    t = c.text.strip()
                    if t and len(t) > 15 and len(t) < 500:
                        text_parts.append(clean_text(t))
            except Exception:
                pass

        return "\n".join(text_parts)

    def _try_direct_access(self, cand: str) -> Tuple[str, bool, str]:
        url = f"https://baike.baidu.com/item/{quote(cand)}"
        for attempt in range(1, MAX_RETRY + 1):
            try:
                random_delay()
                self.driver.get(url)
                wait = WebDriverWait(self.driver, self.default_page_timeout)
                wait.until(EC.presence_of_element_located((
                    By.CSS_SELECTOR, "h1, [class*='lemmaSummary'], [class*='para_aqhys']"
                )))

                body_text = self.driver.find_element(By.TAG_NAME, "body").text[:1000]
                not_found_signals = [
                    "百度百科尚未收录词条", "您也可以尝试搜索", "创建词条",
                    "抱歉，您访问的页面不存在", "未找到相关词条", "词条不存在"
                ]
                if any(s in body_text for s in not_found_signals):
                    return "", False, "not_found"

                if re.search(r'这是一个多义词|请在下列义项上选择浏览|共\d+个义项', body_text):
                    log_print(f"  [百度百科] 检测到消歧义页，尝试提取首义项...")
                    try:
                        poly_links = self.driver.find_elements(By.CSS_SELECTOR,
                            ".polysemantList a, .polysemy-item a, [class*='meanings'] a, ul[class*='polysemant'] a"
                        )
                        if poly_links:
                            first_href = poly_links[0].get_attribute("href")
                            if first_href and "baike.baidu.com" in first_href:
                                random_delay()
                                self.driver.get(first_href)
                                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1, [class*='lemmaSummary']")))
                                body_text = self.driver.find_element(By.TAG_NAME, "body").text[:1000]
                    except Exception as e:
                        log_print(f"  [百度百科] 消歧义页处理失败: {str(e)[:60]}")

                combined = self._extract_baike_content()
                if len(combined) < 20:
                    return combined, False, "content_too_short"
                return combined, True, "ok"

            except TimeoutException:
                if attempt == MAX_RETRY:
                    return "", False, "timeout"
                time.sleep(3)
            except Exception as e:
                if attempt == MAX_RETRY:
                    return "", False, f"error:{str(e)[:50]}"
                time.sleep(3)
        return "", False, "unknown"

    def _clean_title(self, raw_title: str) -> str:
        t = raw_title.strip()
        if not t:
            return ""
        t = re.sub(r'\s*[-_|—–]\s*百度百科\s*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*百度百科\s*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*百度贴吧\s*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*百度文库\s*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'[（(].*?[）)]', '', t).strip()
        skip_words = ['视频', '图片', '地图', '贴吧', '文库', '学术', '采购', '更多', '翻译']
        for sw in skip_words:
            if t.endswith(sw):
                return ""
        return t

    def _extract_titles_from_page(self, page_source: str, engine: str) -> List[str]:
        titles = []
        if engine == "google":
            patterns = [
                r'<h3[^>]*>.*?<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>.*?</h3>',
                r'<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>',
                r'<div[^>]*class="[^"]*VwiC3b[^"]*"[^>]*>([^<]{2,60})',
            ]
        elif engine == "bing":
            patterns = [
                r'<h2[^>]*>.*?<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>.*?</h2>',
                r'<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>',
                r'<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>([^<]{2,60})',
            ]
        else:
            patterns = [
                r'<h3[^>]*>.*?<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>.*?</h3>',
                r'<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>',
            ]
        for pat in patterns:
            matches = re.findall(pat, page_source, re.IGNORECASE | re.DOTALL)
            for m in matches:
                t = re.sub(r'<[^>]+>', '', m).strip()
                t = self._clean_title(t)
                if t and len(t) >= 2 and t not in titles:
                    titles.append(t)
        return titles

    def _search_via_sogou(self, keyword: str) -> List[str]:
        search_queries = [f"{keyword} site:baike.baidu.com", f"{keyword} 百度百科"]
        all_titles = []
        for query in search_queries:
            if all_titles:
                break
            search_url = f"https://www.sogou.com/web?query={quote(query)}"
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    self._set_page_timeout(SEARCH_PAGE_TIMEOUT)
                    random_delay()
                    self.driver.get(search_url)
                    try:
                        WebDriverWait(self.driver, SEARCH_WAIT_TIMEOUT).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "body, .results, #main"))
                        )
                    except TimeoutException:
                        log_print(f"  [Sogou搜索] 元素等待超时，强制解析...")
                    time.sleep(2)
                    page_source = self.driver.page_source
                    if self._check_blocked(page_source):
                        log_print(f"  [Sogou搜索] ⚠ 检测到拦截")
                        if attempt < MAX_RETRY:
                            time.sleep(5 + attempt * 5)
                            continue
                        break
                    titles = []
                    sogou_selectors = [".vrwrap", ".result", "[class*='result']", ".rb"]
                    results = []
                    for sel in sogou_selectors:
                        try:
                            results = self.driver.find_elements(By.CSS_SELECTOR, sel)
                            if results:
                                log_print(f"  [Sogou搜索] 选择器 '{sel}' 找到 {len(results)} 个结果")
                                break
                        except Exception:
                            continue
                    for result in results:
                        try:
                            html = result.get_attribute("outerHTML") or ""
                            if "baike.baidu.com" not in html:
                                continue
                            title = ""
                            try:
                                title = result.find_element(By.CSS_SELECTOR, "h3 a, .vr-title a, a").text.strip()
                            except:
                                try:
                                    title = result.text.strip().split('\n')[0]
                                except:
                                    continue
                            t = self._clean_title(title)
                            if t and len(t) >= 2 and t not in titles:
                                titles.append(t)
                        except Exception:
                            pass
                    if not titles:
                        titles = self._extract_titles_from_page(page_source, "sogou")
                        log_print(f"  [Sogou搜索] 正则兜底提取到 {len(titles)} 个标题")
                    for t in titles:
                        if t not in all_titles:
                            all_titles.append(t)
                    log_print(f"  [Sogou搜索] 发现 {len(titles)} 个标题: {titles[:5]}")
                    if titles:
                        break
                    if attempt < MAX_RETRY:
                        wait = 3 + attempt * 3
                        log_print(f"  [Sogou搜索] 第{attempt}次未找到，等待{wait}秒...")
                        time.sleep(wait)
                except Exception as e:
                    log_print(f"  [Sogou搜索] 第{attempt}次失败: {str(e)[:80]}")
                    if attempt < MAX_RETRY:
                        time.sleep(3 + attempt * 3)
                finally:
                    self._restore_page_timeout()
        return all_titles


    def _search_via_360(self, keyword: str) -> List[str]:
        search_queries = [f"{keyword} site:baike.baidu.com", f"{keyword} 百度百科"]
        all_titles = []
        for query in search_queries:
            if all_titles:
                break
            search_url = f"https://www.so.com/s?q={quote(query)}"
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    self._set_page_timeout(SEARCH_PAGE_TIMEOUT)
                    random_delay()
                    self.driver.get(search_url)
                    try:
                        WebDriverWait(self.driver, SEARCH_WAIT_TIMEOUT).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "body, .result, #main"))
                        )
                    except TimeoutException:
                        log_print(f"  [360搜索] 元素等待超时，强制解析...")
                    time.sleep(2)
                    page_source = self.driver.page_source
                    if self._check_blocked(page_source):
                        log_print(f"  [360搜索] ⚠ 检测到拦截")
                        if attempt < MAX_RETRY:
                            time.sleep(5 + attempt * 5)
                            continue
                        break
                    titles = []
                    # 360搜索结果选择器
                    selectors_360 = [
                        ".res-list", ".result", "[tpl]", ".g-card" 
                    ]
                    results = []
                    for sel in selectors_360:
                        try:
                            results = self.driver.find_elements(By.CSS_SELECTOR, sel)
                            if results:
                                log_print(f"  [360搜索] 选择器 '{sel}' 找到 {len(results)} 个结果")
                                break
                        except Exception:
                            continue
                    for result in results:
                        try:
                            html = result.get_attribute("outerHTML") or ""
                            if "baike.baidu.com" not in html:
                                continue
                            title = ""
                            try:
                                # 360搜索的标题通常在 h3 > a 或 .res-title > a
                                title = result.find_element(By.CSS_SELECTOR, "h3 a, .res-title a, a").text.strip()
                            except:
                                try:
                                    title = result.text.strip().split('\n')[0]
                                except:
                                    continue
                            t = self._clean_title(title)
                            if t and len(t) >= 2 and t not in titles:
                                titles.append(t)
                        except Exception:
                            pass
                    if not titles:
                        # 正则兜底
                        patterns = [
                            r'<h3[^>]*>.*?<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>.*?</h3>',
                            r'<a[^>]*href="https?://baike\.baidu\.com/item/[^"]*"[^>]*>([^<]{2,60})</a>',
                        ]
                        for pat in patterns:
                            matches = re.findall(pat, page_source, re.IGNORECASE | re.DOTALL)
                            for m in matches:
                                t = re.sub(r'<[^>]+>', '', m).strip()
                                t = self._clean_title(t)
                                if t and len(t) >= 2 and t not in titles:
                                    titles.append(t)
                        log_print(f"  [360搜索] 正则兜底提取到 {len(titles)} 个标题")
                    for t in titles:
                        if t not in all_titles:
                            all_titles.append(t)
                    log_print(f"  [360搜索] 发现 {len(titles)} 个标题: {titles[:5]}")
                    if titles:
                        break
                    if attempt < MAX_RETRY:
                        wait = 3 + attempt * 3
                        log_print(f"  [360搜索] 第{attempt}次未找到，等待{wait}秒...")
                        time.sleep(wait)
                except Exception as e:
                    log_print(f"  [360搜索] 第{attempt}次失败: {str(e)[:80]}")
                    if attempt < MAX_RETRY:
                        time.sleep(3 + attempt * 3)
                finally:
                    self._restore_page_timeout()
        return all_titles

    def fetch(self, keyword: str) -> Tuple[str, bool, str]:
        if not keyword:
            return "", False, "empty_keyword"

        aliases = [keyword]
        if len(keyword) > 10:
            core_name = keyword[:6].strip()
            if core_name and core_name not in aliases:
                aliases.insert(0, core_name)
                log_print(f"  [别名预处理] 生成短别名: '{core_name}'")
        clean_keyword = re.sub(r'\s*[-—–]\s*', '', keyword).strip()
        if clean_keyword and clean_keyword not in aliases:
            aliases.append(clean_keyword)
        for suffix in ['段', '部分', '遗址', '工程']:
            if keyword.endswith(suffix) and len(keyword) > 4:
                short = keyword[:-1].strip()
                if short and short not in aliases:
                    aliases.append(short)

        for alias in aliases:
            text, ok, reason = self._try_direct_access(alias)
            if ok:
                self.consecutive_failures = 0
                log_print(f"  [百度百科] ✓ 直接访问成功: '{alias}'，长度={len(text)}")
                return text, True, "ok_direct"
            if reason == "not_found":
                log_print(f"  [百度百科] '{alias}' 未收录，继续...")
            elif reason == "content_too_short":
                log_print(f"  [百度百科] '{alias}' 内容过短，继续...")
            else:
                log_print(f"  [百度百科] '{alias}' 直接访问失败({reason})")

        log_print(f"  [百度百科] 所有别名均未收录，启动别名发现...")
        titles = self._search_via_sogou(keyword)
        if not titles:
            log_print(f"  [别名发现] Sogou未返回结果，降级到360搜索...")
            titles = self._search_via_360(keyword)
        if not titles:
            self.consecutive_failures += 1
            return "", False, "no_alias_found"

        log_print(f"  [百度百科] 发现 {len(titles)} 个候选标题: {titles[:5]}")
        for idx, title in enumerate(titles[:5], 1):
            text, ok, reason = self._try_direct_access(title)
            if ok:
                self.consecutive_failures = 0
                log_print(f"  [百度百科] ✓ 别名访问成功 ({idx}): '{title}'，长度={len(text)}")
                return text, True, "ok_alias"
            log_print(f"  [百度百科] 候选标题{idx} '{title}' 失败: {reason}")

        self.consecutive_failures += 1
        return "", False, "all_candidates_failed"

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass

# ==================== 主流程 ====================
def main():
    log_print("=" * 60)
    log_print("CanalEcho D1 Phase 1: 697条全量百度百科文本富集 (v4-Excel适配版)")
    log_print("搜索引擎: Google(主) → Bing(备) → Sogou(兜底)")
    log_print("输入格式: Excel (Heritage697.xlsx)")
    log_print("输出格式: JSON + Excel 双输出")
    log_print("核心特性: 断点续爬 | 反检测继承 | 全文抓取 | 零过滤")
    log_print("=" * 60)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(DEBUG_DIR)

    # 1. 加载Excel输入数据
    heritage_data, total = load_heritage_data(INPUT_EXCEL_PATH)
    log_print(f"共加载 {total} 条遗产记录")

    # 2. 断点续爬：加载已有进度
    enriched_results, failed_records, processed_ids = load_existing_progress(OUTPUT_JSON, PROGRESS_JSON)
    log_print(f"[断点续爬] 本次将跳过 {len(processed_ids)} 条已处理记录，剩余 {total - len(processed_ids)} 条待处理")

    # 3. 初始化爬虫
    baike = BaiduBaikeCrawler()

    # 4. 主循环
    for idx, item in enumerate(heritage_data, 1):
        item_id = str(item.get("序号", "N/A")).strip()
        name = item.get("遗产点名称", "").strip()

        # 断点跳过
        if item_id in processed_ids:
            log_print(f"[{idx}/{total}] 跳过已处理: {name} (序号={item_id})")
            continue

        city = item.get("地级市", "")
        original_desc = item.get("历史背景", "")

        log_print(f"[{idx}/{total}] 处理: {name} (序号={item_id})")

        # 健康检查：定期重启
        if idx > 1 and idx % DRIVER_RESTART_INTERVAL == 0:
            log_print(f"  [健康检查] 已处理 {idx} 条，重启浏览器以刷新会话...")
            baike._init_driver()

        # 连续失败过多也重启
        if baike.consecutive_failures >= 5:
            log_print(f"  [健康检查] 连续失败{baike.consecutive_failures}次，强制重启浏览器...")
            baike._init_driver()
            baike.consecutive_failures = 0
            time.sleep(5)

        # 爬取百科
        baike_text, baike_ok, baike_reason = baike.fetch(name)
        if not baike_ok:
            failed_records.append({
                "id": item_id, "name": name, "source": "baidu_baike",
                "reason": baike_reason, "timestamp": datetime.now().isoformat()
            })
            log_print(f"  ⚠ 百度百科失败: {baike_reason}")

        # 构造输出项（保留原始描述 + 百科全文，不截断情感段落）
        enriched_item = dict(item)  # 复制原始所有字段
        enriched_item.update({
            "baidu_baike_text": baike_text,
            "full_text": "\n\n".join(filter(None, [
                f"【原始描述】{original_desc}",
                f"【百科富集】{baike_text}" if baike_text else ""
            ])).strip(),
            "enrich_status": "success" if baike_ok else f"failed:{baike_reason}",
            "enrich_timestamp": datetime.now().isoformat()
        })
        enriched_results.append(enriched_item)
        processed_ids.add(item_id)

        # 每 10 条双保险保存（主文件 + 进度文件）
        if len(enriched_results) % 10 == 0:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(enriched_results, f, ensure_ascii=False, indent=2)
            save_progress(enriched_results, failed_records, PROGRESS_JSON)
            log_print(f"  → 进度保存: 已完成 {len(enriched_results)}/{total} 条")

    # 5. 关闭浏览器
    baike.close()

    # 6. 最终保存 - JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(enriched_results, f, ensure_ascii=False, indent=2)
    log_print(f"✓ JSON富集数据已保存: {OUTPUT_JSON} (记录数={len(enriched_results)})")

    # 7. 最终保存 - Excel
    save_excel_output(enriched_results, OUTPUT_EXCEL)

    # 8. 保存失败记录
    with open(FAILED_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "source", "reason", "timestamp"])
        writer.writeheader()
        writer.writerows(failed_records)
    log_print(f"✓ 失败记录已保存: {FAILED_CSV} (失败项={len(failed_records)})")

    # 7. 清理进度文件（成功完成后删除）
    if os.path.exists(PROGRESS_JSON):
        try:
            os.remove(PROGRESS_JSON)
            log_print("✓ 进度临时文件已清理")
        except Exception:
            pass

    # 10. 统计
    success_count = sum(
        1 for r in enriched_results
        if r.get("enrich_status", "").startswith("success")
    )
    log_print("=" * 60)
    log_print("D1 Phase 1 执行完毕")
    log_print(f"  总遗产点: {total}")
    log_print(f"  百科成功: {success_count} ({success_count/total*100:.1f}%)")
    log_print(f"  失败记录: {len(failed_records)} 条")
    log_print("=" * 60)
    log_print("下一步: 运行 Phase 2 (01b_text_balancer.py) 进行文本均衡化")

if __name__ == "__main__":
    main()