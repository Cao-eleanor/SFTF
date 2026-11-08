# -*- coding: utf-8 -*-
"""
Collect Weibo False Information report titles, detail URLs, and timestamps.

Run with --help for paths and browser settings. Login is manual. Chinese
website labels and CSV headers are preserved for compatibility. Existing
output records are deduplicated by report-detail URL. No posts are classified
or downloaded by this script; use resolve_links.py and fetch_threads.py next.
"""

import argparse
from pathlib import Path
from common import OUTPUT, read_table, write_table, require_columns
import pandas as pd
import csv
import time
import os
from dataclasses import dataclass
from typing import List, Tuple

from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException

# Local ChromeDriver executable.
CHROMEDRIVER_PATH = None

# Website, output, and timeout settings.
WEIBO_HOME = "https://weibo.com/"
REPORT_CENTER = "https://service.account.weibo.com/"
CSV_PATH = ""  # Set from --output in main().
MAX_LOGIN_WAIT_SEC = 300
PAGE_LOAD_TIMEOUT = 45
IMPLICIT_WAIT = 0

# Website selectors: retain the original Chinese page labels.
XPATH_MENU_UNTRUTH = "//a[normalize-space()='不实信息']"
XPATH_TABLE_HEADER = "//table//th"
XPATH_TABLE_ROWS = "//table//tr[td]"
XPATH_NEXT_PAGE = "//a[normalize-space()='下一页' and not(contains(@class,'disabled'))]"

@dataclass
class ReportRow:
    """A report title, detail-page URL, and timestamp as displayed online."""

    title: str
    url: str
    report_time: str


def build_driver() -> Chrome:
    """Create Chrome and configure page-load and element-wait timeouts."""
    opts = ChromeOptions()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    service = Service(CHROMEDRIVER_PATH) if CHROMEDRIVER_PATH else Service()
    driver = Chrome(service=service, options=opts)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.implicitly_wait(IMPLICIT_WAIT)
    return driver


def wait_until_logged_in(driver: Chrome) -> None:
    """Wait for manual login and access to the report center."""
    driver.get(WEIBO_HOME)
    wait = WebDriverWait(driver, 60)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder*='搜索']")))
    except TimeoutException:
        pass

    print(f"Please sign in to Weibo within {MAX_LOGIN_WAIT_SEC//60} minutes...")
    end_at = time.time() + MAX_LOGIN_WAIT_SEC
    while time.time() < end_at:
        try:
            driver.get(REPORT_CENTER)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(.,'举报处理大厅') or contains(.,'举报中心')]"))
            )
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, XPATH_MENU_UNTRUTH))
            )
            print("Login detected. Successfully accessed the report center.")
            return
        except TimeoutException:
            driver.get(WEIBO_HOME)
            time.sleep(3)

    raise TimeoutException("Login timed out: the report center was not accessible within the allotted time.")


def click_untruth_menu(driver: Chrome) -> None:
    """Open the False Information section and wait for its report table."""
    WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, XPATH_MENU_UNTRUTH))
    ).click()
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, XPATH_TABLE_ROWS))
    )


def get_header_index_map(driver: Chrome) -> Tuple[int, int]:
    """Return the one-based positions of the report-title and timestamp columns."""
    headers = driver.find_elements(By.XPATH, XPATH_TABLE_HEADER)
    title_idx = time_idx = None
    for i, th in enumerate(headers, start=1):
        txt = th.text.strip()
        if "举报标题" in txt:
            title_idx = i
        if "举报时间" in txt:
            time_idx = i
    if not title_idx or not time_idx:
        raise RuntimeError("Could not locate the report-title or report-time column. Check the page structure.")
    return title_idx, time_idx


def scrape_current_page(driver: Chrome, title_idx: int, time_idx: int) -> List[ReportRow]:
    """Extract report records from the currently displayed table page."""
    rows: List[ReportRow] = []
    trs = driver.find_elements(By.XPATH, XPATH_TABLE_ROWS)
    for tr in trs:
        try:
            tds = tr.find_elements(By.TAG_NAME, "td")
            if len(tds) < max(title_idx, time_idx):
                continue
            title_cell = tds[title_idx - 1]
            a_tags = title_cell.find_elements(By.TAG_NAME, "a")
            if not a_tags:
                continue
            a = a_tags[0]
            title = a.text.strip()
            url = a.get_attribute("href") or ""
            time_text = tds[time_idx - 1].text.strip()
            rows.append(ReportRow(title=title, url=url, report_time=time_text))
        except StaleElementReferenceException:
            continue
    return rows


def go_next_page_if_possible(driver: Chrome) -> bool:
    """Attempt to open the next page; return False if unavailable or timed out."""
    try:
        old_rows = driver.find_elements(By.XPATH, XPATH_TABLE_ROWS)
        next_btn = driver.find_element(By.XPATH, XPATH_NEXT_PAGE)
        driver.execute_script("arguments[0].click();", next_btn)
        if old_rows:
            WebDriverWait(driver, 20).until(EC.staleness_of(old_rows[0]))
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, XPATH_TABLE_ROWS))
        )
        time.sleep(0.8)
        return True
    except NoSuchElementException:
        return False
    except TimeoutException:
        raise TimeoutException("Next-page navigation timed out; output may be incomplete")


# Report title, report-detail URL, and report time (original CSV schema).
HEADER = ["report_title", "URL", "report_time"]

def ensure_csv_header(path: str) -> None:
    """Write a UTF-8 BOM header when the output file is missing or empty."""
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

def append_rows(path, page_rows):
    """Append records without duplicating report-detail URLs."""
    existing = read_table(path) if Path(path).exists() else pd.DataFrame(columns=HEADER)
    require_columns(existing, HEADER)
    incoming = pd.DataFrame([[r.title, r.url, r.report_time] for r in page_rows], columns=HEADER)
    result = pd.concat([existing, incoming], ignore_index=True).drop_duplicates('URL', keep='first')
    write_table(result, path)


def main():
    """Collect report pages, save each page, and always close the browser."""
    global CHROMEDRIVER_PATH, CSV_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "reports.csv")
    parser.add_argument("--driver", help="Optional ChromeDriver executable; otherwise use Selenium Manager")
    args = parser.parse_args()
    CHROMEDRIVER_PATH = args.driver
    CSV_PATH = str(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")
    driver = build_driver()
    try:
        wait_until_logged_in(driver)

        if not driver.current_url.startswith(REPORT_CENTER):
            driver.get(REPORT_CENTER)

        click_untruth_menu(driver)
        title_idx, time_idx = get_header_index_map(driver)


        ensure_csv_header(CSV_PATH)

        page_no = 1
        total = 0
        while True:
            print(f"Collecting page {page_no}...")
            page_rows = scrape_current_page(driver, title_idx, time_idx)


            append_rows(CSV_PATH, page_rows)
            total += len(page_rows)
            print(f"Saved {len(page_rows)} records from page {page_no}; total: {total}.")

            if go_next_page_if_possible(driver):
                page_no += 1
                continue
            else:
                break

        print(f"Finished. Processed {total} records (output is deduplicated). CSV: {CSV_PATH}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
