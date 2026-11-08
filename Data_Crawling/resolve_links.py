# -*- coding: utf-8 -*-
"""
Resolve original Weibo post links from report-detail URLs in a CSV.

Input requires URL; output preserves all columns and fills the original-link column. Login is
manual. Each attempted row is saved. Resume uses an existing output only when
its URL sequence matches the input. Run --help for row ranges and paths.
"""

import argparse
from pathlib import Path
from common import OUTPUT, write_table
import os
import time
import pandas as pd
from typing import Optional

from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

# Inclusive data-row range, excluding the CSV header.
START_ROW = 1    # One-based start row; None starts at the first data row.
END_ROW = None     # Inclusive end row; None processes through the last row.

# Paths are set by command-line arguments.
CSV_PATH = ""
DRIVER_PATH = None
OUTPUT_PATH = ""

PAGE_MAX_WAIT_SEC  = 25
LOGIN_MAX_WAIT_SEC = 300
RETRY_PER_URL      = 3
SAVE_PAUSE_SEC     = 0.2

XPATH_ORIGINAL = "//a[normalize-space()='原文' or contains(text(),'原文')]"
XPATH_DETAIL_STABLE = "//*[contains(., '站方判定') or contains(., '被举报微博') or contains(., '举报人信息') or contains(., '举报处理大厅')]"

def launch_browser():
    """Open visible Chrome for manual login."""
    options = ChromeOptions()
    options.add_argument('--start-maximized')
    service = Service(DRIVER_PATH) if DRIVER_PATH else Service()
    return Chrome(service=service, options=options)

def is_logged_in(driver: Chrome) -> bool:
    """Check login indicators using cookies, the current URL, and page text."""
    try:
        cookies = {c.get("name"): c.get("value") for c in driver.get_cookies()}
        if any(k in cookies for k in ("SUB", "SSOLoginState")):
            return True
    except Exception:
        pass
    try:
        WebDriverWait(driver, 1).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(., '举报处理大厅') or contains(., '不实信息')]"))
        )
        return True
    except Exception:
        return False

def wait_until_logged_in(driver: Chrome):
    """Wait for login indicators until the configured timeout expires."""
    start = time.time()
    last_print = 0
    while time.time() - start < LOGIN_MAX_WAIT_SEC:
        if is_logged_in(driver):
            return True
        if time.time() - last_print > 3:
            try:
                print("Waiting for login... Current URL:", driver.current_url)
            except Exception:
                print("Waiting for login...")
            last_print = time.time()
        time.sleep(1.5)
    return False

def ensure_detail_open(driver: Chrome, detail_url: str) -> bool:
    """Open a report page and wait for login or recognizable detail elements."""
    driver.get(detail_url)
    if "login" in driver.current_url.lower():
        print("Login page detected. Waiting for you to sign in...")
        ok = wait_until_logged_in(driver)
        if not ok:
            print("Login wait timed out.")
            return False
        driver.get(detail_url)
    try:
        WebDriverWait(driver, PAGE_MAX_WAIT_SEC).until(
            EC.presence_of_element_located((By.XPATH, XPATH_DETAIL_STABLE))
        )
        return True
    except TimeoutException:
        try:
            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.XPATH, XPATH_ORIGINAL))
            )
            return True
        except TimeoutException:
            return False

def extract_original_url(driver: Chrome) -> Optional[str]:
    """Return the first visible original-post link, or None if unavailable."""
    try:
        WebDriverWait(driver, PAGE_MAX_WAIT_SEC).until(
            EC.presence_of_element_located((By.XPATH, XPATH_ORIGINAL))
        )
        links = driver.find_elements(By.XPATH, XPATH_ORIGINAL)
        for a in links:
            try:
                if a.is_displayed():
                    href = a.get_attribute("href")
                    if href and href.strip():
                        return href.strip()
            except StaleElementReferenceException:
                continue
    except TimeoutException:
        return None
    return None

def atomic_save(df, path):
    """Atomically save the resolved output, preserving the source by default."""
    write_table(df, path)
    time.sleep(SAVE_PAUSE_SEC)

def main():
    """Extract links for selected rows and save the output CSV."""
    global CSV_PATH, OUTPUT_PATH, DRIVER_PATH, START_ROW, END_ROW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=OUTPUT / 'reports.csv')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'resolved_reports.csv')
    parser.add_argument('--start-row', type=int, default=1)
    parser.add_argument('--end-row', type=int)
    parser.add_argument('--driver')
    args = parser.parse_args()
    if args.start_row < 1 or (args.end_row is not None and args.end_row < args.start_row):
        parser.error('Require 1 <= start-row <= end-row')
    if args.input.suffix.lower() != '.csv' or args.output.suffix.lower() != '.csv':
        parser.error('Input and output must be CSV files')
    CSV_PATH, OUTPUT_PATH = str(args.input), str(args.output)
    DRIVER_PATH, START_ROW, END_ROW = args.driver, args.start_row, args.end_row

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV does not exist: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if Path(OUTPUT_PATH).exists() and Path(OUTPUT_PATH).resolve() != Path(CSV_PATH).resolve():
        saved = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if 'URL' not in saved or 'URL' not in df or saved['URL'].tolist() != df['URL'].tolist():
            parser.error('Existing output does not match input URL order; use a new output path')
        if 'post_url' in saved:
            df['post_url'] = saved['post_url']
    if "URL" not in df.columns:
        raise ValueError("The CSV has no 'URL' column. Check the report-title, URL, and report-time headers.")

    if "post_url" not in df.columns:
        df["post_url"] = ""
        atomic_save(df, OUTPUT_PATH)


    total_rows = len(df)
    start = 1 if START_ROW is None else max(1, int(START_ROW))
    end   = total_rows if END_ROW is None else min(total_rows, int(END_ROW))
    # Convert the one-based data-row range to a zero-based pandas slice.
    target_index = df.index[start-1:end]   # Includes the requested end row.
    print(f"Processing rows {start} through {end} ({len(target_index)} rows); total CSV rows: {total_rows}")

    atomic_save(df, OUTPUT_PATH)
    if len(target_index) == 0:
        return
    driver = launch_browser()
    try:
        for pos, idx in enumerate(target_index, start=start):
            row = df.loc[idx]

            # Skip rows that already contain an original-post URL.
            if isinstance(row.get("post_url", ""), str) and row.get("post_url", "").strip():
                print(f"[{pos}] Original-post URL already present; skipping.")
                continue

            detail = (row["URL"] or "").strip() if "URL" in row and pd.notna(row["URL"]) else ""
            if not detail:
                print(f"[{pos}] No report URL; skipping.")
                continue

            print(f"[{pos}] Visiting: {detail}")
            success = False
            for attempt in range(1, RETRY_PER_URL + 1):
                if ensure_detail_open(driver, detail):
                    orig = extract_original_url(driver) or ""
                    df.at[idx, "post_url"] = orig
                    atomic_save(df, OUTPUT_PATH)
                    print(f"  -> Attempt {attempt}: result: {orig if orig else 'not found'} (saved)")
                    success = True
                    break
                else:
                    print(f"  -> Attempt {attempt}: detail page unavailable or not rendered; retrying...")
                    time.sleep(2)

            if not success:
                df.at[idx, "post_url"] = ""
                atomic_save(df, OUTPUT_PATH)
                print("  -> Attempts exhausted; saved a blank value.")

        print(f"Finished. Results saved to: {OUTPUT_PATH}")

    finally:
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
