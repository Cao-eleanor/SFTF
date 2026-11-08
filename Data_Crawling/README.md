# Data_Crawling

## 1. Overview

The `Data_Crawling` directory contains the Python scripts used in the initial data acquisition stage of the SFTF research project. It provides a step-by-step workflow for collecting Weibo false-information report records, resolving the corresponding original post links, and retrieving posts together with their first-level comments and second-level replies.

The collection process is organized into several independent scripts. Each script performs a specific task and produces an intermediate file that is used by the next step. In addition, an optional script is provided for collecting the historical posts of a specified Weibo account.

## 2. File Descriptions

### 2.1 `common.py`

This file provides shared utilities used by the other collection scripts. Its functions include reading and writing CSV/XLSX files, validating Weibo URLs, checking required table columns, loading authentication cookies, creating HTTP sessions, sending API requests, and generating file-integrity manifests.

`common.py` is a supporting module and does not need to be executed independently.

### 2.2 `fetch_reports.py`

This script collects report records from the Weibo False Information Report Center. After the user completes the login process in the browser, the script navigates to the relevant section, extracts the report title, report-detail URL, and report time, and saves the collected records.

**Run:**

``` bash
python fetch_reports.py
```

**Output:** `outputs/reports.csv`

The output file contains the following fields:

-   `report_title`: report title
-   `URL`: report-detail page URL
-   `report_time`: report timestamp

### 2.3 `resolve_links.py`

This script visits the report-detail pages collected in the previous step and extracts the original Weibo post URL from each page. The resolved URL is saved in the `post_url` column. The script supports processing a specified row range and saves progress during execution.

**Run:**

``` bash
python resolve_links.py \
    --input outputs/reports.csv \
    --output outputs/resolved_reports.csv
```

To process a specific range of records:

``` bash
python resolve_links.py \
    --input outputs/reports.csv \
    --output outputs/resolved_reports.csv \
    --start-row 1 \
    --end-row 500
```

**Output:** `outputs/resolved_reports.csv`

The output preserves the original report fields and adds the `post_url` field containing the resolved original Weibo post URL.

### 2.4 `prepare_data.py`

This script provides local data preparation utilities. It does not access Weibo or require browser login. For the main collection workflow, its `links` command validates and normalizes the resolved post URLs, removes duplicates, and generates a standardized URL list for the subsequent thread collection step.

**Run:**

``` bash
python prepare_data.py links \
    --input outputs/resolved_reports.csv \
    --output outputs/post_links.xlsx
```

**Output:** `outputs/post_links.xlsx`

The generated table contains:

-   `id`: sequential record identifier
-   `post_url`: validated and deduplicated original Weibo post URL

Other available commands include sampling records, filtering comments, standardizing identifiers, modifying URL fields, renaming comment files, converting table formats, and merging yearly files. Run `python prepare_data.py --help` to view the available commands and options.

### 2.5 `fetch_threads.py`

This script collects the original Weibo posts and their associated comment threads. It reads the URL list generated in the previous step, retrieves the post information, collects first-level comments, and then retrieves the second-level replies associated with those comments.

**Run:**

``` bash
python fetch_threads.py \
    --input outputs/post_links.xlsx \
    --url-column post_url \
    --output-dir outputs/threads \
    --cookie-file /path/to/cookie.txt
```

**Main outputs:**

-   `outputs/threads/data.xlsx`: combined post, first-level comment, and second-level reply records
-   `outputs/threads/comment/text_info.xlsx`: collected post information
-   `outputs/threads/comment/{post_id}_1.xlsx`: first-level comments for an individual post
-   `outputs/threads/comment/{post_id}_2.xlsx`: second-level replies for an individual post
-   `outputs/threads/fetch_threads.manifest.json`: collection settings, execution status, and file-integrity information

The script supports configurable request delays and pagination limits. It requires an empty or new output directory for each run to avoid mixing results from different collection runs.

### 2.6 `fetch_timeline.py`

This script provides an optional, independent workflow for collecting posts published by a specified Weibo account during selected months. It retrieves post text, publication time, and interaction counts. It does not collect comment bodies or reply threads.

**Run:**

``` bash
python fetch_timeline.py \
    --uid YOUR_WEIBO_UID \
    --year 2025 \
    --months 1 2 3 \
    --output-dir outputs/timeline \
    --cookie-file /path/to/cookie.txt
```

**Outputs:**

-   Monthly CSV files: `{uid}_{year}_{month}.csv`
-   Combined annual CSV file: `{uid}_{year}_combined.csv`
-   Collection manifest: `fetch_timeline.manifest.json`

This script can be used independently and is not a required step in the main collection workflow.

## 3. Main Data Collection Workflow

The main workflow consists of four sequential steps. Each step uses the output generated by the preceding step.

### Step 1. Collect Report Records

Run `fetch_reports.py` to collect report titles, report-detail URLs, and report timestamps.

**Input:** Weibo False Information Report Center\
**Output:** `outputs/reports.csv`

### Step 2. Resolve Original Post URLs

Run `resolve_links.py` to visit the report-detail pages and extract the corresponding original Weibo post URLs.

**Input:** `outputs/reports.csv`\
**Output:** `outputs/resolved_reports.csv`

### Step 3. Prepare the Post URL List

Run `prepare_data.py links` to validate, normalize, and deduplicate the original post URLs.

**Input:** `outputs/resolved_reports.csv`\
**Output:** `outputs/post_links.xlsx`

### Step 4. Collect Posts and Comment Threads

Run `fetch_threads.py` to retrieve the original posts, first-level comments, and second-level replies.

**Input:** `outputs/post_links.xlsx`\
**Output:** `outputs/threads/`

The overall workflow is summarized below:

``` text
fetch_reports.py
       |
       v
outputs/reports.csv
       |
       v
resolve_links.py
       |
       v
outputs/resolved_reports.csv
       |
       v
prepare_data.py links
       |
       v
outputs/post_links.xlsx
       |
       v
fetch_threads.py
       |
       v
Collected Posts and Comment Threads
```

## 4. Optional Workflow: User Timeline Collection

`fetch_timeline.py` can be executed separately when historical posts from a specified account are required. The user specifies the account UID, target year, and collection months. The script generates monthly CSV files and a combined annual CSV file.

``` text
Weibo Account UID
       |
       v
fetch_timeline.py
       |
       v
Monthly Post Records
       |
       v
Combined Annual CSV
```

## 5. Execution Notes

-   Use Python 3.10 and install dependencies with `pip install -r ../requirements.txt` (or activate the project `.venv`). The crawler scripts require `pandas`, `requests`, `selenium`, and `openpyxl`.
-   All scripts write to the local `outputs/` directory next to these scripts (i.e. `Data_Crawling/outputs/`), not the repository-root `outputs/`.
-   The browser-based scripts `fetch_reports.py` and `resolve_links.py` require manual Weibo login.
-   The API-based scripts `fetch_threads.py` and `fetch_timeline.py` require an authorized Cookie, provided through `--cookie-file` or the `WEIBO_COOKIE` environment variable.
-   Keep authentication cookies and credentials in local private files. Do not commit them to the repository.
-   Run the main workflow in the order described in Section 3.
-   `common.py` is imported by the other scripts and should not be run independently.
-   `fetch_timeline.py` is optional and independent of the main workflow.
-   The collection scripts save intermediate outputs during execution where implemented. For `fetch_threads.py`, use a new or empty output directory for each run.
