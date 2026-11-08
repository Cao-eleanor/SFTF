# `model/data/` — Weibo-25 de-identified dataset

This directory holds the **de-identified** Weibo-25 dataset used to train and
evaluate SFTF: news posts, first-/second-level comments, DeepSeek sentiment
annotations, metapath features, and BERT embeddings.

> 🔒 The data here is **already HMAC-pseudonymized**: original user/profile URLs
> and platform IDs have been replaced with `POST_` / `AUTHOR_` / `REPOST_` /
> `POST_ID_` hashes, media URLs with `[MEDIA_URL]`, and nicknames + raw text have
> been removed. The HMAC key is **internal-only** and is never distributed, so the
> pseudonymization is effectively one-way for recipients.

## Structure

| Subdirectory | Contents |
|---|---|
| `raw_data/` | De-identified news + comment tables (`real|fake_news.xlsx`, `real|fake_comments.xlsx`) |
| `sentiment_data/` | DeepSeek sentiment tables (L0 news-level, L1/L2 comment-level) |
| `metapath_data/` | Emotion metapath tables (`.xlsx`) + time-series `.txt` files |
| `splits_data/` | Train/val/test splits (`real|fake_{train,val,test}.csv`, news-only id tables) |
| `bert_feature/` | BERT sentence embeddings (`.npy`, 768-dim) |

> Note: the `raw_data/` directory name is kept for historical reasons — the tables it holds are already de-identified (pseudonymized), not raw plaintext.

## De-identification

The de-identification policy applied here is HMAC-SHA256 pseudonymization of
UIDs/URLs (→ `POST_` / `AUTHOR_` / `REPOST_` / `POST_ID_` hashes) plus dropping of
nicknames and raw text; media URLs are replaced with `[MEDIA_URL]`.

> The pseudonymization key is **internal-only** and must never be distributed with
> the data; it is excluded from version control (`.gitignore`).

> Note on `preprocessing/encode_ids.py`: that script performs a separate
> *integer factorization* of an already-pseudonymized URL/id column
> (`pd.factorize`) for downstream node encoding. It is **not** the HMAC
> de-identification step — that step is applied out-of-repo and is not part of
> the published code.
