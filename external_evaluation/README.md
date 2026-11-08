# External Evaluation (PHEME & Twitter-COVID19)

This directory contains the **code** for evaluating SFTF on two external Twitter
datasets: **PHEME** and **Twitter-COVID19**. The **datasets themselves are not
included** in this repository — see *Why the data is not distributed* below.

## Why the data is not distributed

The datasets are not included in this repository. Obtain the source data from
the official sources below and rebuild the derived features with the scripts in
this directory.

## Data sources

| Dataset | Source |
| ------- | ------ |
| PHEME — fact-checking updates on the rumorous dataset (Zubiaga et al., 2016) | [figshare 6392078](https://doi.org/10.6084/m9.figshare.6392078) |
| Twitter-COVID19 ← ACLR4RUMOR (Lin et al., NAACL 2022) | [github.com/DanielLin97/ACLR4RUMOR-NAACL2022](https://github.com/DanielLin97/ACLR4RUMOR-NAACL2022) |

> - **PHEME**: this project uses the fact-checking-update version (events such as
>   `ferguson`, `ottawashooting`, `sydneysiege`; `record_source=PHEME2018`), with
>   veracity / misinformation / truth annotations.
> - **Twitter-COVID19**: the project's data was derived from ACLR4RUMOR's English
>   Twitter COVID-19 subset and re-formatted into the news / node / parent / child
>   layout with `delay_seconds` relative timestamps and `real` / `fake` labels.
>   The upstream repo (paper *"Detect Rumors in Microblog Posts for Low-Resource
>   Domains via Adversarial Contrastive Learning"*) also ships Weibo-COVID19 and
>   other language COVID-19 subsets.

## Expected directory layout

The scripts expect the following layout. Recreate it from the source data:

```text
PHEME/
├── pheme_bert.py
├── pheme_metapaths.py
├── pheme_sentiment.py
├── pheme_train.py
├── raw_data/          # nodes / edges / paths / news tables
├── sentiment_data/    # per-level sentiment CSVs (news, L1, L2)
├── splits_data/       # real/fake train/val/test splits
├── metapath_data/     # id tables + metapath .txt files
├── bert_feature/      # {real,fake}_{train,val,test}_bert.npy
└── checkpoints/       # pheme_best.pth, pheme_threshold.json

Twitter-COVID19/       # same layout, twitter_* filenames
├── twitter_bert.py
├── twitter_ensemble.py
├── twitter_metapaths.py
├── twitter_sentiment.py
└── twitter_train.py
```

## Rebuilding the derived features

Run the scripts in this order (fuller description in the root README §11):

1. **Sentiment annotation** (requires the `DEEPSEEK_API_KEY` environment variable):

   ```bash
   python pheme_sentiment.py --level 0   # news level
   python pheme_sentiment.py --level 1   # first-level comments
   python pheme_sentiment.py --level 2   # second-level replies
   ```

   Use `twitter_sentiment.py` identically for Twitter-COVID19.

2. **Meta-path construction**:

   ```bash
   python pheme_metapaths.py
   python twitter_metapaths.py
   ```

3. **BERT feature extraction** (requires local pretrained weights, root README §6.4):

   ```bash
   python pheme_bert.py
   python twitter_bert.py
   ```

4. **Training / evaluation**:

   ```bash
   python pheme_train.py
   python twitter_train.py
   python twitter_ensemble.py   # Twitter-COVID19 multi-seed ensemble
   ```

## De-identification before public use

The raw source data contains personal identifiers (author names, profile URLs,
full tweet text). Before any *public* use of derived data, apply the same
HMAC-SHA256 pseudonymization described in the root README §5: replace node /
author IDs with `POST_ID_` / `AUTHOR_` hashes, replace media URLs with
`[MEDIA_URL]`, and drop tweet text, author names, and author / original / repost
URLs.
