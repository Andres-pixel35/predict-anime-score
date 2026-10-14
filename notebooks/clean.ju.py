# %% [md]
"""
# Cleaning the anime dataset

Takes the raw dataset and leaves only what the model is allowed to see: information
that is known **before** an anime is released, plus the label (`mean_score`).

The same notebook cleans both splits, chosen with the `SPLIT` environment variable:

| `SPLIT` | reads | writes | content |
|---|---|---|---|
| `train` (default) | `data/train/anime.csv` | `data/train/anime_clean.csv` | everything released before 2026 |
| `test` | `data/test/test.csv` | `data/test/test_clean.csv` | only 2026 releases |

Nothing here is learned from the data (no medians, no counts, no thresholds). Every step
that depends on the train set lives in the pipeline of `train.ju.py`.
"""
# %% [md]
"""
## Setup
"""
# %%
import pandas as pd
import unicodedata
import re
import numpy as np
import os
SPLIT = os.environ.get("SPLIT", "train")  # "train" or "test"
pd.set_option("display.max_colwidth", None)
# %%
# A missing season is written in the raw file as the string "None None"
df = pd.read_csv(f"../data/{SPLIT}/{'anime' if SPLIT == 'train' else 'test'}.csv", na_values="None None")
# %% [md]
"""
## Dropping columns

Three kinds of columns are removed:

- **Post-release information**: `score`, `rank`, `popularity`, `members`, `favorites`,
  `watching`, `completed`... They only exist once people have watched the anime, so
  using them to predict the score of a future anime would be cheating.
- **Identifiers, links and metadata**: ids, images, trailer, urls, collection dates.
- **Free text**: titles, synonyms, synopsis, hashtag, characters.

`english_title` is kept for now, only to build `has_english_title` later.
"""
# %%
df.drop(
    columns=
        [
        "anime_id", "mal_id", "broadcast_day", "broadcast_time", "cover_image_large",
        "cover_image_color", "banner_image", "trailer_site", "trailer_id", "main_characters",
        "site_url", "updated_at", "collected_at", "data_source", "rating", "scored_by",
        "score", "rank", "popularity_rank", "members", "favorites", "watching", "popularity",
        "completed", "on_hold", "dropped", "plan_to_watch", "start_date", "end_date", "synopsis",
        "is_licensed", "hashtag", "synonyms", "title", "japanese_title", "user_preferred_title"
    ],
    inplace=True
)
# %% [md]
"""
## Filtering rows

Kept entries are:

- not music videos,
- already finished,
- from Japan,
- released before 2026 for train, released in 2026 for test.

Splitting by year makes the test set a real "future anime" scenario: the model never
sees a 2026 anime while training.
"""
# %%
df_f = df[
    ~(df["type"].eq("MUSIC")) &
    (df["status"].eq("FINISHED")) &
    (df["season_year"].eq(2026) if SPLIT == "test" else ~df["season_year"].eq(2026)) &
    (df["country_of_origin"].eq("JP"))
]
# %% [md]
"""
## Normalizing text columns

### Staff roles

`main_staff` comes as `Name (Role);Name (Role)`. The role is removed and only the name
is kept.
"""
# %%
df_f["main_staff"] = (
    df_f["main_staff"]
    .str.replace(
        r"\([^)]*\)", "", regex=True
    )
    .str.strip()
)
# %% [md]
"""
### List columns

Studios, genres, tags, streaming sites and staff are lists separated by `;`. Each name
is normalized so the same thing written in two ways ends up as a single value:

- `clean_string`: unicode normalization, lowercase and only letters/digits
  (`"Slice of Life"` -> `"sliceoflife"`).
- `clean_list`: applies it to every name of the list, removes duplicates and empty
  names, and sorts them. An empty list becomes nan.
"""
# %%
def clean_string(name):
    name = unicodedata.normalize("NFKC", name)
    name = name.lower()
    name = re.sub(r"[\W_]+", "", name)
    return name

def clean_list(s):
    if pd.isna(s):
        return np.nan
    names = {clean_string(n) for n in s.split(";")} - {""}
    return ";".join(sorted(names)) if names else np.nan

df_f["main_studios_clean"] = df_f["main_studios"].map(clean_list)
df_f["genres_clean"] = df_f["genres"].map(clean_list)
df_f["tags_clean"] = df_f["tags"].map(clean_list)
df_f["streaming_sites_clean"] = df_f["streaming_sites"].map(clean_list)
df_f["main_staff"] = df_f["main_staff"].map(clean_list)
df_f["studios"] = df_f["studios"].map(clean_list)

# season comes as "SPRING 1970", the year is already in season_year
df_f["season"] = df_f["season"].str.split(" ").str[0]
# %% [md]
"""
### Dropping and renaming

`status` and `country_of_origin` have a single value after the filter, and the raw list
columns were replaced by their `_clean` version.
"""
# %%
df_f.drop(
    columns=[
        "status", "country_of_origin", "main_studios",
        "genres", "tags", "streaming_sites"
    ],
    inplace=True
)
# %%
df_f.rename(
    columns={
        "duration": "episode_duration_min",
        "season": "release_season",
        "season_year": "release_year",
        "main_studios_clean": "main_studios",
        "genres_clean": "genres",
        "tags_clean": "tags",
        "streaming_sites_clean": "streaming_sites"
    },
    inplace=True
)
# %% [md]
"""
### Streaming sites with two names

The same site appears under two names, they are merged into one.
"""
# %%
df_f["streaming_sites"] = (
    df_f["streaming_sites"]
    .str.replace(r"(^|;)bilibilitv(;|$)", r"\1bilibili\2", regex=True)
    .str.replace(r"(^|;)amazonprimevideo(;|$)", r"\1primevideo\2", regex=True)
)
# %% [md]
"""
## Treating nan values
"""
# %%
df_f.isna().sum()
# %% [md]
"""
### Manual fixes

Values checked by hand, one by one, so each split only gets the fixes that were
verified for it.
"""
# %%
if SPLIT == "train":
    # There is only one nan value, it is a movie and I verified it at MAL, only one episode
    df_f["episodes"] = df_f["episodes"].fillna(1)
    # Same entry as the previous one, it is the only one with nan in these columns.
    # I verified its start date: Aug 13, 2025
    df_f.loc[df_f["release_year"].isna(), ["release_season", "release_year"]] = ["SUMMER", 2025]
elif SPLIT == "test":
    # There are two nan values, both are movies and I searched them, only one episode each
    df_f["episodes"] = df_f["episodes"].fillna(1)
# %%
# A missing source becomes its own category
df_f["source"] = df_f["source"].fillna("UNKNOWN")
# %% [md]
"""
### Columns turned into simpler features

- `english_title` -> `has_english_title` (1/0): the title itself is not useful, but
  whether it has one might be.
- `streaming_sites` -> `n_streaming_sites`: how many sites, nan counts as 0.
  This number may increase once the anime is released, so it might carry some
  post-release information.
- `studios` -> `n_studios_involved`: how many studios, nan counts as 0.
  The names of the main studios are still kept in `main_studios`.
"""
# %%
df_f["has_english_title"] = df_f["english_title"].notna().astype(int)

df_f["is_adult"] = df_f["is_adult"].astype(int)

# number of names in the list = number of ";" + 1
for col, new in [("streaming_sites", "n_streaming_sites"), ("studios", "n_studios_involved")]:
    df_f[new] = df_f[col].str.count(";").add(1).fillna(0).astype(int)

df_f.drop(columns=["english_title", "streaming_sites", "studios"], inplace=True)
# %% [md]
"""
### Removing the hentai genre

`genre_hentai` and `is_adult` have a Spearman correlation of 0.98, they carry the same
information. Dropping `genre_hentai` barely changed the cross validation RMSE
(≈7.176 -> ≈7.177) while dropping `is_adult` made it worse (≈7.3), so `is_adult` stays.
"""
# %%
# hentai genre carries the same information that is_adult
df_f["genres"] = df_f["genres"].map(
    lambda s: ";".join(g for g in s.split(";") if g != "hentai") or np.nan
    if isinstance(s, str) else s
)
# %% [md]
"""
### What is still nan

The remaining nan values are left on purpose, they are handled inside the pipeline:

- `episode_duration_min`: filled with the **train** median of its type.
- `main_staff`, `main_studios`, `genres`, `tags`: a nan is an empty list for the encoders.
"""
# %%
df_f.isna().sum()
# %% [md]
"""
## Saving
"""
# %%
df_f.to_csv(f"../data/{SPLIT}/{'anime' if SPLIT == 'train' else 'test'}_clean.csv", encoding="utf-8", index=False)
