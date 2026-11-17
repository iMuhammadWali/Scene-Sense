# Dataset: Charades-STA (test split)

All experiments use the **Charades-STA test split**: 1,334 videos (about 30 s each, home recordings) and 3,720
sentence queries, each with the start and end time of the moment it describes.

| Part | Size | Where |
|---|---|---|
| Annotations (video, caption, [start, end]) | 84 KB | https://huggingface.co/datasets/lmms-eval/charades_sta (file `data/test-00000-of-00001.parquet`) |
| Videos (Charades v1, 480p) | 16.3 GB zip, but only 2.1 GB is needed | https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip |

The same videos are also on the Hugging Face page above as `Charades_v1_480_part_*.zip` (about 5 GB each).

## Download only the 1,334 test videos (2.1 GB, about 25 minutes)

`scripts/fetch_charades.py` reads just the needed files out of the big zip with HTTP range requests, so the full
16 GB is never downloaded.

```bash
pip install remotezip pandas pyarrow
mkdir -p data/charades_sta/videos
curl -L -o data/charades_sta/test.parquet \
  https://huggingface.co/datasets/lmms-eval/charades_sta/resolve/main/data/test-00000-of-00001.parquet
python scripts/fetch_charades.py data/charades_sta/test.parquet data/charades_sta/videos
```

The script can be stopped and restarted; finished videos are skipped.

## Point the code at it

```bash
export SSEVAL_ANN=$PWD/data/charades_sta/test.parquet
export SSEVAL_VIDEOS=$PWD/data/charades_sta/videos
```

`scripts/encode_all.py` has its three paths (annotations, videos, output folder) at the top of the file; edit them
to match. Encoding every video once with CLIP takes about 2.5 hours on a 2-core CPU (much less on a GPU). The
experiments then run from these cached embeddings in about an hour. The embeddings and videos are not stored in git.

## Credit

Charades videos: Sigurdsson et al., ECCV 2016. Charades-STA sentence annotations: Gao et al., "TALL", ICCV 2017.
