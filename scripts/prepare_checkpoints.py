import argparse
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIP_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/"
    "ViT-L-14-336px.pt"
)
BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


def download_url(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"exists: {path}")
        return
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, path)
    print(f"saved: {path}")


def prepare_clip():
    download_url(BPE_URL, ROOT / "CLIP" / "bpe_simple_vocab_16e6.txt.gz")
    download_url(CLIP_URL, ROOT / "CLIP" / "ckpt" / "ViT-L-14-336px.pt")


def prepare_conch():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    out_dir = ROOT / "conch" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "pytorch_model_vision.bin"
    if target.exists():
        print(f"exists: {target}")
        return

    src = hf_hub_download(
        repo_id="MahmoodLab/conchv1_5",
        filename="pytorch_model_vision.bin",
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    if Path(src) != target:
        os.replace(src, target)
    print(f"saved: {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="store_true", help="Download OpenAI CLIP files.")
    parser.add_argument("--conch", action="store_true", help="Download gated CONCH v1.5 weights.")
    parser.add_argument("--all", action="store_true", help="Prepare CLIP and CONCH.")
    args = parser.parse_args()

    if not (args.clip or args.conch or args.all):
        args.clip = True

    if args.clip or args.all:
        prepare_clip()
    if args.conch or args.all:
        prepare_conch()


if __name__ == "__main__":
    main()
