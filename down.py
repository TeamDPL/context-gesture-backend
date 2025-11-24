from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir="./hf_models/clip-vit-base-patch32",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="google/owlvit-base-patch16",
    local_dir="./hf_models/owlvit-base-patch16",
    local_dir_use_symlinks=False
)
