from pathlib import Path

def ensure_output_dir(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def get_output_filename(source_path_or_url):
    # 处理本地路径和URL，生成合适的md文件名
    if source_path_or_url.startswith("http://") or source_path_or_url.startswith("https://"):
        name = source_path_or_url.rstrip("/").split("/")[-1]
        if "." in name:
            name = name[: name.rfind(".")]
    else:
        name = Path(source_path_or_url).stem
    return f"{name}.md" 