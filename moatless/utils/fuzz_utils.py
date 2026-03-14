import os
import subprocess
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def run_fuzz_build_standalone(project_name, oss_fuzz_path, sanitizer, engine, architecture):
    """
    独立执行构建命令。仅返回原始日志路径和状态。
    """
    log_dir = os.path.join(os.getcwd(), "fuzz_build_log_file")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "fuzz_build_log.txt")

    helper_path = os.path.join(oss_fuzz_path, "infra/helper.py")
    # 构造 OSS-Fuzz 标准构建命令
    command = [
        "python3", helper_path, "build_fuzzers",
        project_name,
        "--sanitizer", sanitizer,
        "--engine", engine,
        "--architecture", architecture
    ]

    try:
        print(f"Executing: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=oss_fuzz_path,
            encoding='utf-8',
            errors='ignore'
        )

        full_log = []
        for line in process.stdout:
            full_log.append(line)

        process.wait()
        final_log_str = "".join(full_log)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(final_log_str)

        # 判定标准：返回码为 0 且包含成功关键字
        success = (process.returncode == 0) and ("successfully built" in final_log_str.lower())
        return {"status": "success" if success else "error", "log_path": log_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def git_force_reset(path):
    """物理回退：清除所有未提交的修改和新增文件。"""
    if not os.path.exists(path):
        return
    try:
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=path, capture_output=True, check=True)
        subprocess.run(["git", "clean", "-fdx"], cwd=path, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Git reset failed at {path}: {e}")


def git_checkout(path, sha):
    """版本锁定：切换到元数据指定的 SHA。"""
    if not os.path.exists(path):
        return
    try:
        subprocess.run(["git", "checkout", str(sha)], cwd=path, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Git checkout failed at {path} for SHA {sha}: {e}")


def read_baseline_yaml(file_path):
    """读取待处理项目列表，过滤掉已尝试的项目。"""
    if not os.path.exists(file_path):
        print(f"Error: YAML file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    projects = []
    for idx, entry in enumerate(data):
        # 严格检查 state 字段
        if str(entry.get('state', 'no')).lower() == 'no':
            entry['row_index'] = idx
            projects.append(entry)
    return projects


def update_baseline_yaml(file_path, row_index, result):
    """标记项目状态，插入 state, fix_result, fix_date 三行元数据。"""
    if not os.path.exists(file_path):
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if 0 <= row_index < len(data):
        data[row_index]['state'] = 'yes'
        data[row_index]['fix_result'] = result
        data[row_index]['fix_date'] = datetime.now().strftime('%Y-%m-%d')

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Updated YAML index {row_index} with result: {result}")