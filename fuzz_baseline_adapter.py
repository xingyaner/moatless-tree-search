# 文件位置: /root/BuildFixingBaseline/moatless-tree-search/fuzz_baseline_adapter.py

import os
import sys
import time
import yaml
import subprocess
import asyncio
from datetime import datetime

from moatless.message_history import MessageHistoryGenerator
from moatless.schema import MessageHistoryType

# 1. 纯本地路径对齐 (完全隔离)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from moatless.search_tree import SearchTree
from moatless.agent.code_agent import CodingAgent
from moatless.repository.file import FileRepository
# 修正导入：引入 LLMResponseFormat 枚举
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.actions.list_files import ListFiles
from moatless.actions import ViewCode, StringReplace, Finish, Reject
from moatless.actions.fuzz_build import FuzzBuild
from moatless.value_function.base import ValueFunction
from moatless.discriminator import AgentDiscriminator

LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
YAML_PATH = os.path.join(BASE_DIR, "projects.yaml")


# --- 2. 隔离的环境重置工具 ---
def isolated_git_reset(path, sha):
    if not os.path.exists(path):
        print(f"  [Warning] Path not found: {path}")
        return
    print(f"--- [Isolated Reset] path={path}, sha={sha} ---")
    # 强制重置
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=path, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=path, capture_output=True)
    # 切换到指定版本
    res = subprocess.run(["git", "checkout", sha], cwd=path, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [Error] Checkout failed: {res.stderr.strip()}")

def ensure_isolated_repository(path: str, repo_url: str, sha: str):
    """
    确保物理仓库存在并锁定到指定版本。
    """
    if not os.path.exists(path):
        print(f"  [Isolated Clone] Cloning {repo_url} into {path}...")
        # 确保父目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 执行克隆
        clone_res = subprocess.run(["git", "clone", repo_url, path], capture_output=True, text=True)
        if clone_res.returncode != 0:
            print(f"  [CRITICAL ERROR] Clone failed for {repo_url}: {clone_res.stderr}")
            return False

    print(f"  [Isolated Reset] Syncing {path} to SHA: {sha}")
    # 强制重置
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=path, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=path, capture_output=True)
    # 切换到指定 SHA
    res = subprocess.run(["git", "checkout", sha], cwd=path, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [Error] Failed to checkout {sha} in {path}. Error: {res.stderr.strip()}")
        return False
    return True

# --- 3. 核心修复循环 ---
async def run_baseline_cycle(project):
    """
    针对单个项目执行 MCTS 修复周期 (带自动下载、环境锁定与搜索预算校准)。
    """
    p_name = project['project']
    row_index = project['row_index']

    print(f"\n" + "=" * 60)
    print(f"🚀 [BASELINE MCTS] STARTING REPAIR FOR: {p_name}")
    print("=" * 60)

    # --- 1. 环境准备 (自动下载与重置) ---
    oss_fuzz_url = "https://github.com/google/oss-fuzz.git"
    oss_fuzz_path = os.path.join(BASE_DIR, "oss-fuzz")
    project_src_path = os.path.join(BASE_DIR, "project", p_name)

    # 自动准备 OSS-Fuzz 基础设施
    if not ensure_isolated_repository(oss_fuzz_path, oss_fuzz_url, project['oss-fuzz_sha']):
        print(f"--- [ABORT] OSS-Fuzz setup failed for {p_name} ---")
        return

    # 自动准备第三方源代码
    if not ensure_isolated_repository(project_src_path, project['software_repo_url'], project['software_sha']):
        print(f"--- [ABORT] Source code setup failed for {p_name} ---")
        return

    # --- 2. 角色与模型配置 ---
    # 使用 LLMResponseFormat.JSON 开启框架的 JSON 模式，配合我们修改后的适配器
    action_llm = CompletionModel(
        model="deepseek/deepseek-chat",
        temperature=0.2,
        max_tokens=4000,
        response_format=LLMResponseFormat.JSON
    )

    # --- 3. 初始化基线环境 ---
    # 核心：Repository 指向适配器根目录，使 Agent 能通过相对路径看到所有文件
    repository = FileRepository(repo_path=BASE_DIR)

    build_tool = FuzzBuild(
        project_name=p_name,
        oss_fuzz_path=oss_fuzz_path,
        sanitizer=project['sanitizer'],
        engine=project['engine'],
        architecture=project['architecture']
    )

    # 对齐动作空间
    actions = [
        ListFiles(repository=repository),
        ViewCode(
            repository=repository,
            completion_model=action_llm  # 传入 LLM，修复 'NoneType' 报错
        ),
        StringReplace(
            repository=repository,
            completion_model=action_llm  # StringReplace 通常也需要 LLM 验证逻辑
        ),
        build_tool,
        Finish(),
        Reject()
    ]

    # 严格的系统提示词，强制 DeepSeek 遵守格式要求
    strict_system_prompt = (
        "You are a senior software engineer specialized in fixing build errors.\n"
        "GUIDELINES:\n"
        "1. Use 'ListFiles' to explore directory contents. Do NOT use 'ViewCode' on directories.\n"
        "2. Use 'ViewCode' to read specific source files or build scripts.\n"
        "3. After applying changes, you MUST use 'FuzzBuild' to verify your fix.\n"
        "4. ALWAYS respond with a nested JSON object containing 'action_type' and 'action'."
    )

    agent = CodingAgent(
        completion=action_llm,
        actions=actions,
        system_prompt=strict_system_prompt,
        message_generator=MessageHistoryGenerator(
            message_history_type=MessageHistoryType.MESSAGES,
            include_file_context=True
        )
    )

    value_fn = ValueFunction(completion_model=action_llm)
    discriminator = AgentDiscriminator(completion=action_llm, n_agents=5, n_rounds=3)

    # 问题描述：提供物理路径索引，不提供任何诊断引导
    problem_msg = (
        f"You are a senior software engineer fixing a build error for project: {p_name}\n\n"
        f"CRITICAL ENVIRONMENT INFO:\n"
        f"- The OSS-Fuzz configuration (Dockerfile, build.sh) is in: oss-fuzz/projects/{p_name}\n"
        f"- The project source code is in: project/{p_name}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Use 'ListFiles' to explore the directories mentioned above first.\n"
        f"2. Only after locating the file, use 'ViewCode' to read its content.\n"
        f"3. After applying fixes, verify using 'FuzzBuild'.\n"
        f"ALWAYS respond with a nested JSON object containing 'action_type' and 'action'."

        # --- 追加的部分 ---
        f"CRITICAL REPETITION & FEEDBACK POLICY:\n"
        f"- Every action you take will be evaluated with a score (0-100) and feedback.\n"
        f"- If an action receives a score below 50, it means that action is useless or repetitive.\n"
        f"- YOU MUST NOT repeat the same action with the same parameters if the score was low.\n"
        f"- Always check the 'Evaluation' of your previous steps in the history before deciding the next action."
    )

    # --- 4. 运行搜索树 ---
    start_time = time.time()

    def metadata_processor(node):
        """
        核心逻辑：将 MCTS 的评估分数和反馈建议强行注入到节点的观察结果中。
        """
        # 仅处理已经评分且未注入过的节点
        if node.reward is not None and not getattr(node, "_feedback_injected", False):
            # 获取评分和反馈数据
            reward = node.reward
            # 兼容处理 feedback，Discriminator 通常返回字典
            feedback_data = node.feedback if isinstance(node.feedback, dict) else {}
            explanation = feedback_data.get('explanation', 'N/A')
            guidance = feedback_data.get('feedback', 'No specific guidance.')

            # 定义状态标识
            if reward < 30:
                status = "🚨 CRITICAL FAILURE: LOOP DETECTED"
            elif reward < 50:
                status = "⚠️ STAGNANT: NO PROGRESS"
            else:
                status = "✅ PROGRESSING"

            # 构造注入文本
            # 放在 observation 的末尾，确保 LLM 先看到执行结果，再看到评价
            feedback_block = (
                f"\n\n[SYSTEM EVALUATION]\n"
                f"Score: {reward}/100 - {status}\n"
                f"Reasoning: {explanation}\n"
                f"Instruction: {guidance}\n"
                f"Rule: If the score is low, you MUST change your action or parameters. DO NOT repeat the same failed step."
            )

            # 修改 node 对象（物理注入）
            node.observation = (node.observation or "") + feedback_block
            # 标记已注入，防止 MCTS 在回溯时重复累加字符串
            node._feedback_injected = True

    def tree_event_handler(event_type, node=None, data=None):
        """
        增强版事件处理器：支持动态反馈注入。
        """
        if event_type == "search_step" and node:
            # 1. 先打印日志，方便我们在后台观察
            value = getattr(node, "value", "N/A")
            print(f"--- [MCTS Search] Node ID: {node.id} | Value: {value} ---")

            # 2. 调用处理器进行 Prompt 注入
            # 在 search_step 触发时，节点已经完成了评估（Value Function / Discriminator 已运行）
            metadata_processor(node)

        elif event_type == "tree_iteration":
            # 打印迭代概况
            it_data = data or {}
            print(f"--- [Tree Iteration] {it_data.get('iteration')} | Best Reward: {it_data.get('best_reward')} ---")


    # 关键修正：max_iterations = 6 (1 根节点 + 5 次扩展尝试)
    search_tree = SearchTree.create(
        message=problem_msg,
        agent=agent,
        repository=repository,
        value_function=value_fn,
        discriminator=discriminator,
        max_iterations=15,
        max_expansions=1,
        persist_path=os.path.join(LOG_DIR, f"{p_name}_trajectory.json")
    )

    search_tree.event_handlers.append(tree_event_handler)

    best_node = search_tree.run_search()

    # --- 5. 提取指标与持久化 ---
    end_time = time.time()
    # 只有当最好节点是 Finish 时才视为成功
    is_success = best_node.is_finished() if best_node else False
    duration_min = (end_time - start_time) / 60
    usage = search_tree.total_usage()

    # 统计调用构建动作的次数
    repair_rounds = 0
    if search_tree.root:
        for n in search_tree.root.get_all_nodes():
            if getattr(n, 'action_name', '') == 'FuzzBuild':
                repair_rounds += 1

    report = (
        f"============================================================\n"
        f"🏁 FINAL BASELINE REPORT: {p_name}\n"
        f"[RESULT]           {'✅ SUCCESS' if is_success else '❌ FAILURE'}\n"
        f"[DISCUSSION]       YES (5 Agents, 3 Rounds)\n"
        f"[SEARCH DEPTH]     {best_node.get_depth() if best_node else 0}\n"
        f"[REPAIR ROUNDS]    {repair_rounds}\n"
        f"[TOKEN USAGE]      {usage.prompt_tokens + usage.completion_tokens}\n"
        f"[TIME COST]        {duration_min:.2f} minutes\n"
        f"============================================================\n"
    )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{p_name}_{timestamp_str}_baseline.txt"
    with open(os.path.join(LOG_DIR, report_filename), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)

    # 更新 YAML 状态 (完全隔离的原子操作)
    try:
        with open(YAML_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        data[row_index]['state'] = 'yes'
        data[row_index]['fix_result'] = "Success" if is_success else "Failure"
        data[row_index]['fix_date'] = datetime.now().strftime('%Y-%m-%d')
        with open(YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"Status: YAML updated for index {row_index}.")
    except Exception as e:
        print(f"Error updating YAML: {e}")



async def main():
    print("--- [Isolated Baseline Runner Init] ---")
    if not os.path.exists(YAML_PATH):
        print(f"Error: YAML missing at {YAML_PATH}");
        return

    with open(YAML_PATH, 'r') as f:
        projects = yaml.safe_load(f)

    for idx, p in enumerate(projects):
        # 支持字符串 'no' 判定
        if str(p.get('state', 'no')).lower() == 'no':
            p['row_index'] = idx
            await run_baseline_cycle(p)
        else:
            print(f"Skipping {p.get('project')} (already processed or state is yes)")


if __name__ == "__main__":
    asyncio.run(main())