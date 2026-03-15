import os
import sys
import time
import yaml
import subprocess
import asyncio
import traceback
from datetime import datetime
from moatless.actions.simple_view import SimpleViewCode
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


class TeeLogger:
    """将控制台输出同步复制到文件。"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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
    针对单个项目执行 MCTS 修复周期 (最终校准版)。
    集成：物理状态同步、流式日志拦截、反馈物理注入、错误路径防护。
    """
    # 1. 基础信息提取 (对齐您的 YAML 键名)
    p_name = project['project']
    row_index = project['row_index']

    # 初始化起始时间
    start_time = time.time()

    # --- 2. 开启全量日志记录 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = os.path.join(LOG_DIR, f"{p_name}_{timestamp}_full_console.txt")
    original_stdout = sys.stdout
    sys.stdout = TeeLogger(full_log_path)

    try:
        print(f"\n" + "=" * 60)
        print(f"🚀 [BASELINE MCTS] STARTING REPAIR FOR: {p_name}")
        print(f"📍 YAML Index: {row_index} | Engine: {project['engine']}")
        print("=" * 60)

        # --- 3. 物理隔离与环境锁定 ---
        oss_fuzz_url = "https://github.com/google/oss-fuzz.git"
        oss_fuzz_path = os.path.join(BASE_DIR, "oss-fuzz")
        project_src_path = os.path.join(BASE_DIR, "project", p_name)

        # 锁定基础设施版本
        if not ensure_isolated_repository(oss_fuzz_path, oss_fuzz_url, project['oss-fuzz_sha']):
            print(f"--- [ABORT] OSS-Fuzz setup failed for {p_name} ---")
            return

        # 锁定第三方源码版本
        if not ensure_isolated_repository(project_src_path, project['software_repo_url'], project['software_sha']):
            print(f"--- [ABORT] Source code setup failed for {p_name} ---")
            return

        # --- 4. 初始化模型与工具 ---
        action_llm = CompletionModel(
            model="deepseek/deepseek-chat",
            temperature=0.2,
            max_tokens=4000,
            response_format=LLMResponseFormat.JSON
        )

        repository = FileRepository(repo_path=BASE_DIR)

        build_tool = FuzzBuild(
            project_name=p_name,
            oss_fuzz_path=oss_fuzz_path,
            sanitizer=project['sanitizer'],
            engine=project['engine'],
            architecture=project['architecture']
        )

        actions = [
            ListFiles(repository=repository),
            SimpleViewCode(repository=repository),
            StringReplace(repository=repository, completion_model=action_llm),
            build_tool,
            Finish(),
            Reject()
        ]

        # --- 5. 构造 Agent 与提示词 ---
        strict_system_prompt = (
            "You are a senior software engineer specialized in fixing build errors.\n"
            "GUIDELINES:\n"
            "1. Use 'ListFiles' to explore directory contents. Do NOT use 'ViewCode' on directories.\n"
            "2. Use 'SimpleViewCode' to read specific source files or build scripts.\n"
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

        # 构造强力的结果导向型 Problem Message
        problem_msg = (
            f"You are a senior software engineer fixing a build error for project: {p_name}\n\n"
            f"ENVIRONMENT:\n"
            f"- OSS-Fuzz Configs: oss-fuzz/projects/{p_name}\n"
            f"- Target Source: project/{p_name}\n\n"
            f"--- MANDATORY REPAIR PROTOCOL ---\n"
            f"1. IDENTIFY ROOT CAUSE FIRST: Your first priority is to determine the exact reason for the build failure. You MUST identify the specific compiler error (e.g., 'cannot find symbol') or missing dependency by analyzing the 'FuzzBuild' logs or configuration files before attempting to write a patch.\n"
            f"2. EXPLORATION BUDGET: You have a strict budget of 3 to 5 steps for initial exploration (using 'ListFiles' or 'SimpleViewCode'). You must use this time to locate the error source. \n"
            f"3. COMPULSORY ACTION: Once the error is identified or the exploration budget (3-5 steps) is reached, you MUST transition to the 'Modify' phase. This means issuing a 'StringReplace' action followed IMMEDIATELY by 'FuzzBuild' to verify the result.\n"
            f"4. NO MEANINGLESS TESTING: Never execute 'FuzzBuild' without a preceding code modification. The error will not change unless you apply a patch. \n"
            f"5. SUCCESS IS BINARY: Intermediate scores in your history are just hints for the MCTS engine. Your only goal is to make 'FuzzBuild' return SUCCESS. Analysis without a patch is zero progress.\n\n"
            f"Respond with a nested JSON object: {{'action_type': '...', 'action': {{'thoughts': '...', ...}}}}"
        )

        # --- 6. 监控与动态反馈处理器 ---
        def metadata_processor(node):
            """将评分反馈物理注入到上下文，打破循环。"""
            if node.reward and not getattr(node, "_feedback_injected", False):
                score = node.reward.value
                explanation = node.reward.explanation or "N/A"
                # 从 reward 对象中安全获取反馈
                guidance = getattr(node.reward, "feedback", "No specific guidance.")

                status = "✅ PROGRESSING" if score >= 50 else "⚠️ STAGNANT/FAILED"
                if score < 30: status = "🚨 CRITICAL FAILURE: REPETITION OR ERROR"

                feedback_block = (
                    f"\n\n--- [SYSTEM EVALUATION] ---\n"
                    f"Action Score: {score}/100 - {status}\n"
                    f"Reasoning: {explanation}\n"
                    f"Strategic Guidance: {guidance}\n"
                    f"Constraint: If the score is low, DO NOT repeat the same action. Change your path."
                )

                if node.observation:
                    node.observation.message = (node.observation.message or "") + feedback_block
                node._feedback_injected = True

        def tree_event_handler(event_type, node=None, data=None):
            if event_type == "tree_iteration":
                it_data = data or {}
                print(f"\n--- [Iteration {it_data.get('iteration')}] Best Reward: {it_data.get('best_reward')} ---")
            elif event_type == "search_step" and node:
                # 修正：使用 node.node_id
                print(f"--- [Search] Node: {node.node_id} | Depth: {node.get_depth()} ---")
                metadata_processor(node)

        # --- 7. 执行搜索树 ---
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

        print(f"--- [MCTS SEARCH START] Monitoring for loops... ---")
        best_node = search_tree.run_search()

        # --- 8. 统计与持久化报告 ---
        end_time = time.time()
        is_success = best_node.is_finished() if (best_node and hasattr(best_node, 'is_finished')) else False
        duration_min = (end_time - start_time) / 60
        usage = search_tree.total_usage()

        repair_rounds = 0
        if search_tree.root:
            # 遍历 MCTS 树中的所有节点
            for n in search_tree.root.get_all_nodes():
                # 核心修复点：通过框架定义的 .name 属性进行匹配
                # 只有当节点包含动作，且动作逻辑名为 'FuzzBuild' 时计入轮数
                if n.action and n.action.name == 'FuzzBuild':
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
        print(report)

        # 写入物理报告
        with open(os.path.join(LOG_DIR, f"{p_name}_final_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        # 更新 YAML (原子读取并更新)
        with open(YAML_PATH, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        yaml_data[row_index]['state'] = 'yes'
        yaml_data[row_index]['fix_result'] = "Success" if is_success else "Failure"
        yaml_data[row_index]['fix_date'] = datetime.now().strftime('%Y-%m-%d')

        with open(YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)
        print(f"Status: projects.yaml updated for {p_name}.")

    except Exception as e:
        print(f"--- [CRITICAL ERROR in run_baseline_cycle] ---")
        traceback.print_exc()
    finally:
        # 彻底恢复 stdout 避免后续打印丢失
        sys.stdout = original_stdout
        print(f"--- [Project Finish] All logs synced to: {full_log_path} ---")
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