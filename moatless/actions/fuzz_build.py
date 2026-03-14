import os
from pydantic import Field
from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation
from moatless.utils.fuzz_utils import run_fuzz_build_standalone


class FuzzBuildArgs(ActionArguments):
    """Execute a black-box build to verify if your current code modifications fixed the error."""
    thoughts: str = Field(..., description="Explain your reasoning for triggering the build at this stage.")

    class Config:
        title = "FuzzBuild"


class FuzzBuild(Action):
    args_schema = FuzzBuildArgs
    # 这些属性将在适配器中被注入
    project_name: str = ""
    oss_fuzz_path: str = ""
    sanitizer: str = ""
    engine: str = ""
    architecture: str = ""

    def execute(self, args: FuzzBuildArgs, file_context=None, workspace=None) -> Observation:
        """执行黑盒构建动作，并实现实时日志流式打印。"""
        import os
        import subprocess

        print(f"\n--- [Isolated Build Started] Project: {self.project_name} ---")

        # 1. 构造 OSS-Fuzz 标准构建命令
        helper_path = os.path.join(self.oss_fuzz_path, "infra/helper.py")
        command = [
            "python3", helper_path, "build_fuzzers",
            self.project_name,
            "--sanitizer", self.sanitizer,
            "--engine", self.engine,
            "--architecture", self.architecture
        ]

        # 确保日志存储目录存在
        log_dir = "fuzz_build_log_file"
        log_file_path = os.path.join(log_dir, "fuzz_build_log.txt")
        os.makedirs(log_dir, exist_ok=True)

        full_log_content = []

        # 2. 使用 Popen 执行并实时捕获输出
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.oss_fuzz_path,
                encoding='utf-8',
                errors='ignore'
            )

            # 实时流式读取并打印到控制台
            for line in process.stdout:
                print(line, end='', flush=True)
                full_log_content.append(line)

            process.wait()
            return_code = process.returncode
            final_log = "".join(full_log_content)

            # 保存完整日志到文件，供后续 Observation 逻辑读取
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(final_log)

            # 3. 判定逻辑
            # 兼容：如果 returncode 不为 0 或日志中没有成功标识，则判定为失败
            is_success = (return_code == 0 and "successfully built" in final_log.lower())

            # 提取末尾 100 行作为给模型的反馈（增加信息密度）
            log_tail = "".join(full_log_content[-100:])

            if is_success:
                return Observation(
                    message="BUILD SUCCESSFUL. All fuzzing targets were built successfully.",
                    summary="Success"
                )
            else:
                return Observation(
                    message=f"BUILD FAILED. Please analyze the raw compiler output below:\n\n{log_tail}",
                    summary="Failure"
                )

        except Exception as e:
            error_msg = f"CRITICAL: Build execution failed: {str(e)}"
            print(error_msg)
            return Observation(message=error_msg, summary="Error")

