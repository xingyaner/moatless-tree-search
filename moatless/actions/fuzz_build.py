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
        """执行黑盒构建动作。"""
        # 调用此前定义的 standalone 工具
        res = run_fuzz_build_standalone(
            self.project_name,
            self.oss_fuzz_path,
            self.sanitizer,
            self.engine,
            self.architecture
        )

        log_content = ""
        log_path = res.get('log_path', "")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                # 仅返回末尾 500 行作为黑盒反馈
                log_content = "".join(lines[-500:])

        if res['status'] == "success":
            return Observation(
                message="BUILD SUCCESSFUL. All fuzzing targets were built successfully.",
                summary="Success"
            )
        else:
            return Observation(
                message=f"BUILD FAILED. Please analyze the following raw compiler output:\n\n{log_content}",
                summary="Failure"
            )