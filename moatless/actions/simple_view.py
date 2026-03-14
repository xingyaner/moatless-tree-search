import logging
import os
from typing import Literal, Optional
from pydantic import Field, PrivateAttr
from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class SimpleViewCodeArgs(ActionArguments):
    """鲁棒的文件读取工具参数。"""
    thoughts: str = Field(..., description="查看文件的理由。")
    file_path: str = Field(..., description="相对路径。")
    view_mode: Literal["full", "head", "tail"] = Field(default="full")

    class Config:
        title = "SimpleViewCode"


class SimpleViewCode(Action):
    args_schema = SimpleViewCodeArgs
    # 使用 PrivateAttr 存储仓库句柄
    _repository: Repository = PrivateAttr()

    def __init__(self, repository: Repository = None, **data):
        super().__init__(**data)
        self._repository = repository

    def execute(
            self,
            args: SimpleViewCodeArgs,
            file_context=None,
            workspace=None,
    ) -> Observation:
        repo = self._repository or workspace
        if not repo:
            return Observation(message="ERROR: Repository link failed.", summary="internal_error")

        full_path = repo.get_full_path(args.file_path)

        if not os.path.exists(full_path):
            return Observation(message=f"ERROR: File '{args.file_path}' not found.", summary="file_not_found")

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # --- 【核心修复：物理状态同步】 ---
            # 只有更新了 was_viewed，框架才会允许后续的 StringReplace 动作
            if file_context:
                context_file = file_context.get_context_file(args.file_path)
                if not context_file:
                    context_file = file_context.add_file(args.file_path)
                context_file.was_viewed = True
                logger.info(f"SimpleViewCode: Marked {args.file_path} as viewed.")
            # --------------------------------

            # ... (后续行号处理逻辑保持不变)
            total_lines = len(lines)
            start_line = 1
            if args.view_mode == "head":
                display_lines = lines[:300]
            elif args.view_mode == "tail":
                count = min(total_lines, 100)
                display_lines = lines[-count:]
                start_line = max(1, total_lines - count + 1)
            else:
                display_lines = lines

            output = f"--- File: {args.file_path} ---\n"
            for i, line in enumerate(display_lines):
                output += f"{start_line + i:6}\t{line}"

            return Observation(message=output, summary=f"Viewed {args.file_path}")

        except Exception as e:
            return Observation(message=f"ERROR: {str(e)}", summary="read_failure")