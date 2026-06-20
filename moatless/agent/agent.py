import importlib
import json
import logging
import traceback
from typing import List, Type, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError

from moatless.actions.action import Action
from moatless.actions.model import (
    ActionArguments,
    Observation,
)
from moatless.agent.settings import AgentSettings
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.model import Completion
from moatless.exceptions import RuntimeError, CompletionRejectError
from moatless.index.code_index import CodeIndex
from moatless.message_history import MessageHistoryGenerator
from moatless.node import Node, ActionStep
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)


class ActionAgent(BaseModel):
    system_prompt: str = Field(
        ..., description="System prompt to be used for generating completions"
    )
    use_few_shots: bool = Field(
        True, description="Whether to use few-shot examples for generating completions"
    )
    thoughts_in_action: bool = Field(True, description="")
    actions: List[Action] = Field(default_factory=list)
    mandatory_patch_verification: bool = Field(
        default=False,
        description="Whether to force at least one patch and one verification action before completion.",
    )
    exploration_budget: int = Field(
        default=0,
        description="How many executed steps are allowed before gating exploration actions.",
    )
    patch_action_names: List[str] = Field(
        default_factory=lambda: ["StringReplace"],
        description="Action names that count as applying a patch.",
    )
    verification_action_names: List[str] = Field(
        default_factory=lambda: ["FuzzBuild"],
        description="Action names that count as verification.",
    )
    force_patch_after_steps: int = Field(
        default=0,
        description="Force at least one patch after this many executed steps in the current trajectory.",
    )
    force_verify_after_steps: int = Field(
        default=0,
        description="Force verification after this many executed steps once a patch exists.",
    )
    fail_if_no_patch_by_steps: int = Field(
        default=0,
        description="Fail the node if no patch has been applied by this many executed steps.",
    )
    fail_if_no_verification_by_steps: int = Field(
        default=0,
        description="Fail the node if no verification has been executed by this many steps after patching.",
    )
    message_generator: MessageHistoryGenerator = Field(
        default_factory=lambda: MessageHistoryGenerator(),
        description="Generator for message history",
    )

    _completion: CompletionModel = PrivateAttr()
    _action_map: dict[Type[ActionArguments], Action] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        completion: CompletionModel,
        system_prompt: str | None = None,
        actions: List[Action] | None = None,
        message_generator: MessageHistoryGenerator | None = None,
        **data,
    ):
        actions = actions or []
        message_generator = message_generator or MessageHistoryGenerator()
        super().__init__(
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            **data,
        )
        self.set_actions(actions)
        self._completion = completion

    @classmethod
    def from_agent_settings(
        cls, agent_settings: AgentSettings, actions: List[Action] | None = None
    ):
        if agent_settings.actions:
            actions = [
                action
                for action in actions
                if action.__class__.__name__ in agent_settings.actions
            ]

        return cls(
            completion=agent_settings.completion_model,
            system_prompt=agent_settings.system_prompt,
            actions=actions,
        )

    def set_actions(self, actions: List[Action]):
        self.actions = actions
        self._action_map = {action.args_schema: action for action in actions}

    @model_validator(mode="after")
    def verify_actions(self) -> "ActionAgent":
        for action in self.actions:
            if not isinstance(action, Action):
                raise ValidationError(
                    f"Invalid action type: {type(action)}. Expected Action subclass."
                )
            if not hasattr(action, "args_schema"):
                raise ValidationError(
                    f"Action {action.__class__.__name__} is missing args_schema attribute"
                )
        return self

    def run(self, node: Node):
        """Run the agent on a node to generate and execute an action."""

        if node.action:
            logger.info(f"Node{node.node_id}: Resetting node")
            node.reset()

        gate_failure = self._get_gate_failure(node)
        if gate_failure:
            print(
                f"[DIAG ActionAgent] Node {node.node_id} gate failure before action generation: {gate_failure}",
                flush=True,
            )
            node.terminal = True
            node.error = gate_failure
            node.observation = Observation(message=gate_failure, terminal=True)
            logger.warning(f"Node{node.node_id}: {gate_failure}")
            return

        available_actions = self._get_available_actions(node)
        node.possible_actions = [action.name for action in available_actions]
        trajectory = node.get_trajectory()
        executed_action_names = [
            step.action.name
            for trajectory_node in trajectory
            for step in trajectory_node.action_steps
            if step.action
        ]
        has_patch = any(name in self.patch_action_names for name in executed_action_names)
        has_verification = any(
            name in self.verification_action_names for name in executed_action_names
        )
        print(
            "[DIAG ActionAgent] "
            f"Node {node.node_id} available actions={node.possible_actions} | "
            f"executed_steps={len(executed_action_names)} | "
            f"has_patch={has_patch} | has_verification={has_verification} | "
            f"mandatory_patch_verification={self.mandatory_patch_verification}",
            flush=True,
        )
        if "StringReplace" not in node.possible_actions:
            print(
                f"[DIAG ActionAgent] Node {node.node_id} StringReplace is NOT available in this step.",
                flush=True,
            )
        else:
            print(
                f"[DIAG ActionAgent] Node {node.node_id} StringReplace is available for selection.",
                flush=True,
            )
        system_prompt = self.generate_system_prompt()
        action_args = [action.args_schema for action in available_actions]

        messages = self.message_generator.generate(node)
        logger.info(f"Node{node.node_id}: Build action with {len(messages)} messages")
        try:
            completion_response = self._completion.create_completion(
                messages, system_prompt=system_prompt, response_model=action_args
            )

            if completion_response.structured_outputs:
                node.action_steps = [
                    ActionStep(action=action)
                    for action in completion_response.structured_outputs
                ]
                selected_actions = [action.name for action in completion_response.structured_outputs]
                print(
                    f"[DIAG ActionAgent] Node {node.node_id} model selected actions={selected_actions}",
                    flush=True,
                )
                if "StringReplace" in node.possible_actions and "StringReplace" not in selected_actions:
                    print(
                        f"[DIAG ActionAgent] Node {node.node_id} model skipped StringReplace despite it being available.",
                        flush=True,
                    )

            node.assistant_message = completion_response.text_response

            node.completions["build_action"] = completion_response.completion
        except Exception as e:
            node.terminal = True
            node.error = traceback.format_exc()

            if hasattr(e, "messages") and hasattr(e, "last_completion"):
                # TODO: Move mapping to completion.py
                node.completions["build_action"] = Completion.from_llm_completion(
                    input_messages=e.messages,
                    completion_response=e.last_completion,
                    model=self.completion.model,
                )
                logger.warning(
                    f"Node{node.node_id}: Build action failed with error: {e}"
                )
                return
            else:
                raise e

        if node.action is None:
            print(
                f"[DIAG ActionAgent] Node {node.node_id} produced no executable action after completion parsing.",
                flush=True,
            )
            return

        duplicate_node = node.find_duplicate()
        if duplicate_node:
            node.is_duplicate = True
            print(
                f"[DIAG ActionAgent] Node {node.node_id} is duplicate of Node {duplicate_node.node_id}; execution skipped.",
                flush=True,
            )
            logger.info(
                f"Node{node.node_id} is a duplicate to Node{duplicate_node.node_id}. Skipping execution."
            )
            return

        logger.info(f"Node{node.node_id}: Execute {len(node.action_steps)} actions")
        for action_step in node.action_steps:
            self._execute(node, action_step)

    def _get_available_actions(self, node: Node) -> List[Action]:
        if not self.mandatory_patch_verification:
            print(
                f"[DIAG ActionAgent] Node {node.node_id} patch gating disabled; returning all actions.",
                flush=True,
            )
            return self.actions

        trajectory = node.get_trajectory()
        executed_action_names = [
            step.action.name
            for trajectory_node in trajectory
            for step in trajectory_node.action_steps
            if step.action
        ]

        executed_steps = len(executed_action_names)
        has_patch = any(name in self.patch_action_names for name in executed_action_names)
        has_verification = any(
            name in self.verification_action_names for name in executed_action_names
        )

        if self.exploration_budget and executed_steps < self.exploration_budget:
            print(
                "[DIAG ActionAgent] "
                f"Node {node.node_id} within exploration budget: executed_steps={executed_steps}, "
                f"exploration_budget={self.exploration_budget}. Returning all actions.",
                flush=True,
            )
            return self.actions

        if self.force_patch_after_steps and executed_steps >= self.force_patch_after_steps and not has_patch:
            gated_names = set(self.patch_action_names)
            gated_actions = [action for action in self.actions if action.name in gated_names]
            if gated_actions:
                print(
                    "[DIAG ActionAgent] "
                    f"Node {node.node_id} forcing patch-only actions after {executed_steps} steps. "
                    f"Allowed now={[action.name for action in gated_actions]}",
                    flush=True,
                )
                logger.info(
                    f"Node{node.node_id}: Forcing patch-only actions after {executed_steps} steps"
                )
                return gated_actions

        if (
            self.force_verify_after_steps
            and executed_steps >= self.force_verify_after_steps
            and has_patch
            and not has_verification
        ):
            gated_names = set(self.patch_action_names + self.verification_action_names)
            gated_actions = [action for action in self.actions if action.name in gated_names]
            if gated_actions:
                print(
                    "[DIAG ActionAgent] "
                    f"Node {node.node_id} forcing patch/verification actions after patch application. "
                    f"Allowed now={[action.name for action in gated_actions]}",
                    flush=True,
                )
                logger.info(
                    f"Node{node.node_id}: Forcing verification-capable actions after patch application"
                )
                return gated_actions

        print(
            "[DIAG ActionAgent] "
            f"Node {node.node_id} no patch/verification gate applied. Returning all actions.",
            flush=True,
        )
        return self.actions

    def _get_gate_failure(self, node: Node) -> str | None:
        if not self.mandatory_patch_verification:
            return None

        trajectory = node.get_trajectory()
        executed_action_names = [
            step.action.name
            for trajectory_node in trajectory
            for step in trajectory_node.action_steps
            if step.action
        ]

        executed_steps = len(executed_action_names)
        has_patch = any(name in self.patch_action_names for name in executed_action_names)
        has_verification = any(
            name in self.verification_action_names for name in executed_action_names
        )

        if self.fail_if_no_patch_by_steps and executed_steps >= self.fail_if_no_patch_by_steps and not has_patch:
            return (
                f"Mandatory patch phase failed: no patch action was executed by step {executed_steps}."
            )

        if (
            self.fail_if_no_verification_by_steps
            and executed_steps >= self.fail_if_no_verification_by_steps
            and has_patch
            and not has_verification
        ):
            return (
                f"Mandatory verification phase failed: no verification action was executed by step {executed_steps}."
            )

        return None

    def _execute(self, node: Node, action_step: ActionStep):
        action = self._action_map.get(type(action_step.action))
        if not action:
            logger.error(
                f"Node{node.node_id}: Action {node.action.name} not found in action map. "
                f"Available actions: {self._action_map.keys()}"
            )
            raise RuntimeError(f"Action {type(node.action)} not found in action map.")

        try:
            print(
                f"[DIAG ActionAgent] Node {node.node_id} executing action={action_step.action.name}",
                flush=True,
            )
            action_step.observation = action.execute(
                action_step.action, node.file_context, node.workspace
            )
            if not action_step.observation:
                logger.warning(
                    f"Node{node.node_id}: Action {action_step.action.name} returned no observation"
                )
            else:
                node.terminal = action_step.observation.terminal
                if action_step.observation.execution_completion:
                    action_step.completion = (
                        action_step.observation.execution_completion
                    )

            logger.info(
                f"Executed action: {action_step.action.name}. "
                f"Terminal: {action_step.observation.terminal if node.observation else False}. "
                f"Output: {action_step.observation.message if node.observation else None}"
            )
            if action_step.observation:
                fail_reason = None
                if action_step.observation.properties:
                    fail_reason = action_step.observation.properties.get("fail_reason")
                print(
                    "[DIAG ActionAgent] "
                    f"Node {node.node_id} action={action_step.action.name} summary={action_step.observation.summary} | "
                    f"expect_correction={action_step.observation.expect_correction} | "
                    f"fail_reason={fail_reason}",
                    flush=True,
                )

        except CompletionRejectError as e:
            logger.warning(f"Node{node.node_id}: Action rejected: {e.message}")
            action_step.completion = e.last_completion
            action_step.observation = Observation(
                message=e.message,
                is_terminal=True,
            )

    def generate_system_prompt(self) -> str:
        """Generate a system prompt for the agent."""

        system_prompt = self.system_prompt
        if self.use_few_shots:
            system_prompt += "\n\n" + self.generate_few_shots()

        return system_prompt

    def generate_few_shots(self) -> str:
        few_shot_examples = []
        for action in self.actions:
            examples = action.get_few_shot_examples()
            if examples:
                few_shot_examples.extend(examples)

        prompt = ""
        if few_shot_examples:
            prompt += "\n\n# Examples\nHere are some examples of how to use the available actions:\n\n"
            for i, example in enumerate(few_shot_examples):
                if self.completion.response_format == LLMResponseFormat.REACT:
                    prompt += f"\n**Example {i + 1}**"
                    action_data = example.action.model_dump()
                    thoughts = action_data.pop("thoughts", "")

                    # Special handling for StringReplace and CreateFile action
                    if example.action.__class__.__name__ in [
                        "StringReplaceArgs",
                        "CreateFileArgs",
                        "AppendStringArgs",
                        "InsertLinesArgs",
                    ]:
                        prompt += f"\nTask: {example.user_input}"
                        prompt += f"\nThought: {thoughts}\n"
                        prompt += f"Action: {str(example.action.name)}\n"

                        if example.action.__class__.__name__ == "StringReplaceArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += (
                                f"<old_str>\n{action_data['old_str']}\n</old_str>\n"
                            )
                            prompt += (
                                f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                            )
                        elif example.action.__class__.__name__ == "AppendStringArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += (
                                f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                            )
                        elif example.action.__class__.__name__ == "CreateFileArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<file_text>\n{action_data['file_text']}\n</file_text>\n"
                        elif example.action.__class__.__name__ == "InsertLinesArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<insert_line>{action_data['insert_line']}</insert_line>\n"
                            prompt += (
                                f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                            )
                    else:
                        # Original JSON format for other actions
                        prompt += (
                            f"\nTask: {example.user_input}"
                            f"\nThought: {thoughts}\n"
                            f"Action: {str(example.action.name)}\n"
                            f"{json.dumps(action_data)}\n\n"
                        )

                elif self.completion.response_format == LLMResponseFormat.JSON:
                    action_json = {
                        "action": example.action.model_dump(),
                        "action_type": example.action.name,
                    }
                    prompt += f"User: {example.user_input}\nAssistant:\n```json\n{json.dumps(action_json, indent=2)}\n```\n\n"

                elif self.completion.response_format == LLMResponseFormat.TOOLS:
                    tools_json = {"tool": example.action.name}
                    if self.thoughts_in_action:
                        tools_json.update(example.action.model_dump())
                    else:
                        tools_json.update(
                            example.action.model_dump(exclude={"thoughts"})
                        )

                    prompt += f"Task: {example.user_input}\n"
                    if not self.thoughts_in_action:
                        prompt += f"<thoughts>{example.action.thoughts}</thoughts>\n"
                    prompt += json.dumps(tools_json)
                    prompt += "\n\n"

        return prompt

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        dump["actions"] = []
        dump["agent_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        for action in self.actions:
            dump["actions"].append(action.model_dump(**kwargs))
        return dump

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        repository: Repository = None,
        runtime: Any = None,
        code_index: CodeIndex = None,
    ) -> "ActionAgent":
        if isinstance(obj, dict):
            obj = obj.copy()
            completion_data = obj.pop("completion", None)
            agent_class_path = obj.pop("agent_class", None)

            message_generator_data = obj.get("message_generator", {})
            if message_generator_data:
                obj["message_generator"] = MessageHistoryGenerator.model_validate(
                    message_generator_data
                )

            if completion_data:
                obj["completion"] = CompletionModel.model_validate(completion_data)
            else:
                obj["completion"] = None

            if repository:
                obj["actions"] = [
                    Action.model_validate(
                        action_data,
                        repository=repository,
                        runtime=runtime,
                        code_index=code_index,
                    )
                    for action_data in obj.get("actions", [])
                ]
            else:
                logger.info(f"No repository provided, skip initiating actions")
                obj["actions"] = []

            if agent_class_path:
                module_name, class_name = agent_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                instance = agent_class(**obj)
            else:
                instance = cls(**obj)

            return instance

        return super().model_validate(obj)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        runtime: Any | None = None,
    ) -> "ActionAgent":
        """Create an ActionAgent from a dictionary, properly handling dependencies."""
        data = data.copy()

        # Handle completion model
        if "completion" in data and isinstance(data["completion"], dict):
            data["completion"] = CompletionModel.model_validate(data["completion"])

        # Handle actions with dependencies
        if repository and "actions" in data and isinstance(data["actions"], list):
            data["actions"] = [
                Action.model_validate(
                    action_data,
                    repository=repository,
                    runtime=runtime,
                    code_index=code_index,
                )
                for action_data in data["actions"]
            ]

        # Handle message generator
        if "message_generator" in data and isinstance(data["message_generator"], dict):
            data["message_generator"] = MessageHistoryGenerator.model_validate(
                data["message_generator"]
            )

        # Handle agent class if specified
        if "agent_class" in data:
            module_name, class_name = data["agent_class"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            return agent_class(**data)

        return cls.model_validate(data)

    @property
    def completion(self) -> CompletionModel:
        return self._completion

    @completion.setter
    def completion(self, value: CompletionModel):
        self._completion = value
