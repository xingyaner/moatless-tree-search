import os
from unittest.mock import patch

from moatless.actions.fuzz_build import FuzzBuild, FuzzBuildArgs
from moatless.actions.simple_view import SimpleViewCode, SimpleViewCodeArgs
from moatless.actions.string_replace import StringReplace, StringReplaceArgs
from moatless.file_context import FileContext
from moatless.repository.file import FileRepository


class _FakeStdout:
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class _FakeProcess:
    def __init__(self, lines, returncode):
        self.stdout = _FakeStdout(lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def test_temp_capability_view_patch_apply_verify_flow(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    target_file = repo_root / "sample.txt"
    target_file.write_text("hello\nworld\n", encoding="utf-8")

    oss_fuzz_root = tmp_path / "oss-fuzz"
    infra_dir = oss_fuzz_root / "infra"
    infra_dir.mkdir(parents=True)
    (infra_dir / "helper.py").write_text("# stub helper\n", encoding="utf-8")

    repository = FileRepository(repo_path=str(repo_root))
    file_context = FileContext(repo=repository)

    view_action = SimpleViewCode(repository=repository)
    view_args = SimpleViewCodeArgs(
        thoughts="View the file before editing.",
        file_path="sample.txt",
    )
    view_observation = view_action.execute(view_args, file_context)
    assert view_observation.summary == "Viewed sample.txt"

    replace_action = StringReplace(repository=repository)
    replace_args = StringReplaceArgs(
        thoughts="Replace the target line.",
        path="sample.txt",
        old_str="world",
        new_str="patched",
    )
    replace_observation = replace_action.execute(replace_args, file_context)
    assert replace_observation.summary == "Applied patch to sample.txt"
    assert "diff" in replace_observation.properties

    context_file = file_context.get_context_file("sample.txt")
    assert context_file is not None
    assert "patched" in context_file.content
    assert file_context.generate_git_patch()

    popen_calls = []

    def fake_popen(command, **kwargs):
        popen_calls.append(command)
        if len(popen_calls) == 1:
            return _FakeProcess(["build completed\n"], 0)
        if len(popen_calls) == 2:
            return _FakeProcess(["check_build passed\n"], 0)
        raise AssertionError("Unexpected extra subprocess invocation")

    fuzz_action = FuzzBuild(
        project_name="demo-project",
        oss_fuzz_path=str(oss_fuzz_root),
        sanitizer="address",
        engine="libfuzzer",
        architecture="x86_64",
    )
    fuzz_args = FuzzBuildArgs(thoughts="Verify the applied patch.")

    with patch("subprocess.Popen", side_effect=fake_popen):
        fuzz_observation = fuzz_action.execute(fuzz_args, file_context)

    assert fuzz_observation.summary == "Success"
    assert fuzz_observation.message == "BUILD SUCCESSFUL. All fuzzing targets were built successfully."
    assert len(popen_calls) == 2
    assert popen_calls[0][2] == "build_fuzzers"
    assert popen_calls[1][2] == "check_build"

    log_path = os.path.join("fuzz_build_log_file", "fuzz_build_log.txt")
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as handle:
        log_content = handle.read()
    assert "--- [check_build] ---" in log_content