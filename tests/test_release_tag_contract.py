"""Network-free checks for the actual release guard and both workflow callers."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="Release tooling explicitly uses Python 3.12")
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/check_release_tag.py"
spec = importlib.util.spec_from_file_location("release_tag_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
TAG_PUSH = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "user.email", "release-fixture@example.invalid")
    git(tmp_path, "config", "user.name", "Release fixture")
    (tmp_path / "src/example").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "example"\nversion = "0.1.0"\n')
    (tmp_path / "src/example/__init__.py").write_text('__version__ = "0.1.0"\nraise RuntimeError("must not import")\n')
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "stable fixture")
    sha = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "update-ref", "refs/remotes/origin/main", sha)
    return tmp_path, sha


def test_matching_stable_tag_without_importing_package(repository):
    root, sha = repository
    assert guard.validate_release(root, "push", "refs/tags/v0.1.0", sha) == "0.1.0"


@pytest.mark.parametrize(
    "ref",
    [
        "refs/tags/v0.1.1",
        "refs/tags/v0.1.0.dev0",
        "refs/tags/v0.1.0rc1",
        "refs/tags/v0.1.0.post1",
        "refs/tags/v0.1.0+local",
        "refs/tags/anything",
        "refs/tags/0.1.0",
        "refs/tags/v00.1.0",
        "refs/heads/main",
    ],
)
def test_reject_wrong_or_nonstable_tag(repository, ref):
    root, sha = repository
    with pytest.raises(ValueError):
        guard.validate_release(root, "push", ref, sha)


def test_reject_manual_dispatch_even_on_stable_tag(repository):
    root, sha = repository
    with pytest.raises(ValueError, match="requires a push"):
        guard.validate_release(root, "workflow_dispatch", "refs/tags/v0.1.0", sha)


@pytest.mark.parametrize("literal", ['"0.1.1"', '"0.1.0.dev0"', "None"])
def test_reject_source_version_mismatch(repository, literal):
    root, sha = repository
    (root / "src/example/__init__.py").write_text(f"__version__ = {literal}\n")
    with pytest.raises(ValueError, match="source __version__"):
        guard.validate_release(root, "push", "refs/tags/v0.1.0", sha)


def test_tag_may_be_ancestor_of_main(repository):
    root, sha = repository
    git(root, "commit", "--allow-empty", "-m", "later main")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(root, "checkout", "--detach", sha)
    assert guard.validate_release(root, "push", "refs/tags/v0.1.0", sha) == "0.1.0"


def test_reject_commit_outside_main(repository):
    root, _ = repository
    git(root, "checkout", "-b", "unmerged")
    git(root, "commit", "--allow-empty", "-m", "unmerged")
    with pytest.raises(subprocess.CalledProcessError):
        guard.validate_release(root, "push", "refs/tags/v0.1.0", git(root, "rev-parse", "HEAD"))


def test_reject_wrong_checkout(repository):
    root, sha = repository
    git(root, "commit", "--allow-empty", "-m", "other checkout")
    with pytest.raises(ValueError, match="checkout"):
        guard.validate_release(root, "push", "refs/tags/v0.1.0", sha)


def test_missing_main_fails_closed(repository):
    root, sha = repository
    git(root, "update-ref", "-d", "refs/remotes/origin/main")
    with pytest.raises(subprocess.CalledProcessError):
        guard.validate_release(root, "push", "refs/tags/v0.1.0", sha)


def test_cli_failure_and_success(repository):
    root, sha = repository
    command = [sys.executable, str(SCRIPT), "--root", str(root), "--event", "push", "--sha", sha, "--ref"]
    good = subprocess.run([*command, "refs/tags/v0.1.0"], capture_output=True, text=True)
    bad = subprocess.run([*command, "refs/tags/v0.1.0rc1"], capture_output=True, text=True)
    assert good.returncode == 0 and "Verified stable release 0.1.0" in good.stdout
    assert bad.returncode == 1 and "Release rejected" in bad.stderr


def test_workflows_gate_publication_and_share_the_guard():
    publish = yaml.load((ROOT / ".github/workflows/publish.yml").read_text(), Loader=yaml.BaseLoader)
    release = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    assert "workflow_dispatch" in publish["on"]
    assert publish["jobs"]["publish"]["if"] == TAG_PUSH
    assert publish["jobs"]["publish"]["needs"] == "build"
    assert release["jobs"]["release"]["if"] == TAG_PUSH
    assert "workflow_dispatch" not in release["on"]
    for steps, destination in [
        (publish["jobs"]["build"]["steps"], "Build a binary wheel and a source tarball"),
        (release["jobs"]["release"]["steps"], "Create Release"),
    ]:
        checks = [step for step in steps if step.get("name") == "Verify stable release tag"]
        assert len(checks) == 1
        check = checks[0]
        assert check["if"] == TAG_PUSH
        assert "git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main" in check["run"]
        assert "--no-project --python 3.12 python .github/scripts/check_release_tag.py" in check["run"]
        assert '--event "$RELEASE_EVENT" --ref "$RELEASE_REF" --sha "$RELEASE_SHA"' in check["run"]
        assert check["env"] == {
            "RELEASE_EVENT": "${{ github.event_name }}",
            "RELEASE_REF": "${{ github.ref }}",
            "RELEASE_SHA": "${{ github.sha }}",
        }
        assert steps.index(check) < next(i for i, step in enumerate(steps) if step.get("name") == destination)
    assert "secrets.PYPI_TOKEN" in publish["jobs"]["publish"]["steps"][-1]["run"]
