"""Wheel qualification rejects masked package origins in actual child records."""

import runpy
from copy import deepcopy
from pathlib import Path

import pytest

# The release helper deliberately targets Python 3.11; library code also allows 3.10.
pytest.importorskip("tomllib")
verify_record_packages = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/check_paired_install.py"))[
    "verify_record_packages"
]
VERSIONS = {"lighter": "0.2.0.dev0", "sparkwheel": "0.1.0.dev0"}


@pytest.fixture
def installed_record(tmp_path):
    environment = tmp_path / "environment"
    packages = {}
    for name, version in VERSIONS.items():
        path = environment / "lib/python3.11/site-packages" / name / "__init__.py"
        path.parent.mkdir(parents=True)
        path.write_text(f'__version__ = "{version}"\n')
        packages[name] = {"module_path": str(path), "loaded_version": version, "distribution_version": version}
    return environment, {"attempt_id": "child-attempt", "environment": {"packages": packages}}


def test_completed_child_uses_installed_versions_and_origins(installed_record):
    environment, record = installed_record
    verify_record_packages(record, environment, VERSIONS)


@pytest.mark.parametrize("package", VERSIONS)
def test_example_local_package_cannot_mask_installed_wheel(installed_record, tmp_path, package):
    environment, record = installed_record
    masked = tmp_path / "copied-example" / package / "__init__.py"
    masked.parent.mkdir(parents=True)
    masked.write_text("# A local package must not qualify as the installed wheel.\n")
    record["environment"]["packages"][package]["module_path"] = str(masked)
    with pytest.raises(ValueError, match=f"loaded {package} outside"):
        verify_record_packages(record, environment, VERSIONS)


@pytest.mark.parametrize("field", ["loaded_version", "distribution_version"])
def test_child_version_must_match_built_artifact(installed_record, field):
    environment, record = installed_record
    record["environment"]["packages"]["lighter"][field] = "0.1.0"
    with pytest.raises(ValueError, match="inconsistent lighter versions"):
        verify_record_packages(record, environment, VERSIONS)


def test_missing_or_symlinked_origin_is_not_wheel_provenance(installed_record, tmp_path):
    environment, record = installed_record
    missing = deepcopy(record)
    missing["environment"]["packages"]["lighter"].pop("module_path")
    with pytest.raises(ValueError, match="outside"):
        verify_record_packages(missing, environment, VERSIONS)
    real = tmp_path / "source.py"
    real.write_text("# outside environment\n")
    link = environment / "masked.py"
    link.symlink_to(real)
    record["environment"]["packages"]["lighter"]["module_path"] = str(link)
    with pytest.raises(ValueError, match="outside"):
        verify_record_packages(record, environment, VERSIONS)
