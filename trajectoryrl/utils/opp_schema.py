"""OpenClaw Policy Pack (OPP) v1 schema validation.

The OPP format is designed to be:
- Portable: Copy files to workspace, apply tool_policy to config
- Content-addressed: sha256(pack_json) uniquely identifies the artifact
- OpenClaw-compatible: Maps 1:1 to AGENTS.md, SOUL.md, tool policy
- Validator-friendly: Can be evaluated in sandbox without trusting miner
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


OPP_SCHEMA_V1 = {
    "schema_version": 1,
    "required_fields": [
        "schema_version",
        "files",
        "tool_policy",
        "metadata"
    ],
    "optional_fields": [
        "target_runtime",
        "min_runtime_version",
        "approval_gates",
        "stop_rules"
    ]
}


@dataclass
class ValidationResult:
    """Result of OPP schema validation."""
    passed: bool
    issues: List[str]

    def __bool__(self) -> bool:
        return self.passed


def validate_opp_schema(pack: dict) -> ValidationResult:
    """Validate pack against OPP v1 schema.

    Args:
        pack: Policy pack dictionary

    Returns:
        ValidationResult with passed=True if valid, issues list if not
    """
    issues = []

    # Check schema version
    if pack.get("schema_version") != 1:
        issues.append(f"Unsupported schema_version: {pack.get('schema_version')}")

    # Check required fields
    for field in OPP_SCHEMA_V1["required_fields"]:
        if field not in pack:
            issues.append(f"Missing required field: {field}")

    # Validate files section
    if "files" in pack:
        if not isinstance(pack["files"], dict):
            issues.append("'files' must be a dict")
        else:
            # Check for required files
            if "AGENTS.md" not in pack["files"]:
                issues.append("Missing required file: AGENTS.md")

            # Validate file content is string
            for filename, content in pack["files"].items():
                if not isinstance(content, str):
                    issues.append(f"File '{filename}' content must be string")

    # Validate tool_policy section
    if "tool_policy" in pack:
        issues.extend(_validate_tool_policy(pack["tool_policy"]))

    # Validate metadata section
    if "metadata" in pack:
        issues.extend(_validate_metadata(pack["metadata"]))

    # Size check (prevent token bombs)
    import json
    pack_size = len(json.dumps(pack))
    if pack_size > 32768:  # 32KB limit
        issues.append(f"Pack too large: {pack_size} bytes (max 32KB)")

    return ValidationResult(
        passed=len(issues) == 0,
        issues=issues
    )


def _validate_tool_policy(tool_policy: dict) -> List[str]:
    """Validate tool_policy section.

    Args:
        tool_policy: Tool policy dict with allow/deny lists

    Returns:
        List of validation issues
    """
    issues = []

    # Check for legacy 'allowed'/'denied' (should be 'allow'/'deny')
    if "allowed" in tool_policy or "denied" in tool_policy:
        issues.append(
            "Use 'allow'/'deny' (OpenClaw semantics), not 'allowed'/'denied'"
        )

    # Validate allow list
    if "allow" in tool_policy:
        if not isinstance(tool_policy["allow"], list):
            issues.append("'allow' must be a list")
        else:
            for item in tool_policy["allow"]:
                if not isinstance(item, str):
                    issues.append(f"'allow' item must be string: {item}")

    # Validate deny list
    if "deny" in tool_policy:
        if not isinstance(tool_policy["deny"], list):
            issues.append("'deny' must be a list")
        else:
            for item in tool_policy["deny"]:
                if not isinstance(item, str):
                    issues.append(f"'deny' item must be string: {item}")

    # Check for dangerous tools without explicit deny
    dangerous = {"exec", "shell", "group:runtime", "admin_*"}
    allow_set = set(tool_policy.get("allow", []))
    deny_set = set(tool_policy.get("deny", []))

    dangerous_allowed = dangerous & allow_set
    if dangerous_allowed and not (dangerous & deny_set):
        issues.append(
            "Dangerous tools in 'allow' but no dangerous tools in 'deny' "
            "(defense-in-depth: consider denying specific dangerous tools)"
        )

    return issues


def _validate_metadata(metadata: dict) -> List[str]:
    """Validate metadata section.

    Args:
        metadata: Pack metadata dict

    Returns:
        List of validation issues
    """
    issues = []

    # Check for required metadata fields
    required = ["pack_name", "pack_version", "target_suite"]
    for field in required:
        if field not in metadata:
            issues.append(f"Required metadata field missing: {field}")

    # Validate pack_version format if present
    if "pack_version" in metadata:
        version = metadata["pack_version"]
        if not isinstance(version, str):
            issues.append("'pack_version' must be string")
        elif not _is_valid_semver(version):
            issues.append(
                f"'pack_version' should follow semver (e.g., '1.0.0'): {version}"
            )

    return issues


def _is_valid_semver(version: str) -> bool:
    """Check if version string follows semantic versioning.

    Args:
        version: Version string

    Returns:
        True if valid semver format
    """
    parts = version.split(".")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)
