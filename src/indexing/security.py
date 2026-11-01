# Security utilities for sanitizing credentials and performing
# static code analysis (SAST).

import ast
import re
from functools import partial
from typing import List, Tuple


_SECRET_NAMES = (
    r"\w*(?:api[_-]?key|secret|password|passwd|private[_-]?key"
    r"|(?:access|auth|refresh|bearer|hf|github|slack)[_-]?token)\w*"
    r"|token"
)


class SecuritySanitizer:
    """Scans and masks secrets and credentials in text using
    pre-compiled regex patterns.

    Patterns run in order, so specific token formats come before the
    generic assignment rules. When a pattern has a `secret` group, only
    that group is masked and the surrounding text (variable name,
    quotes, URL) is kept so the LLM still gets useful context.
    """

    PATTERNS_COMPILED = {

        "PRIVATE_KEY": re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
            r"[\s\S]*?"
            r"(?:-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----|\Z)"
        ),

        "PRIVATE_KEY_TAIL": re.compile(
            r"\A[A-Za-z0-9+/=\s]*?"
            r"-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
        ),
        "URL_PASSWORD": re.compile(
            r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:(?P<secret>[^\s/@]+)@",
            re.IGNORECASE,
        ),
        "ANTHROPIC_KEY": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
        "OPENAI_KEY": re.compile(
            r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{16,}"
        ),
        "STRIPE_KEY": re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"),
        "GITHUB_TOKEN": re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"
        ),
        "SLACK_TOKEN": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
        "GOOGLE_API_KEY": re.compile(r"\bAIza[A-Za-z0-9_-]{35}"),
        "AWS_KEY": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        "JWT_TOKEN": re.compile(
            r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"
        ),

        "CREDENTIAL": re.compile(
            r"\b(?:" + _SECRET_NAMES + r")[\"']?\s*[:=]\s*"
            r"(?P<q>[\"'])(?!\[MASKED_)(?P<secret>[^\"'\s]{6,})(?P=q)",
            re.IGNORECASE,
        ),

        "ENV_SECRET": re.compile(
            r"^[ \t]*(?:export[ \t]+)?[A-Z0-9_]*"
            r"(?:API_KEY|SECRET|PASSWORD|PASSWD|TOKEN)[A-Z0-9_]*="
            r"(?!\[MASKED_|\d+(?:\s|$))(?P<secret>[^\s'\"#]{6,})",
            re.MULTILINE,
        ),
    }

    @staticmethod
    def _mask(secret_type: str, match: "re.Match[str]") -> str:
        """Build the replacement for one match.

        Args:
            secret_type (str): Name of the pattern that matched.
            match (re.Match[str]): The regex match to replace.

        """
        mask = f"[MASKED_{secret_type.removesuffix('_TAIL')}]"
        if "secret" not in match.re.groupindex:
            return mask
        text = match.group(0)
        start = match.start("secret") - match.start()
        end = match.end("secret") - match.start()
        return text[:start] + mask + text[end:]

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, int]:
        """Scan and replace sensitive credentials in text with
        placeholder mask tokens.

        Args:
            text (str): The input string to sanitize.
        """
        sanitized_text = text
        total_masked = 0

        for secret_type, pattern in cls.PATTERNS_COMPILED.items():
            sanitized_text, count = pattern.subn(
                partial(cls._mask, secret_type), sanitized_text
            )
            total_masked += count

        return sanitized_text, total_masked


class SASTScanner:
    """Performs lightweight static security audits using AST
    parsing with regex fallback."""

    REGEX_RULES_COMPILED = [
        (
            re.compile(r"\beval\s*\("),
            "CWE-95: Dangerous use of eval() detected.",
        ),
        (
            re.compile(r"\bexec\s*\("),
            "CWE-95: Dangerous use of exec() detected.",
        ),
        (
            re.compile(r"\bpickle\.loads?\b"),
            "CWE-502: Insecure deserialization with `pickle`.",
        ),
        (
            re.compile(r"\bos\.system\s*\("),
            "CWE-78: Command injection using os.system().",
        ),
        (
            re.compile(r"shell\s*=\s*True"),
            "CWE-78: Possible command injection via `shell=True`.",
        ),
        (
            re.compile(r"\byaml\.load\s*\("),
            "CWE-20: Loading YAML unsafely without SafeLoader.",
        ),
    ]

    DANGEROUS_FUNCTIONS = {
        "eval": "CWE-95: Dangerous use of eval() detected.",
        "exec": "CWE-95: Dangerous use of exec() detected.",
        "compile": "CWE-95: Dangerous use of compile() detected.",
        "__import__": "CWE-95: Dangerous use of __import__() detected.",
    }

    DANGEROUS_ATTRIBUTES = {
        ("os", "system"): "CWE-78: Command injection using os.system().",
        ("os", "popen"): "CWE-78: Command injection using os.popen().",
        (
            "pickle",
            "loads",
        ): "CWE-502: Insecure deserialization with `pickle`.",
        (
            "pickle",
            "load",
        ): "CWE-502: Insecure deserialization with `pickle`.",
        ("yaml", "load"): "CWE-20: Loading YAML unsafely without SafeLoader.",
        (
            "subprocess",
            "Popen",
        ): "AST-Alert: Use of subprocess.Popen() detected.",
        (
            "subprocess",
            "call",
        ): "AST-Alert: Use of subprocess.call() detected.",
        ("subprocess", "run"): "AST-Alert: Use of subprocess.run() detected.",
    }

    @classmethod
    def scan(cls, text: str, is_python: bool = False) -> List[str]:
        """Scan text or Python source code for security vulnerabilities.

        Args:
            text (str): The source code or plain text content to scan.
            is_python (bool, optional): Whether to inspect the input as
            Python code using AST analysis. Defaults to False.

        Returns:
            List[str]: A list of warning messages indicating
            detected security issues.
        """
        warnings: List[str] = []

        if not is_python:
            cls._scan_regex(text, warnings)
        else:
            try:
                parsed_ast = ast.parse(text)
                for node in ast.walk(parsed_ast):
                    if isinstance(node, ast.Call):
                        if (
                            isinstance(node.func, ast.Name)
                            and node.func.id in cls.DANGEROUS_FUNCTIONS
                        ):
                            msg = cls.DANGEROUS_FUNCTIONS[node.func.id]
                            if msg not in warnings:
                                warnings.append(msg)

                        elif isinstance(
                            node.func, ast.Attribute
                        ) and isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                            func_name = node.func.attr
                            key = (module_name, func_name)
                            if key in cls.DANGEROUS_ATTRIBUTES:
                                msg = cls.DANGEROUS_ATTRIBUTES[key]
                                if msg not in warnings:
                                    warnings.append(msg)

                        for kw in node.keywords:
                            if (
                                kw.arg == "shell"
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value is True
                            ):
                                msg = (
                                    "CWE-78: Possible command injection "
                                    "via `shell=True`."
                                )
                                if msg not in warnings:
                                    warnings.append(msg)
            except SyntaxError:
                cls._scan_regex(text, warnings)

        return warnings

    @classmethod
    def _scan_regex(cls, text: str, warnings: List[str]) -> None:
        """Scan input text against regex rules and append
        missing warning messages.

        Args:
            text (str): The content to scan using compiled regex patterns.
            warnings (List[str]): The list of warnings where
            new findings are appended.
        """
        for pattern, warning_msg in cls.REGEX_RULES_COMPILED:
            if pattern.search(text) and warning_msg not in warnings:
                warnings.append(warning_msg)
