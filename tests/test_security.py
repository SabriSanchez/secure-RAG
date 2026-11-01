"""Tests for SecuritySanitizer.

Every positive case checks two things: the expected mask appears and
the raw secret is gone from the output. Negative cases make sure normal
code and docs from the corpus are left untouched.
"""

import pytest

from src.indexing.security import SecuritySanitizer

RSA_BODY = "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gun"
OPENSSH_BODY = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMw"
PKCS8_BODY = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj"
JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


def _pem(label: str, body: str) -> str:
    return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----"


# (input text, raw secret that must disappear, expected mask)
POSITIVE_CASES = [
    pytest.param(
        _pem("RSA PRIVATE KEY", RSA_BODY), RSA_BODY,
        "[MASKED_PRIVATE_KEY]", id="rsa-private-key",
    ),
    pytest.param(
        _pem("OPENSSH PRIVATE KEY", OPENSSH_BODY), OPENSSH_BODY,
        "[MASKED_PRIVATE_KEY]", id="openssh-private-key",
    ),
    pytest.param(
        _pem("PRIVATE KEY", PKCS8_BODY), PKCS8_BODY,
        "[MASKED_PRIVATE_KEY]", id="pkcs8-private-key-body",
    ),
    pytest.param(
        _pem("ENCRYPTED PRIVATE KEY", PKCS8_BODY), PKCS8_BODY,
        "[MASKED_PRIVATE_KEY]", id="encrypted-private-key",
    ),
    pytest.param(
        _pem("PGP PRIVATE KEY BLOCK", PKCS8_BODY), PKCS8_BODY,
        "[MASKED_PRIVATE_KEY]", id="pgp-private-key-block",
    ),
    pytest.param(
        f"config:\n-----BEGIN RSA PRIVATE KEY-----\n{RSA_BODY}", RSA_BODY,
        "[MASKED_PRIVATE_KEY]", id="key-cut-by-chunker-head",
    ),
    pytest.param(
        f"{RSA_BODY}\n-----END RSA PRIVATE KEY-----\nmore text", RSA_BODY,
        "[MASKED_PRIVATE_KEY]", id="key-cut-by-chunker-tail",
    ),
    pytest.param(
        "API_KEY=sk_live_51Habcdef1234567890XYZ",
        "sk_live_51Habcdef1234567890XYZ",
        "[MASKED_STRIPE_KEY]", id="env-stripe-key",
    ),
    pytest.param(
        "export DB_PASSWORD=Hunter2Hunter2", "Hunter2Hunter2",
        "[MASKED_ENV_SECRET]", id="env-export-password",
    ),
    pytest.param(
        'password = "Sup3rS3cret!"', "Sup3rS3cret!",
        "[MASKED_CREDENTIAL]", id="python-password",
    ),
    pytest.param(
        '{"client_secret": "a1b2c3d4e5f6"}', "a1b2c3d4e5f6",
        "[MASKED_CREDENTIAL]", id="json-client-secret",
    ),
    pytest.param(
        "aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCY'",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCY",
        "[MASKED_CREDENTIAL]", id="aws-secret-access-key",
    ),
    pytest.param(
        "token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "[MASKED_GITHUB_TOKEN]", id="github-token",
    ),
    pytest.param(
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
        "[MASKED_GITHUB_TOKEN]", id="github-fine-grained-pat",
    ),
    pytest.param(
        "postgres://admin:hunter2@db.local:5432/app", "hunter2",
        "[MASKED_URL_PASSWORD]", id="url-credentials",
    ),
    pytest.param(
        "key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "[MASKED_ANTHROPIC_KEY]", id="anthropic-key",
    ),
    pytest.param(
        "OPENAI: sk-proj-1234567890abcdefXYZ", "sk-proj-1234567890abcdefXYZ",
        "[MASKED_OPENAI_KEY]", id="openai-key",
    ),
    pytest.param(
        "slack xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij",
        "[MASKED_SLACK_TOKEN]", id="slack-token",
    ),
    pytest.param(
        "maps AIzaSyA1234567890abcdefghijklmnopqrstuv",
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
        "[MASKED_GOOGLE_API_KEY]", id="google-api-key",
    ),
    pytest.param(
        "id AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE",
        "[MASKED_AWS_KEY]", id="aws-access-key-id",
    ),
    pytest.param(
        f"Authorization: Bearer {JWT}", JWT,
        "[MASKED_JWT_TOKEN]", id="jwt",
    ),
]


@pytest.mark.parametrize("text, secret, mask", POSITIVE_CASES)
def test_secret_is_masked(text: str, secret: str, mask: str) -> None:
    sanitized, count = SecuritySanitizer.sanitize(text)

    assert secret not in sanitized
    assert mask in sanitized
    assert count >= 1


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('tokenizer = "Qwen/Qwen3-0.6B"', id="tokenizer-name"),
        pytest.param('eos_token = "<|endoftext|>"', id="eos-token"),
        pytest.param('cache_key = "model_weights_v2"', id="cache-key"),
        pytest.param("password = get_password()", id="password-from-call"),
        pytest.param(
            'api_key = os.environ["OPENAI_API_KEY"]', id="key-from-env"
        ),
        pytest.param('password = "abc"', id="too-short"),
        pytest.param("MAX_TOKENS=100000", id="numeric-env-value"),
        pytest.param("See https://docs.vllm.ai/en/latest/", id="plain-url"),
        pytest.param(
            "Never hardcode your API key or password in the code.",
            id="prose",
        ),
    ],
)
def test_benign_text_is_untouched(text: str) -> None:
    assert SecuritySanitizer.sanitize(text) == (text, 0)


def test_context_around_secret_is_kept() -> None:
    sanitized, count = SecuritySanitizer.sanitize('password = "Sup3rS3cret!"')

    assert sanitized == 'password = "[MASKED_CREDENTIAL]"'
    assert count == 1


def test_secret_is_masked_only_once() -> None:
    sanitized, count = SecuritySanitizer.sanitize(
        'api_key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"'
    )

    assert sanitized == 'api_key = "[MASKED_ANTHROPIC_KEY]"'
    assert count == 1


def test_count_matches_number_of_secrets() -> None:
    text = (
        'api_key = "sk-proj-1234567890abcdef" '
        "and aws_access_key = AKIAIOSFODNN7EXAMPLE "
        f"and token = {JWT}"
    )

    sanitized, count = SecuritySanitizer.sanitize(text)

    assert count == 3
    assert "[MASKED_OPENAI_KEY]" in sanitized
    assert "[MASKED_AWS_KEY]" in sanitized
    assert "[MASKED_JWT_TOKEN]" in sanitized
