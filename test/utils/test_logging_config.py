"""Tests for visionframework.utils.logging_config."""

import logging

import pytest

from visionframework.utils.logging_config import (
    configure_visionframework_logging,
    reset_visionframework_logging,
)


@pytest.fixture(autouse=True)
def _reset_vf_logging():
    reset_visionframework_logging()
    yield
    reset_visionframework_logging()


def test_configure_defaults_to_warning_when_no_env(monkeypatch):
    monkeypatch.delenv("VISIONFRAMEWORK_LOG_LEVEL", raising=False)
    monkeypatch.delenv("VF_LOG_LEVEL", raising=False)
    configure_visionframework_logging()
    assert logging.getLogger("visionframework").level == logging.WARNING


def test_configure_reads_visionframework_log_level(monkeypatch):
    monkeypatch.setenv("VISIONFRAMEWORK_LOG_LEVEL", "DEBUG")
    configure_visionframework_logging()
    assert logging.getLogger("visionframework").level == logging.DEBUG


def test_vf_log_level_alias(monkeypatch):
    monkeypatch.delenv("VISIONFRAMEWORK_LOG_LEVEL", raising=False)
    monkeypatch.setenv("VF_LOG_LEVEL", "ERROR")
    configure_visionframework_logging()
    assert logging.getLogger("visionframework").level == logging.ERROR


def test_invalid_level_falls_back_to_warning(monkeypatch):
    monkeypatch.setenv("VISIONFRAMEWORK_LOG_LEVEL", "not_a_valid_level")
    configure_visionframework_logging()
    assert logging.getLogger("visionframework").level == logging.WARNING


def test_idempotent_second_call_ignores_env_change(monkeypatch):
    monkeypatch.setenv("VISIONFRAMEWORK_LOG_LEVEL", "INFO")
    configure_visionframework_logging()
    monkeypatch.setenv("VISIONFRAMEWORK_LOG_LEVEL", "DEBUG")
    configure_visionframework_logging()
    assert logging.getLogger("visionframework").level == logging.INFO


def test_explicit_level_overrides_idempotency():
    reset_visionframework_logging()
    configure_visionframework_logging(logging.WARNING)
    configure_visionframework_logging(logging.DEBUG)
    assert logging.getLogger("visionframework").level == logging.DEBUG
