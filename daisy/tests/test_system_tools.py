"""Tests for system_tools — command allowlist, time, system info."""

import pytest
from daisy.tools.system_tools import _command_allowed, get_time_date


class TestCommandAllowed:
    def test_allowed_command_matches(self):
        assert _command_allowed("ls -la", ["ls", "cat", "df"]) is True

    def test_not_in_allowlist(self):
        assert _command_allowed("rm -rf /", ["ls", "cat"]) is False

    def test_empty_command(self):
        assert _command_allowed("", ["ls"]) is False

    def test_whitespace_only(self):
        assert _command_allowed("   ", ["ls"]) is False

    def test_command_with_pipe(self):
        # The first token is the command; pipe is a shell metachar
        # With shlex.split, "ls|cat" is one token, so it won't match "ls"
        assert _command_allowed("ls | cat", ["ls"]) is True

    def test_empty_allowlist(self):
        assert _command_allowed("ls", []) is False

    def test_exact_match_required(self):
        # "l" should not match "ls"
        assert _command_allowed("l", ["ls"]) is False


@pytest.mark.asyncio
async def test_get_time_date_returns_string():
    result = await get_time_date()
    assert isinstance(result, str)
    assert "It is" in result
    assert "Timezone" in result
