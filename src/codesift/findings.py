# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""What a measurement found, as fields rather than as a sentence.

A ledger holding "failed at 48k" cannot be improved by editing: the wording was
fixed when the model was measured, and only another sweep would replace it. So a
gate records what it found and every reader builds its own text from that, which
means the phrasing can change at any time without measuring again.

Both paths that reject a model report through here -- the gates in `triage` and the
records in `analysis` -- so the same fault cannot be worded two ways on one page.
"""
from __future__ import annotations


def _k(tokens) -> str:
    """A token count as a reader says it: 65536 -> 64k, 48000 -> 48k."""
    if not tokens:
        return "the configured"
    return f"{round(tokens / 1024)}k" if tokens % 1024 == 0 else f"{round(tokens / 1000)}k"


def _share(count, total) -> str:
    """A count against what it was drawn from, where that is known."""
    return f"{100 * count / total:.0f}%" if total else f"{count}"


def describe(finding: dict) -> str:
    """One finding as a sentence."""
    code = finding.get("code")
    if code == "slow_generation":
        return f"too slow to use: {finding['tok_s']:.0f} tok/s"
    if code == "malformed_tool_calls":
        return f"{_share(finding['malformed'], finding['total'])} of tool calls unparseable"
    if code == "wrong_tool":
        # Well formed and sent to the wrong place. A harness returns the wrong
        # result and the model gets another turn, so this degrades a session
        # rather than ending it.
        return f"{_share(finding['wrong'], finding['total'])} of tool calls to the wrong tool"
    if code == "context_truncated":
        return f"prompt truncated at its {_k(finding['num_ctx'])} window"
    if code == "not_screened":
        return "not screened"
    if code == "not_probed":
        return "not probed"
    if code == "replies_capped":
        return (f"{_share(finding['count'], finding['total'])} of replies cut off "
                f"at the output limit")
    if code == "replies_silent":
        return (f"{_share(finding['count'], finding['total'])} of replies produced "
                f"no answer within the output limit")
    if code == "error":
        return finding.get("message") or "the request failed"
    if code == "generation_unreadable":
        return "generation rate unreadable"
    if code == "tools_ok":
        return f"{finding['total']} tool calls, all well formed"
    if code == "context_ok":
        return "kept the whole prompt"
    if code == "generation_ok":
        return f"generates {finding['tok_s']:.0f} tok/s"
    return code or ""


def sentence(rec: dict) -> str:
    """Every finding on a record, joined."""
    return "; ".join(describe(f) for f in rec.get("findings") or [])
