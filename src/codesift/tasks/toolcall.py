# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Tasks of kind `toolcall`: tools offered -> a valid call, right name and arguments

A task may carry `turns` and `results`, which let the model look before it acts:
a call to a tool other than the wanted one is answered from `results` and the
model is asked again, up to `turns` replies in all.
"""

TASKS = [
    dict(
        id="tc_single", kind="toolcall", want="search_files",
        prompt="Find every file in the repository whose name contains 'config'. Use the tools.",
        tools=[{
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for files by a substring of their name",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string", "description": "substring to match"}},
                    "required": ["pattern"],
                },
            },
        }],
    ),
    dict(
        id="tc_choose", kind="toolcall", want="run_tests",
        # Listing the test files before running them is a step a harness would
        # serve rather than a wrong answer, so the lookup is answered and the
        # model gets a second turn to reach the tool the task asks for.
        turns=2, results={"search_files": ["tests/test_api.py", "tests/test_db.py"]},
        prompt="The test suite is failing and I need to see the output. Use the tools.",
        tools=[
            {"type": "function", "function": {
                "name": "search_files",
                "description": "Search for files by a substring of their name",
                "parameters": {"type": "object",
                               "properties": {"pattern": {"type": "string"}},
                               "required": ["pattern"]}}},
            {"type": "function", "function": {
                "name": "run_tests",
                "description": "Run the project's test suite and return the output",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}},
                               "required": []}}},
        ],
    ),
    dict(id="tc_restraint", kind="toolcall", want=None,
         prompt="What is 17 + 25? Answer directly.",
         tools=[{"type": "function", "function": {
             "name": "search_web",
             "description": "Search the web for current information about a topic",
             "parameters": {"type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"]}}}]),
    dict(id="tc_nested", kind="toolcall", want="create_issue",
         want_args={"title": str, "labels": list},
         prompt="Open a bug report titled 'Crash on empty input' and tag it with both "
                "'bug' and 'urgent'. Use the tools.",
         tools=[{"type": "function", "function": {
             "name": "create_issue",
             "description": "Create a new issue in the tracker",
             "parameters": {"type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "body": {"type": "string"},
                                "labels": {"type": "array", "items": {"type": "string"}}},
                            "required": ["title", "labels"]}}}]),
]
