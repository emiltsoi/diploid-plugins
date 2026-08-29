from diploid_plugins.self_management.self_management_mcp import (
    SelfManagementMcpServer,
)


def test_mcp_server_lists_tools() -> None:
    server = SelfManagementMcpServer(
        chat_id="12345",
        harness_url="http://127.0.0.1:4003",
        sessions_root="/tmp",
    )
    tools = server._tools()
    names = {t["name"] for t in tools}
    assert "plugin_list" in names
    assert "plugin_add" in names
    assert "plugin_sandbox" in names


def test_mcp_server_handles_initialize() -> None:
    server = SelfManagementMcpServer(
        chat_id="12345",
        harness_url="http://127.0.0.1:4003",
        sessions_root="/tmp",
    )
    response = server._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response["result"]["serverInfo"]["name"] == "diploid-self-management"
