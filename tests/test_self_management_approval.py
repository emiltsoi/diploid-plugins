from pathlib import Path

from diploid_plugins.self_management.approvals import ApprovalStore


def test_approval_flow(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals.json")
    token = store.propose("add", {"name": "test", "module": "diploid_plugins.continuity"})
    assert store.is_approved(token) is False
    assert store.approve(token) is True
    assert store.is_approved(token) is True
