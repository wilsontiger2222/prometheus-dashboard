"""Review queue for Herald content drafts.

Drafts must be approved before publishing. The queue holds
drafts in memory and provides approve/reject/list operations.
"""

from agents.herald.composer import Draft


class ReviewQueue:
    """In-memory draft review queue."""

    def __init__(self):
        self._drafts: dict[str, Draft] = {}

    def add(self, draft: Draft):
        """Add a draft to the review queue."""
        self._drafts[draft.id] = draft

    def approve(self, draft_id: str) -> bool:
        """Approve a draft. Returns False if not found."""
        draft = self._drafts.get(draft_id)
        if not draft:
            return False
        draft.approve()
        return True

    def reject(self, draft_id: str) -> bool:
        """Reject a draft. Returns False if not found."""
        draft = self._drafts.get(draft_id)
        if not draft:
            return False
        draft.reject()
        return True

    def remove(self, draft_id: str):
        """Remove a draft from the queue."""
        self._drafts.pop(draft_id, None)

    def get_pending(self) -> list[Draft]:
        """Get all pending (unapproved) drafts."""
        return [d for d in self._drafts.values() if d.status == "pending"]

    def get_approved(self) -> list[Draft]:
        """Get all approved drafts ready to publish."""
        return [d for d in self._drafts.values() if d.status == "approved"]

    def get_all(self) -> list[Draft]:
        """Get all drafts regardless of status."""
        return list(self._drafts.values())
