"""Tests for the Herald review queue."""

import pytest
from agents.herald.queue import ReviewQueue
from agents.herald.composer import Draft


def _make_draft(content="test", draft_id="d1"):
    return Draft(
        content=content, platform="twitter", trigger="manual",
        topic="test", id=draft_id,
    )


class TestReviewQueue:
    def test_add_draft(self):
        queue = ReviewQueue()
        draft = _make_draft()
        queue.add(draft)
        assert len(queue.get_pending()) == 1

    def test_get_pending_only_returns_pending(self):
        queue = ReviewQueue()
        d1 = _make_draft(content="pending", draft_id="d1")
        d2 = _make_draft(content="approved", draft_id="d2")
        d2.approve()
        queue.add(d1)
        queue.add(d2)
        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "d1"

    def test_approve_draft(self):
        queue = ReviewQueue()
        draft = _make_draft(draft_id="d1")
        queue.add(draft)
        result = queue.approve("d1")
        assert result is True
        assert draft.status == "approved"

    def test_reject_draft(self):
        queue = ReviewQueue()
        draft = _make_draft(draft_id="d1")
        queue.add(draft)
        result = queue.reject("d1")
        assert result is True
        assert draft.status == "rejected"

    def test_approve_nonexistent_returns_false(self):
        queue = ReviewQueue()
        assert queue.approve("nope") is False

    def test_get_approved(self):
        queue = ReviewQueue()
        d1 = _make_draft(draft_id="d1")
        d2 = _make_draft(draft_id="d2")
        queue.add(d1)
        queue.add(d2)
        queue.approve("d1")
        approved = queue.get_approved()
        assert len(approved) == 1
        assert approved[0].id == "d1"

    def test_remove_draft(self):
        queue = ReviewQueue()
        draft = _make_draft(draft_id="d1")
        queue.add(draft)
        queue.remove("d1")
        assert len(queue.get_pending()) == 0

    def test_get_all(self):
        queue = ReviewQueue()
        queue.add(_make_draft(draft_id="d1"))
        queue.add(_make_draft(draft_id="d2"))
        queue.add(_make_draft(draft_id="d3"))
        assert len(queue.get_all()) == 3
