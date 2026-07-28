"""Optional live API smoke test, excluded from the default test run."""

from datetime import UTC, datetime, timedelta

import pytest

from arxiv_digest.arxiv_client import ArxivClient
from arxiv_digest.config import ArxivConfig, QueryConfig
from arxiv_digest.models import DateWindow


@pytest.mark.live
def test_live_arxiv_api_returns_a_valid_feed() -> None:
    now = datetime.now(UTC)
    window = DateWindow(start=now - timedelta(days=30), end=now)
    with ArxivClient(ArxivConfig(page_size=1)) as client:
        result = client.fetch(
            [QueryConfig(name="smoke", terms=["condensed matter"])],
            window,
            maximum_candidates=1,
        )
    assert result.query_results[0].records_received >= 0
