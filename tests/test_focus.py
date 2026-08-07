"""Only episodes featuring someone we follow should be summarised.

The channel filter is show-based, which is not the same thing: 4點痴線財經
rotates its hosts, so an episode presented by 湯麗鴻 Kimmy matched the show
filter and was summarised even though nobody follows her.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from ytdigest import db as D
from ytdigest.summarize import episode_key, in_focus


@pytest.fixture
def conn():
    path = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    c = D.open_db(path, pathlib.Path("migrations"))
    c.execute("INSERT INTO channels (id,title,added_at) VALUES ('UC1','c',?)",
              (D.now_iso(),))
    return c


def add(conn, vid, title):
    conn.execute(
        "INSERT INTO videos (id,channel_id,title,published_at,discovered_at,status) "
        "VALUES (?,?,?,'2026-08-06','2026-08-06','new')", (vid, "UC1", title))
    return {"id": vid, "channel_id": "UC1", "title": title}


def test_episode_key_links_parts_to_their_parent():
    """第一節 / 第二節 uploads carry no host names; only the parent does."""
    parent = "Raga Finance：4點痴線財經 20260806 - 主持：湯麗鴻 Kimmy、阮子曦"
    part = "Raga Finance：4點痴線財經 20260806 - 第二節：境外保單背後目的"
    assert episode_key(parent) == episode_key(part) is not None


def test_episode_hosted_by_nobody_we_follow_is_skipped(conn):
    v = add(conn, "p1", "Raga Finance：4點痴線財經 20260806 - 主持：湯麗鴻 Kimmy、阮子曦")
    keep, reason = in_focus(conn, v)
    assert keep is False
    assert "湯麗鴻" in reason


def test_a_part_inherits_its_parents_hosts(conn):
    """This is the case that shipped: the part passed the show filter because
    its own title names nobody at all."""
    add(conn, "p1", "Raga Finance：4點痴線財經 20260806 - 主持：湯麗鴻 Kimmy、阮子曦")
    part = add(conn, "p2", "Raga Finance：4點痴線財經 20260806 - 第二節：境外保單背後目的")
    keep, _ = in_focus(conn, part)
    assert keep is False


def test_a_focus_speaker_in_a_sibling_keeps_the_whole_episode(conn):
    add(conn, "k1", "CC Raga Finance：一名經人 20260806：主持：羅家聰 KC 博士、Eugene")
    part = add(conn, "k2", "CC Raga Finance：一名經人 20260806：第一節：中國向境外分紅")
    keep, reason = in_focus(conn, part)
    assert keep is True
    assert "羅家聰" in reason


def test_a_title_format_without_主持_still_matches_on_the_name(conn):
    """A 【KC博士】 upload names him as "|| 羅家聰||"; the 主持 parser returns
    "哈富證券||26-07-22" instead. Filtering on parsed hosts alone skipped KC's
    own videos."""
    v = add(conn, "q1", "【KC博士】美企業績陸續出爐｜Kimi K3｜| 羅家聰|| Season ||#哈富證券||26-07-22")
    keep, reason = in_focus(conn, v)
    assert keep is True
    assert "羅家聰" in reason


def test_unknown_hosts_are_kept(conn):
    """Fail open. Dropping a video because its title format changed would
    silently lose exactly the content the filter exists to protect."""
    v = add(conn, "u1", "錢錢錢打到嚟 2026724 - Part 1/4：證券行事件\\美股")
    keep, reason = in_focus(conn, v)
    assert keep is True
    assert "could not be determined" in reason


def test_skipped_videos_are_not_reclaimed(conn):
    """Terminal, like done and abandoned. Re-claiming would re-evaluate and
    re-log the same decision on every run."""
    add(conn, "s1", "anything")
    conn.execute("UPDATE videos SET status=? WHERE id='s1'", (D.SKIPPED,))
    assert [r["id"] for r in D.claim_queue(conn, 10)] == []
