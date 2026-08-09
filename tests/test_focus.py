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


# --- 全職炒家 RON LAU and JK爸爸: two channels whose titles broke the parser ---

RON_LIVE = ("【週二直播】一片睇清8月恆指部署！ 阿里10億大資金流出！小米, 藥明, 滙豐, "
            "AAPL, GOOG 最新分析｜投資花越少時間越容易成功？｜RON LAU｜主持 Wendy #港股 #美股")
RON_DAILY = ("恆指急升近1000點，潛力股已偷步上升！阿里,騰訊,華虹,中芯,滙豐,渣打,MU,NVDA "
             "最新分析【熱點先機】 #恆指 #倍升股 #牛市 #熊市")
JK_LIVE = ("【午後開股】30/07/2026 倍數責任完成 ?｜#恒指 挑戰 26000 #港股 8 月點分析 ?｜"
           "#3690 #美團 可以繼續上 ?｜JK Sir｜Jason Sir｜Car｜投創教育")


def test_a_declaration_without_a_colon_is_still_a_declaration():
    """「主持 Wendy」 separates the keyword from the name with a space. Requiring
    the colon read it as no declaration at all and fell through to the hashtag
    branch, which answered 港股, 美股 — the topic tags — as the hosts."""
    from ytdigest.summarize import declared_hosts

    assert declared_hosts(RON_LIVE) == ["Wendy"]
    assert declared_hosts(RON_DAILY) == []


def test_topic_hashtags_are_never_read_as_speakers():
    """#恆指 #倍升股 #牛市 are indices and a market condition. Handing them to the
    prompt declared three market indices to be the people speaking."""
    from ytdigest.summarize import hosts_from_title

    assert hosts_from_title(RON_DAILY) == []
    # The tag fallback still works where a tag really is a person (1號月台).
    assert hosts_from_title("港股7月未升完？ #Kctalk #羅家聰") == ["羅家聰 (KC)"]


def test_a_declaration_is_widened_by_anyone_else_the_title_names():
    """Wendy moderates; Ron is the analyst and the reason anyone watches. The
    host list is also the whitelist parse_views filters attributions against, so
    a list of just the moderator discards every view Ron makes on his own
    channel."""
    from ytdigest.summarize import hosts_from_title

    assert hosts_from_title(RON_LIVE) == ["Wendy", "Ron Lau (全職炒家)"]


def test_roster_names_alone_never_create_a_host_list():
    """JK Sir｜Jason Sir｜Car declares nobody, and only JK is on the roster.
    Answering "the sole host is JK Sir" would hand him the other two's calls;
    an empty list makes the prompt write 主持 throughout instead."""
    from ytdigest.summarize import hosts_from_title

    assert hosts_from_title(JK_LIVE) == []


def test_both_new_channels_are_kept(conn):
    for vid, title in (("r1", RON_LIVE), ("r2", RON_DAILY), ("j1", JK_LIVE)):
        keep, reason = in_focus(conn, add(conn, vid, title))
        assert keep, f"{title[:40]} skipped: {reason}"
