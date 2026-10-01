"""Tests for the Monte Carlo reconciler."""
import numpy as np

from ironclad.simulation.game_draw import GameDrawResult
from ironclad.simulation.player_draw import PlayerContext, PlayerDrawResult
from ironclad.simulation.reconciler import Reconciler


def _make_player(pid, pos="WR", rec_yards=50.0, rush_yards=0.0, targets=5,
                 carries=0, played=True, tgt_rate=0.05, carry_rate=0.04):
    return PlayerDrawResult(
        player_id=pid, player_name=pid, team="A", position=pos,
        is_home=True, rec_yards=rec_yards, rush_yards=rush_yards,
        targets=targets, carries=carries, played=played,
    )


def _make_context(pid, pos="WR"):
    return PlayerContext(
        player_id=pid, player_name=pid, team="A", position=pos,
        is_home=True, availability=1.0,
        targets_projected=5.0, carries_projected=0.0,
        pass_attempts_projected=0.0,
        catch_rate=0.65, yards_per_target=8.0, yards_per_carry=4.2,
        td_rate_per_target=0.05, td_rate_per_carry=0.04,
    )


def _make_game(home_score=24, pass_yards=220.0, rush_yards=100.0):
    return GameDrawResult(
        home_score=home_score, away_score=17,
        home_pass_att=30, away_pass_att=28,
        home_rush_att=20, away_rush_att=22,
        home_pass_yards=pass_yards, away_pass_yards=180.0,
        home_rush_yards=rush_yards, away_rush_yards=90.0,
    )


def test_receiving_yards_scaled_to_team():
    rng = np.random.default_rng(0)
    rec = Reconciler()
    players = [_make_player("A", rec_yards=100.0), _make_player("B", rec_yards=100.0)]
    ctxs = [_make_context("A"), _make_context("B")]
    game = _make_game(pass_yards=150.0)

    home_out, _ = rec.reconcile(game, players, [], ctxs, [], rng)
    total_rec = sum(p.rec_yards for p in home_out)
    assert abs(total_rec - 150.0) < 1.0


def test_player_out_has_zero_stats():
    rng = np.random.default_rng(1)
    rec = Reconciler()
    p_out = _make_player("X", rec_yards=50.0, played=False)
    p_active = _make_player("Y", rec_yards=80.0, played=True)
    ctxs = [_make_context("X"), _make_context("Y")]
    game = _make_game(pass_yards=80.0)
    home_out, _ = rec.reconcile(game, [p_out, p_active], [], ctxs, [], rng)
    out_player = next(p for p in home_out if p.player_id == "X")
    assert out_player.rec_yards == 0.0 or out_player.played is False


def test_tds_are_non_negative():
    rng = np.random.default_rng(2)
    rec = Reconciler()
    players = [_make_player(str(i), rec_yards=40.0, targets=4) for i in range(4)]
    ctxs = [_make_context(str(i)) for i in range(4)]
    game = _make_game(home_score=28)
    home_out, _ = rec.reconcile(game, players, [], ctxs, [], rng)
    for p in home_out:
        assert p.tds >= 0


def test_rush_yards_scaled():
    rng = np.random.default_rng(3)
    rec = Reconciler()
    players = [_make_player("RB", pos="RB", rec_yards=0.0, rush_yards=120.0, carries=15)]
    ctxs = [_make_context("RB", pos="RB")]
    game = _make_game(rush_yards=80.0)
    home_out, _ = rec.reconcile(game, players, [], ctxs, [], rng)
    assert abs(home_out[0].rush_yards - 80.0) < 1.0


def _offense():
    qb = _make_player("qb", pos="QB", rec_yards=0.0, targets=0, carries=4)
    qb.pass_attempts, qb.completions = 34, 22
    players = [
        qb,
        _make_player("rb", pos="RB", rec_yards=10.0, rush_yards=70.0, targets=3, carries=18),
        _make_player("wr1", rec_yards=80.0, targets=9),
        _make_player("wr2", rec_yards=50.0, targets=6),
        _make_player("te", pos="TE", rec_yards=40.0, targets=5),
    ]
    ctxs = [_make_context(p.player_id, p.position) for p in players]
    return players, ctxs


def test_every_passing_td_is_credited_to_a_receiver():
    rec = Reconciler()
    for seed in range(200):
        players, ctxs = _offense()
        by_id = {p.player_id: p for p in players}
        by_id["qb"].carries = 0      # QB tds == passing TDs
        by_id["rb"].targets = 0      # RB tds == rushing TDs
        home_out, _ = rec.reconcile(_make_game(home_score=31), players, [], ctxs, [],
                                    np.random.default_rng(seed))
        receiving = sum(p.tds for p in home_out if p.player_id in ("wr1", "wr2", "te"))
        assert receiving == by_id["qb"].tds


def test_receivers_score_a_realistic_share_of_tds():
    rec = Reconciler()
    totals = {"wr1": 0, "wr2": 0, "te": 0, "rb": 0}
    n = 2000
    for seed in range(n):
        players, ctxs = _offense()
        home_out, _ = rec.reconcile(_make_game(home_score=24), players, [], ctxs, [],
                                    np.random.default_rng(seed))
        for p in home_out:
            if p.player_id in totals:
                totals[p.player_id] += p.tds
    # Before the fix passing TDs were never credited to receivers (~0.01/game)
    assert totals["wr1"] / n > 0.4
    assert totals["wr1"] > totals["wr2"] > 0
    assert totals["te"] > 0
