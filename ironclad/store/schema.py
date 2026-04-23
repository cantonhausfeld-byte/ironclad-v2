"""DDL for all bronze / silver / gold tables."""
import duckdb


def create_all_tables(conn: duckdb.DuckDBPyConnection) -> None:
    _bronze(conn)
    _silver(conn)
    _gold(conn)


# ── Bronze ─────────────────────────────────────────────────────────────────────

def _bronze(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.schedules (
        game_id          VARCHAR NOT NULL,
        season           INTEGER NOT NULL,
        season_type      VARCHAR NOT NULL,
        week             INTEGER NOT NULL,
        gameday          DATE NOT NULL,
        gametime         VARCHAR,
        away_team        VARCHAR NOT NULL,
        home_team        VARCHAR NOT NULL,
        away_score       INTEGER,
        home_score       INTEGER,
        stadium_id       VARCHAR,
        roof             VARCHAR,
        surface          VARCHAR,
        temp             FLOAT,
        wind             FLOAT,
        spread_line      FLOAT,
        total_line       FLOAT,
        away_moneyline   INTEGER,
        home_moneyline   INTEGER,
        div_game         BOOLEAN,
        overtime         BOOLEAN,
        _ingest_ts       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (game_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.play_by_play (
        play_id                VARCHAR NOT NULL,
        game_id                VARCHAR NOT NULL,
        season                 INTEGER NOT NULL,
        week                   INTEGER NOT NULL,
        home_team              VARCHAR NOT NULL,
        away_team              VARCHAR NOT NULL,
        posteam                VARCHAR,
        defteam                VARCHAR,
        play_type              VARCHAR,
        yards_gained           FLOAT,
        pass_attempt           INTEGER,
        complete_pass          INTEGER,
        rush_attempt           INTEGER,
        sack                   INTEGER,
        touchdown              INTEGER,
        interception           INTEGER,
        fumble_lost            INTEGER,
        penalty                INTEGER,
        field_goal_attempt     INTEGER,
        field_goal_result      VARCHAR,
        kick_distance          FLOAT,
        ep                     FLOAT,
        epa                    FLOAT,
        wp                     FLOAT,
        down                   INTEGER,
        ydstogo                INTEGER,
        yardline_100           INTEGER,
        game_seconds_remaining INTEGER,
        qb_dropback            INTEGER,
        qb_scramble            INTEGER,
        air_yards              FLOAT,
        yards_after_catch      FLOAT,
        passer_player_id       VARCHAR,
        passer_player_name     VARCHAR,
        receiver_player_id     VARCHAR,
        receiver_player_name   VARCHAR,
        rusher_player_id       VARCHAR,
        rusher_player_name     VARCHAR,
        _ingest_ts             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (play_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.rosters (
        season          INTEGER NOT NULL,
        week            INTEGER NOT NULL,
        player_id       VARCHAR NOT NULL,
        player_name     VARCHAR NOT NULL,
        team            VARCHAR NOT NULL,
        position        VARCHAR NOT NULL,
        depth_chart_pos VARCHAR,
        jersey_number   INTEGER,
        status          VARCHAR,
        height          VARCHAR,
        weight          INTEGER,
        years_exp       INTEGER,
        _ingest_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (season, week, player_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.injuries (
        season          INTEGER NOT NULL,
        week            INTEGER NOT NULL,
        player_id       VARCHAR NOT NULL,
        player_name     VARCHAR NOT NULL,
        team            VARCHAR NOT NULL,
        position        VARCHAR NOT NULL,
        report_status   VARCHAR,
        practice_status VARCHAR,
        injury_type     VARCHAR,
        report_date     DATE,
        _ingest_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.depth_charts (
        season      INTEGER NOT NULL,
        week        INTEGER NOT NULL,
        team        VARCHAR NOT NULL,
        position    VARCHAR NOT NULL,
        depth_team  INTEGER NOT NULL,
        player_id   VARCHAR NOT NULL,
        player_name VARCHAR NOT NULL,
        _ingest_ts  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.odds (
        game_id           VARCHAR NOT NULL,
        source            VARCHAR NOT NULL,
        retrieved_at      TIMESTAMPTZ NOT NULL,
        spread_home       FLOAT,
        spread_away       FLOAT,
        spread_juice_home FLOAT,
        spread_juice_away FLOAT,
        total_over        FLOAT,
        total_juice_over  FLOAT,
        total_juice_under FLOAT,
        moneyline_home    INTEGER,
        moneyline_away    INTEGER,
        _ingest_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.weather (
        game_id      VARCHAR NOT NULL,
        retrieved_at TIMESTAMPTZ NOT NULL,
        source       VARCHAR NOT NULL,
        temp_f       FLOAT,
        wind_mph     FLOAT,
        wind_dir     VARCHAR,
        precip_in    FLOAT,
        humidity_pct FLOAT,
        conditions   VARCHAR,
        _ingest_ts   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.snap_counts (
        season        INTEGER NOT NULL,
        week          INTEGER NOT NULL,
        game_id       VARCHAR,
        player_id     VARCHAR NOT NULL,
        player_name   VARCHAR,
        team          VARCHAR NOT NULL,
        position      VARCHAR,
        offense_snaps INTEGER DEFAULT 0,
        offense_pct   FLOAT   DEFAULT 0.0,
        defense_snaps INTEGER DEFAULT 0,
        defense_pct   FLOAT   DEFAULT 0.0,
        st_snaps      INTEGER DEFAULT 0,
        st_pct        FLOAT   DEFAULT 0.0,
        _ingest_ts    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (season, week, player_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.player_stats_weekly (
        season          INTEGER NOT NULL,
        week            INTEGER NOT NULL,
        player_id       VARCHAR NOT NULL,
        player_name     VARCHAR,
        team            VARCHAR,
        position        VARCHAR,
        completions     INTEGER DEFAULT 0,
        attempts        INTEGER DEFAULT 0,
        passing_yards   FLOAT   DEFAULT 0,
        passing_tds     INTEGER DEFAULT 0,
        interceptions   INTEGER DEFAULT 0,
        carries         INTEGER DEFAULT 0,
        rushing_yards   FLOAT   DEFAULT 0,
        rushing_tds     INTEGER DEFAULT 0,
        receptions      INTEGER DEFAULT 0,
        targets         INTEGER DEFAULT 0,
        receiving_yards FLOAT   DEFAULT 0,
        receiving_tds   INTEGER DEFAULT 0,
        target_share    FLOAT,
        air_yards_share FLOAT,
        wopr            FLOAT,
        _ingest_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (season, week, player_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS bronze.stadiums (
        stadium_id   VARCHAR NOT NULL PRIMARY KEY,
        stadium_name VARCHAR NOT NULL,
        team         VARCHAR NOT NULL,
        city         VARCHAR NOT NULL,
        state        VARCHAR,
        country      VARCHAR DEFAULT 'USA',
        is_dome      BOOLEAN NOT NULL,
        surface      VARCHAR NOT NULL,
        altitude_ft  INTEGER NOT NULL DEFAULT 0,
        capacity     INTEGER,
        lat          FLOAT,
        lon          FLOAT
    )
    """)


# ── Silver ─────────────────────────────────────────────────────────────────────

def _silver(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS silver.games (
        game_id           VARCHAR NOT NULL PRIMARY KEY,
        season            INTEGER NOT NULL,
        season_type       VARCHAR NOT NULL,
        week              INTEGER NOT NULL,
        gameday           DATE NOT NULL,
        gametime_local    VARCHAR,
        away_team         VARCHAR NOT NULL,
        home_team         VARCHAR NOT NULL,
        stadium_id        VARCHAR,
        is_dome           BOOLEAN,
        surface           VARCHAR,
        altitude_ft       INTEGER,
        temp_f            FLOAT,
        wind_mph          FLOAT,
        precip_in         FLOAT,
        spread_consensus  FLOAT,
        total_consensus   FLOAT,
        home_ml_implied   FLOAT,
        away_ml_implied   FLOAT,
        away_score        INTEGER,
        home_score        INTEGER,
        total_score       INTEGER,
        home_margin       INTEGER,
        home_win          BOOLEAN,
        overtime          BOOLEAN,
        _silver_ts        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS silver.team_game_stats (
        game_id          VARCHAR NOT NULL,
        season           INTEGER NOT NULL,
        week             INTEGER NOT NULL,
        team             VARCHAR NOT NULL,
        opponent         VARCHAR NOT NULL,
        is_home          BOOLEAN NOT NULL,
        plays_total      INTEGER,
        pass_attempts    INTEGER,
        completions      INTEGER,
        pass_yards       FLOAT,
        rush_attempts    INTEGER,
        rush_yards       FLOAT,
        total_yards      FLOAT,
        first_downs      INTEGER,
        touchdowns       INTEGER,
        turnovers        INTEGER,
        sacks_allowed    INTEGER,
        sack_yards_lost  FLOAT,
        field_goals_made INTEGER,
        field_goals_att  INTEGER,
        punts            INTEGER,
        epa_per_play     FLOAT,
        epa_pass         FLOAT,
        epa_rush         FLOAT,
        success_rate     FLOAT,
        pass_rate        FLOAT,
        points_scored    INTEGER,
        points_allowed   INTEGER,
        -- Red zone
        rz_pass_attempts INTEGER,
        rz_rush_attempts INTEGER,
        rz_touchdowns    INTEGER,
        -- Air yards
        total_air_yards  FLOAT,
        -- Pressure proxy
        sack_rate        FLOAT,
        PRIMARY KEY (game_id, team)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS silver.player_game_stats (
        game_id           VARCHAR NOT NULL,
        season            INTEGER NOT NULL,
        week              INTEGER NOT NULL,
        player_id         VARCHAR NOT NULL,
        player_name       VARCHAR NOT NULL,
        team              VARCHAR NOT NULL,
        opponent          VARCHAR NOT NULL,
        position          VARCHAR NOT NULL,
        is_home           BOOLEAN NOT NULL,
        pass_attempts     INTEGER DEFAULT 0,
        completions       INTEGER DEFAULT 0,
        pass_yards        FLOAT   DEFAULT 0,
        pass_tds          INTEGER DEFAULT 0,
        interceptions     INTEGER DEFAULT 0,
        carries           INTEGER DEFAULT 0,
        rush_yards        FLOAT   DEFAULT 0,
        rush_tds          INTEGER DEFAULT 0,
        targets           INTEGER DEFAULT 0,
        receptions        INTEGER DEFAULT 0,
        rec_yards         FLOAT   DEFAULT 0,
        rec_tds           INTEGER DEFAULT 0,
        air_yards         FLOAT   DEFAULT 0,
        yards_after_catch FLOAT   DEFAULT 0,
        rz_targets        INTEGER DEFAULT 0,
        rz_carries        INTEGER DEFAULT 0,
        total_tds         INTEGER DEFAULT 0,
        PRIMARY KEY (game_id, player_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS silver.player_weekly_status (
        season        INTEGER NOT NULL,
        week          INTEGER NOT NULL,
        player_id     VARCHAR NOT NULL,
        player_name   VARCHAR NOT NULL,
        team          VARCHAR NOT NULL,
        position      VARCHAR NOT NULL,
        depth_team    INTEGER,
        injury_status VARCHAR,
        availability  FLOAT NOT NULL DEFAULT 1.0,
        snap_rate     FLOAT,
        PRIMARY KEY (season, week, player_id)
    )
    """)


# ── Gold ───────────────────────────────────────────────────────────────────────

def _gold(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS gold.team_game_features (
        game_id                  VARCHAR NOT NULL,
        season                   INTEGER NOT NULL,
        week                     INTEGER NOT NULL,
        team                     VARCHAR NOT NULL,
        opponent                 VARCHAR NOT NULL,
        is_home                  BOOLEAN NOT NULL,
        cutoff_ts                TIMESTAMPTZ NOT NULL,
        feature_version          VARCHAR NOT NULL,
        data_completeness_score  FLOAT NOT NULL DEFAULT 0.0,
        -- Rolling offense (L4)
        off_epa_per_play_l4      FLOAT,
        off_pass_epa_l4          FLOAT,
        off_rush_epa_l4          FLOAT,
        off_pass_rate_l4         FLOAT,
        off_yards_per_play_l4    FLOAT,
        off_success_rate_l4      FLOAT,
        off_points_per_game_l4   FLOAT,
        -- Rolling defense (L4)
        def_epa_per_play_l4      FLOAT,
        def_pass_epa_l4          FLOAT,
        def_rush_epa_l4          FLOAT,
        def_yards_allowed_l4     FLOAT,
        def_success_rate_l4      FLOAT,
        def_points_allowed_l4    FLOAT,
        def_sack_rate_l4         FLOAT,
        -- Season-to-date
        off_epa_per_play_std     FLOAT,
        def_epa_per_play_std     FLOAT,
        -- Context
        rest_days                INTEGER,
        is_divisional            BOOLEAN,
        implied_total_from_odds  FLOAT,
        spread_from_odds         FLOAT,
        home_win_prob_from_odds  FLOAT,
        altitude_ft              INTEGER,
        is_dome                  BOOLEAN,
        temp_f                   FLOAT,
        wind_mph                 FLOAT,
        precip_in                FLOAT,
        surface_grass            BOOLEAN,
        -- Targets (NULL until game played)
        target_points_scored     FLOAT,
        target_yards_total       FLOAT,
        target_pass_rate         FLOAT,
        PRIMARY KEY (game_id, team)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS gold.player_game_features (
        game_id                  VARCHAR NOT NULL,
        season                   INTEGER NOT NULL,
        week                     INTEGER NOT NULL,
        player_id                VARCHAR NOT NULL,
        player_name              VARCHAR NOT NULL,
        team                     VARCHAR NOT NULL,
        opponent                 VARCHAR NOT NULL,
        position                 VARCHAR NOT NULL,
        is_home                  BOOLEAN NOT NULL,
        cutoff_ts                TIMESTAMPTZ NOT NULL,
        feature_version          VARCHAR NOT NULL,
        data_completeness_score  FLOAT NOT NULL DEFAULT 0.0,
        -- Availability
        availability             FLOAT,
        depth_team               INTEGER,
        -- Usage (rolling L4)
        snap_rate_l4             FLOAT,
        target_share_l4          FLOAT,
        carry_share_l4           FLOAT,
        air_yards_share_l4       FLOAT,
        route_rate_l4            FLOAT,
        redzone_target_share_l4  FLOAT,
        redzone_carry_share_l4   FLOAT,
        -- Efficiency (rolling L4)
        catch_rate_l4            FLOAT,
        yards_per_target_l4      FLOAT,
        yards_per_carry_l4       FLOAT,
        yac_per_rec_l4           FLOAT,
        td_rate_per_target_l4    FLOAT,
        td_rate_per_carry_l4     FLOAT,
        -- Opponent defense
        opp_def_pass_epa_l4      FLOAT,
        opp_def_rush_epa_l4      FLOAT,
        opp_def_sack_rate_l4     FLOAT,
        -- Team environment
        team_off_pass_rate_l4    FLOAT,
        team_off_epa_l4          FLOAT,
        team_implied_total        FLOAT,
        -- Targets (NULL until game played)
        target_targets           FLOAT,
        target_carries           FLOAT,
        target_receptions        FLOAT,
        target_rec_yards         FLOAT,
        target_rush_yards        FLOAT,
        target_total_tds         FLOAT,
        PRIMARY KEY (game_id, player_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS gold.model_predictions (
        prediction_id   VARCHAR NOT NULL,
        game_id         VARCHAR NOT NULL,
        model_name      VARCHAR NOT NULL,
        model_version   VARCHAR NOT NULL,
        entity_id       VARCHAR NOT NULL,
        predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        cutoff_ts       TIMESTAMPTZ NOT NULL,
        prediction_json JSON NOT NULL,
        PRIMARY KEY (prediction_id)
    )
    """)
