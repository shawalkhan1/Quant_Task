"""
Polymarket Data Fetcher — REAL prediction market data.

Uses the three FREE, no-auth Polymarket APIs:
  - Gamma API  (https://gamma-api.polymarket.com)  → market discovery, metadata, resolution
  - CLOB API   (https://clob.polymarket.com)        → price history per token (prices-history)
  - Data API   (https://data-api.polymarket.com)    → alternative price timeseries

No Black-Scholes. No synthetic data. Real crowd-sourced probabilities.

Key insight about Polymarket market schema:
  - 'outcomes'      : list of outcome labels, e.g. ["Yes", "No"] or ["Team A", "Team B"]
  - 'clobTokenIds'  : list of CLOB token IDs parallel to outcomes
  - 'outcomePrices' : list of current prices for each outcome (resolved = "1"/"0")
  - 'closedTime'    : ISO timestamp when market closed
  - 'umaResolutionStatus': "resolved" when settled
  - 'volume'        : total USD traded
"""

import logging
import os
import time
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple, Callable, Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints (all publicly accessible, no API key required)
# ──────────────────────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API  = "https://data-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "PolymarketResearchBot/1.0"})


def _get(url: str, params: dict = None, retries: int = 3, backoff: float = 1.0):
    """GET with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = backoff * (2 ** attempt)
                logger.warning(f"Rate limited. Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise
    return None


def _parse_clob_token_ids(market: dict) -> List[str]:
    """Extract CLOB token IDs from different possible field formats."""
    raw = market.get("clobTokenIds", [])
    if isinstance(raw, str):
        try:
            import json
            raw = json.loads(raw)
        except Exception:
            raw = []
    return [str(t) for t in raw] if raw else []


def _parse_outcome_prices(market: dict) -> List[float]:
    """Extract current outcome prices as floats."""
    raw = market.get("outcomePrices", [])
    if isinstance(raw, str):
        try:
            import json
            raw = json.loads(raw)
        except Exception:
            raw = []
    try:
        return [float(p) for p in raw]
    except Exception:
        return []


class PolymarketFetcher:
    """
    Fetches real binary prediction market data from Polymarket's public APIs.

    Builds a market DataFrame with the same schema expected by the backtesting
    engine — but with REAL prices, REAL volume, and REAL resolutions instead of
    synthetic Black-Scholes estimates.

    Schema produced (per market row / minute observation):
        market_id           — Polymarket slug or conditionId
        timestamp           — UTC datetime index
        market_price_yes    — Price of outcome[0] token (0-1)
        market_price_no     — Price of outcome[1] token (0-1)
        fair_price          — Mid-point of implied probabilities
        implied_spread      — price_yes + price_no - 1 (positive = house margin)
        resolution          — Final outcome: 1=outcome[0] won, 0=outcome[1] won
        time_to_expiry_min  — Minutes until market closes
        minutes_elapsed     — Minutes since market opened
        volume_usd          — Total trading volume (USD)
        liquidity_usd       — Available liquidity (USD)
        question            — Human-readable market question
        category            — Market category tag
        symbol              — "Polymarket"
    """

    def __init__(self, cache_dir: str = None):
        try:
            from config.settings import MARKET_DATA_DIR
            self.cache_dir = cache_dir or MARKET_DATA_DIR
        except Exception:
            self.cache_dir = cache_dir or os.path.join(os.getcwd(), "data", "markets")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def fetch_active_markets(
        self,
        min_volume: float = 500.0,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch currently active Polymarket markets from the Gamma API.

        Parameters:
            min_volume : Only include markets with at least this much USD volume
            limit      : Max markets to return

        Returns:
            DataFrame of market metadata
        """
        params = {
            "limit": min(limit, 500),
            "order": "volume24hr",
            "ascending": "false",
            "active": "true",
            "closed": "false",
        }

        try:
            data = _get(f"{GAMMA_API}/markets", params=params)
        except Exception as e:
            logger.error(f"Failed to fetch active markets: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        rows = []
        for m in data:
            vol = float(m.get("volumeNum", 0) or 0)
            if vol < min_volume:
                continue

            token_ids = _parse_clob_token_ids(m)
            out_prices = _parse_outcome_prices(m)
            outcomes = m.get("outcomes", [])
            if isinstance(outcomes, str):
                try:
                    import json
                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = []

            if len(token_ids) < 2:
                continue

            rows.append({
                "condition_id":   m.get("conditionId", ""),
                "market_id":      m.get("slug") or m.get("conditionId", ""),
                "question":       m.get("question", ""),
                "category":       m.get("category", ""),
                "end_date":       m.get("endDate", ""),
                "volume_usd":     vol,
                "liquidity_usd":  float(m.get("liquidityNum", 0) or 0),
                "token_id_0":     token_ids[0],
                "token_id_1":     token_ids[1] if len(token_ids) > 1 else "",
                "outcome_0":      outcomes[0] if outcomes else "Outcome0",
                "outcome_1":      outcomes[1] if len(outcomes) > 1 else "Outcome1",
                "price_0":        out_prices[0] if out_prices else 0.5,
                "price_1":        out_prices[1] if len(out_prices) > 1 else 0.5,
                "last_trade":     float(m.get("lastTradePrice", 0.5) or 0.5),
                "best_bid":       float(m.get("bestBid", 0) or 0),
                "best_ask":       float(m.get("bestAsk", 1) or 1),
            })

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} active markets (volume >= ${min_volume:,.0f})")
        return df

    def fetch_resolved_markets(
        self,
        days_back: int = 30,
        min_volume: float = 1000.0,
        limit: int = 500,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch recently RESOLVED Polymarket markets.

        Strategy: Fetch high-volume markets sorted by volume, then filter for
        those that have:
          - closedTime in the past (within days_back)
          - multiple clobTokenIds (binary markets)
          - umaResolutionStatus = 'resolved' OR outcomePrices showing settled prices

        Parameters:
            days_back   : How many days back to look for resolved markets
            min_volume  : Only include markets with at least this USD volume
            limit       : Max markets to scan
            category    : Optional category filter

        Returns:
            DataFrame of resolved market metadata
        """
        import json as _json

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days_back)

        # Fetch a large batch sorted by 1-week volume to capture recently active markets
        rows_collected = []
        for order_field in ["volume1wk", "volume", "volume1mo"]:
            params = {
                "limit": min(limit, 500),
                "order": order_field,
                "ascending": "false",
            }
            if category:
                params["category"] = category

            try:
                data = _get(f"{GAMMA_API}/markets", params=params)
            except Exception as e:
                logger.error(f"Failed to fetch markets (order={order_field}): {e}")
                continue

            if not data:
                continue

            for m in data:
                vol = float(m.get("volumeNum", 0) or 0)
                if vol < min_volume:
                    continue

                # Must have at least 2 CLOB token IDs
                token_ids = _parse_clob_token_ids(m)
                if len(token_ids) < 2:
                    continue

                # closedTime must be in the look-back window
                closed_str = m.get("closedTime") or m.get("endDate", "")
                if not closed_str:
                    continue
                try:
                    closed_dt = pd.to_datetime(closed_str, utc=True)
                except Exception:
                    continue

                if closed_dt >= pd.Timestamp(now):    # not yet closed
                    continue
                if closed_dt < pd.Timestamp(cutoff):  # too old
                    continue

                # Determine resolution from outcomePrices
                out_prices = _parse_outcome_prices(m)
                outcomes = m.get("outcomes", [])
                if isinstance(outcomes, str):
                    try:
                        outcomes = _json.loads(outcomes)
                    except Exception:
                        outcomes = []

                if len(out_prices) >= 2:
                    price_0 = out_prices[0]
                    price_1 = out_prices[1]
                    # Resolution: outcome 0 won if price_0 > 0.5
                    resolution = 1 if price_0 >= 0.5 else 0
                else:
                    resolution = 0
                    price_0 = 0.5
                    price_1 = 0.5

                # Check if resolution is clear (price near 0 or 1)
                is_resolved = (
                    price_0 <= 0.05 or price_0 >= 0.95 or
                    m.get("umaResolutionStatus") == "resolved" or
                    m.get("automaticallyResolved", False)
                )

                mid = m.get("conditionId", "")
                # Skip duplicates
                if any(r["condition_id"] == mid for r in rows_collected):
                    continue

                rows_collected.append({
                    "condition_id":      mid,
                    "market_id":         m.get("slug") or mid,
                    "question":          m.get("question", ""),
                    "category":          str(m.get("category", "") or ""),
                    "end_date":          str(closed_str),
                    "end_datetime":      closed_dt,
                    "volume_usd":        vol,
                    "liquidity_usd":     float(m.get("liquidityNum", 0) or 0),
                    "token_id_0":        token_ids[0],
                    "token_id_1":        token_ids[1] if len(token_ids) > 1 else "",
                    "outcome_0":         outcomes[0] if outcomes else "Yes",
                    "outcome_1":         outcomes[1] if len(outcomes) > 1 else "No",
                    "resolution":        resolution,
                    "is_resolved":       is_resolved,
                    "price_0_final":     price_0,
                    "price_1_final":     price_1,
                    # legacy alias for compatibility
                    "yes_token_id":      token_ids[0],
                    "no_token_id":       token_ids[1] if len(token_ids) > 1 else "",
                })

            if len(rows_collected) >= limit:
                break

        df = pd.DataFrame(rows_collected)
        if not df.empty:
            df = df.sort_values("end_datetime", ascending=False).reset_index(drop=True)

        resolved_count = int(df["is_resolved"].sum()) if not df.empty and "is_resolved" in df.columns else 0
        logger.info(
            f"Found {len(df)} markets in last {days_back} days "
            f"(vol >= ${min_volume:,.0f}); {resolved_count} confirmed resolved"
        )
        return df

    def fetch_market_timeseries(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        fidelity: int = 1,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute price timeseries for a single Polymarket token.

        Tries two endpoints in order:
          1. CLOB prices-history (best coverage, interval-based)
          2. Data API prices (timestamp-range based, alternative)

        Parameters:
            token_id  : The CLOB token ID
            start_ts  : Unix timestamp (seconds) — start of window
            end_ts    : Unix timestamp (seconds) — end of window
            fidelity  : Resolution in minutes for Data API (1 = 1-minute bars)

        Returns:
            DataFrame with columns: timestamp (UTC index), price (0-1)
        """
        if not token_id:
            return pd.DataFrame()

        # ── Try 1: CLOB prices-history ──────────────────────────────────
        # IMPORTANT: CLOB 'interval' parameter means "last N of history from NOW"
        # not a custom time window. So always request 'max' to get all history,
        # then filter to the requested window afterwards.
        try:
            data = _get(
                f"{CLOB_API}/prices-history",
                params={"market": token_id, "interval": "max", "fidelity": fidelity},
            )

            if data and isinstance(data, dict):
                hist = data.get("history", [])
                if hist:
                    rows = []
                    for point in hist:
                        try:
                            ts = pd.to_datetime(point.get("t"), unit="s", utc=True)
                            price = float(point.get("p", 0.5))
                            rows.append({"timestamp": ts, "price": price})
                        except Exception:
                            continue
                    if rows:
                        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
                        df = df[~df.index.duplicated(keep="first")]
                        # Filter to requested window
                        if start_ts:
                            df = df[df.index >= pd.Timestamp(start_ts, unit="s", tz="UTC")]
                        if end_ts:
                            df = df[df.index <= pd.Timestamp(end_ts + 3600, unit="s", tz="UTC")]
                        if not df.empty:
                            logger.debug(f"CLOB: {len(df)} points for {token_id[:16]}...")
                            return df
        except Exception as e:
            logger.debug(f"CLOB prices-history failed for {token_id[:16]}: {e}")

        # ── Try 2: Data API ─────────────────────────────────────────────
        try:
            params = {"market": token_id, "fidelity": fidelity}
            if start_ts:
                params["startTs"] = start_ts
            if end_ts:
                params["endTs"] = end_ts + 3600  # add buffer

            data = _get(f"{DATA_API}/prices", params=params)
            if data and isinstance(data, list):
                rows = []
                for point in data:
                    try:
                        ts_val = point.get("t") or point.get("timestamp")
                        price = float(point.get("p") or point.get("price") or 0.5)
                        ts = pd.to_datetime(ts_val, unit="s", utc=True)
                        rows.append({"timestamp": ts, "price": price})
                    except Exception:
                        continue
                if rows:
                    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
                    df = df[~df.index.duplicated(keep="first")]
                    logger.debug(f"DataAPI: {len(df)} points for {token_id[:16]}...")
                    return df
        except Exception as e:
            logger.debug(f"Data API failed for {token_id[:16]}: {e}")

        return pd.DataFrame()

    def build_market_dataset(
        self,
        markets_meta: pd.DataFrame,
        days_back: int = 14,
        use_cache: bool = True,
        max_markets: int = 100,
    ) -> pd.DataFrame:
        """
        Build a full market observations DataFrame from market metadata.

        For each market:
          1. Fetch token_0 timeseries → market_price_yes over time
          2. Fetch token_1 timeseries → market_price_no  over time
          3. Align and compute derived fields
          4. Set resolution from actual outcome

        Parameters:
            markets_meta : DataFrame from fetch_resolved_markets() or fetch_active_markets()
            days_back    : How many days of history to fetch per market
            use_cache    : If True, read/write per-market Parquets to cache_dir
            max_markets  : Maximum number of markets to process

        Returns:
            Combined DataFrame with ALL market observations, indexed by timestamp
        """
        all_records = []
        meta_slice = markets_meta.head(max_markets)
        n = len(meta_slice)

        for i, (_, row) in enumerate(meta_slice.iterrows(), 1):
            mid = str(row.get("market_id", f"market_{i}"))
            # Sanitize for filename
            safe_mid = mid.replace("/", "_").replace(":", "_")[:40]
            cache_path = os.path.join(self.cache_dir, f"{safe_mid}_obs.parquet")

            if use_cache and os.path.exists(cache_path):
                try:
                    cached = pd.read_parquet(cache_path)
                    all_records.append(cached)
                    logger.debug(f"[{i}/{n}] Cache hit: {mid[:30]}")
                    continue
                except Exception:
                    pass

            logger.info(f"[{i}/{n}] Fetching: {mid[:50]}")

            tid_0 = str(row.get("token_id_0") or row.get("yes_token_id", ""))
            tid_1 = str(row.get("token_id_1") or row.get("no_token_id", ""))
            resolution = int(row.get("resolution", 0))

            # Time window
            end_dt = row.get("end_datetime")
            if pd.isna(end_dt) or end_dt is None:
                # For active markets use now
                end_dt = pd.Timestamp.now(tz="UTC")

            start_dt = end_dt - timedelta(days=days_back)
            start_unix = int(start_dt.timestamp())
            end_unix   = int(end_dt.timestamp())

            # Fetch timeseries for both outcomes
            ts0 = self.fetch_market_timeseries(tid_0, start_unix, end_unix) if tid_0 else pd.DataFrame()
            ts1 = self.fetch_market_timeseries(tid_1, start_unix, end_unix) if tid_1 else pd.DataFrame()

            if ts0.empty and ts1.empty:
                logger.warning(f"  No timeseries data for {mid[:40]}, skipping.")
                continue

            # Build common time index
            if not ts0.empty and not ts1.empty:
                common_idx = ts0.index.union(ts1.index)
            elif not ts0.empty:
                common_idx = ts0.index
            else:
                common_idx = ts1.index

            if len(common_idx) < 5:
                logger.warning(f"  Too few data points ({len(common_idx)}) for {mid[:30]}, skipping.")
                continue

            prices_0 = ts0["price"].reindex(common_idx).ffill().fillna(0.5) if not ts0.empty else pd.Series(0.5, index=common_idx)
            prices_1 = ts1["price"].reindex(common_idx).ffill().fillna(0.5) if not ts1.empty else pd.Series(0.5, index=common_idx)

            # Clip to valid probability range
            prices_0 = prices_0.clip(0.01, 0.99)
            prices_1 = prices_1.clip(0.01, 0.99)

            open_dt = common_idx.min()
            total_duration = max((end_dt - open_dt).total_seconds() / 60.0, 1.0)

            obs = pd.DataFrame({
                "market_id":         mid,
                "market_price_yes":  prices_0,   # outcome_0 = "Yes" analog
                "market_price_no":   prices_1,   # outcome_1 = "No"  analog
            }, index=common_idx)

            obs["fair_price"]      = (obs["market_price_yes"] + (1 - obs["market_price_no"])) / 2.0
            obs["implied_spread"]  = obs["market_price_yes"] + obs["market_price_no"] - 1.0
            obs["resolution"]      = resolution
            obs["volume_usd"]      = float(row.get("volume_usd", 0) or 0)
            obs["liquidity_usd"]   = float(row.get("liquidity_usd", 0) or 0)
            obs["question"]        = str(row.get("question", ""))
            obs["outcome_yes"]     = str(row.get("outcome_0", "Yes"))
            obs["outcome_no"]      = str(row.get("outcome_1", "No"))
            obs["category"]        = str(row.get("category", "") or "")
            obs["symbol"]          = "Polymarket"

            obs["time_to_expiry_min"] = [
                max((end_dt - ts).total_seconds() / 60.0, 0.0)
                for ts in common_idx
            ]
            obs["minutes_elapsed"] = [
                max((ts - open_dt).total_seconds() / 60.0, 0.0)
                for ts in common_idx
            ]
            obs["total_duration_min"] = total_duration

            # Implied volatility from price changes
            obs["implied_vol"] = (
                obs["market_price_yes"].diff().abs().rolling(10, min_periods=2).std()
                * np.sqrt(525600)
            ).fillna(0.3)

            obs.index.name = "timestamp"

            if use_cache:
                try:
                    obs.to_parquet(cache_path)
                except Exception as ex:
                    logger.debug(f"Cache write failed: {ex}")

            all_records.append(obs)
            time.sleep(0.3)  # polite rate limiting

        if not all_records:
            logger.warning("No market observations collected.")
            return pd.DataFrame()

        combined = pd.concat(all_records).sort_index()
        logger.info(
            f"Built dataset: {len(combined):,} observations across "
            f"{combined['market_id'].nunique()} markets, "
            f"date range: {combined.index.min()} → {combined.index.max()}"
        )
        return combined

    def fetch_clob_orderbook(self, token_id: str) -> dict:
        """
        Fetch live order book for a token from the CLOB API.
        Returns best bid, best ask, and mid price.
        """
        try:
            data = _get(f"{CLOB_API}/book", params={"token_id": token_id})
            if not data:
                return {}
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else None
            best_ask = float(asks[0]["price"]) if asks else None
            mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else None
            return {"best_bid": best_bid, "best_ask": best_ask, "mid": mid}
        except Exception as e:
            logger.debug(f"CLOB fetch failed for {token_id[:16]}...: {e}")
            return {}

    def get_live_prices(self, token_ids: List[str]) -> Dict[str, float]:
        """
        Get current prices for a list of token IDs via the CLOB API.
        Returns: {token_id: current_price}
        """
        prices = {}
        try:
            data = _get(f"{CLOB_API}/prices", params={"token_id": ",".join(token_ids)})
            if isinstance(data, dict):
                for tid, price_info in data.items():
                    if isinstance(price_info, dict):
                        prices[tid] = float(price_info.get("price", 0.5))
                    else:
                        prices[tid] = float(price_info)
            elif isinstance(data, list):
                for item in data:
                    tid = item.get("token_id", "")
                    if tid:
                        prices[tid] = float(item.get("price", 0.5))
        except Exception as e:
            logger.debug(f"Live prices fetch failed: {e}")
        return prices

    def stream_live_prices(
        self,
        token_ids: List[str],
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        custom_feature_enabled: bool = True,
    ):
        """
        Connect to Polymarket CLOB websocket and stream live market updates.

        Parameters:
            token_ids : token IDs to subscribe to
            on_update : callback receiving parsed message dicts
            ws_url    : websocket endpoint for market subscriptions
            custom_feature_enabled : enables additional market channel events

        Returns:
            websocket-client WebSocketApp (caller should run_forever() or close())
        """
        try:
            from websocket import WebSocketApp
        except ImportError as ex:
            raise ImportError(
                "websocket-client is required for live websocket streaming. "
                "Install with: pip install websocket-client"
            ) from ex

        if not token_ids:
            raise ValueError("token_ids cannot be empty for websocket subscription")

        def _on_open(ws):
            # Initial market-channel subscribe payload per Polymarket docs.
            subscribe_payload = {
                "type": "market",
                "assets_ids": token_ids,
                "custom_feature_enabled": custom_feature_enabled,
            }
            ws.send(json.dumps(subscribe_payload))
            logger.info("Subscribed to Polymarket websocket for %d assets", len(token_ids))

        def _on_message(_, message):
            try:
                payload = json.loads(message)
            except Exception:
                logger.debug("Non-JSON websocket payload ignored")
                return

            if on_update is not None:
                on_update(payload)

        def _on_error(_, error):
            logger.warning("Polymarket websocket error: %s", error)

        def _on_close(_, status_code, close_msg):
            logger.info("Polymarket websocket closed (%s): %s", status_code, close_msg)

        return WebSocketApp(
            ws_url,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )

    # ──────────────────────────────────────────────────────────────────
    # Convenience: build a complete backtesting dataset
    # ──────────────────────────────────────────────────────────────────

    def fetch_dataset_for_backtest(
        self,
        days_back: int = 30,
        min_volume: float = 5000.0,
        max_markets: int = 80,
        category: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        End-to-end convenience method: fetch resolved markets and build
        the observation dataset ready for backtesting.

        Parameters:
            days_back   : How many days back to look for resolved markets
            min_volume  : Minimum USD trading volume per market
            max_markets : Maximum number of markets to process
            category    : Optional Polymarket category filter
            use_cache   : Cache per-market timeseries locally

        Returns:
            Combined market observations DataFrame (standardized schema)
        """
        logger.info(
            f"Fetching Polymarket dataset: {days_back} days back, "
            f"min_vol=${min_volume:,.0f}, max_markets={max_markets}"
        )

        meta = self.fetch_resolved_markets(
            days_back=days_back,
            min_volume=min_volume,
            limit=max_markets * 4,
            category=category,
        )

        if meta.empty:
            logger.error("Could not fetch market metadata from Polymarket.")
            return pd.DataFrame()

        logger.info(f"Found {len(meta)} resolved markets; fetching timeseries...")
        dataset = self.build_market_dataset(
            meta,
            days_back=days_back,
            use_cache=use_cache,
            max_markets=max_markets,
        )

        return dataset
