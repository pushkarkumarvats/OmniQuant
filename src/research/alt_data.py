"""
Alternative Data Ingestion & Processing

Institutional-grade alt data pipeline:
  - News & sentiment (NLP with FinBERT / transformer models)
  - SEC filings (EDGAR API, 10-K/10-Q/8-K parsing)
  - Satellite imagery processing (e.g., parking lot counts)
  - Social media signals (Reddit, Twitter/X, StockTwits)
  - Supply chain / shipping data
  - Options flow analysis (unusual activity detection)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
#  Types                                                                       #
# --------------------------------------------------------------------------- #

class AltDataSource(Enum):
    NEWS = "news"
    SEC_FILINGS = "sec_filings"
    SOCIAL_MEDIA = "social_media"
    SATELLITE = "satellite"
    OPTIONS_FLOW = "options_flow"
    SUPPLY_CHAIN = "supply_chain"
    WEB_TRAFFIC = "web_traffic"
    CREDIT_CARD = "credit_card"


class SentimentLabel(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class AltDataRecord:
    source: AltDataSource
    timestamp: int          # epoch nanoseconds
    symbol: str
    data_type: str
    value: float
    confidence: float = 1.0
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.record_id:
            content = f"{self.source.value}:{self.symbol}:{self.timestamp}:{self.data_type}"
            self.record_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class NewsArticle:
    title: str
    text: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    relevance_score: float = 0.0
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)


@dataclass
class SECFiling:
    ticker: str
    cik: str
    filing_type: str         # 10-K, 10-Q, 8-K, etc.
    filed_date: datetime
    accepted_date: datetime
    accession_number: str
    url: str
    text_content: str = ""
    risk_factors: List[str] = field(default_factory=list)
    key_metrics: Dict[str, float] = field(default_factory=dict)
    sentiment_score: float = 0.0


@dataclass
class OptionsFlowRecord:
    timestamp: int
    symbol: str
    expiry: str
    strike: float
    option_type: str     # "call" or "put"
    volume: int
    open_interest: int
    implied_vol: float
    delta: float
    premium: float
    is_sweep: bool = False
    is_unusual: bool = False
    vol_oi_ratio: float = 0.0
    block_trade: bool = False


# --------------------------------------------------------------------------- #
#  Sentiment Analyzer                                                          #
# --------------------------------------------------------------------------- #

class SentimentAnalyzer:
    """Financial sentiment analysis using FinBERT with rule-based fallback."""

    def __init__(self, use_gpu: bool = False) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if use_gpu else "cpu"
        self._initialized = False

    def initialize(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()
            self._initialized = True
            logger.info(f"FinBERT loaded on {self._device}")

        except ImportError:
            logger.warning("transformers/torch not installed, using rule-based sentiment")
            self._initialized = False

    def analyze(self, text: str) -> Tuple[float, SentimentLabel]:
        """Returns (score, label) where score is in [-1, 1]."""
        if self._initialized and self._model:
            return self._analyze_finbert(text)
        return self._analyze_rule_based(text)

    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, SentimentLabel]]:
        return [self.analyze(t) for t in texts]

    def _analyze_finbert(self, text: str) -> Tuple[float, SentimentLabel]:
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=512,
            truncation=True, padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # FinBERT: [positive, negative, neutral]
        pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
        score = pos - neg

        if score > 0.3:
            label = SentimentLabel.POSITIVE if score < 0.7 else SentimentLabel.VERY_POSITIVE
        elif score < -0.3:
            label = SentimentLabel.NEGATIVE if score > -0.7 else SentimentLabel.VERY_NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return score, label

    def _analyze_rule_based(self, text: str) -> Tuple[float, SentimentLabel]:
        text_lower = text.lower()

        positive_words = {
            "beat", "exceed", "strong", "growth", "surge", "rally", "upgrade",
            "outperform", "bullish", "profit", "gain", "record", "optimistic",
            "upside", "innovative", "breakthrough", "momentum",
        }
        negative_words = {
            "miss", "decline", "weak", "loss", "drop", "sell", "downgrade",
            "underperform", "bearish", "warning", "risk", "cut", "concern",
            "downside", "lawsuit", "investigation", "recession", "default",
        }

        words = set(re.findall(r'\b\w+\b', text_lower))
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count

        if total == 0:
            return 0.0, SentimentLabel.NEUTRAL

        score = (pos_count - neg_count) / total
        if score > 0.3:
            label = SentimentLabel.POSITIVE
        elif score < -0.3:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return score, label


# --------------------------------------------------------------------------- #
#  Alt Data Sources                                                            #
# --------------------------------------------------------------------------- #

class AltDataConnector(ABC):
    """Base class for alternative data source connectors."""

    @abstractmethod
    async def fetch(
        self, symbols: List[str], start: datetime, end: datetime,
    ) -> List[AltDataRecord]: ...

    @abstractmethod
    async def stream(
        self, symbols: List[str], callback: Callable[[AltDataRecord], None],
    ) -> None: ...


class NewsConnector(AltDataConnector):
    """News data ingestion from multiple providers (Benzinga, NewsAPI, etc.)."""

    def __init__(
        self,
        api_key: str = "",
        provider: str = "newsapi",
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    ) -> None:
        self._api_key = api_key
        self._provider = provider
        self._sentiment = sentiment_analyzer or SentimentAnalyzer()

    async def fetch(
        self, symbols: List[str], start: datetime, end: datetime,
    ) -> List[AltDataRecord]:
        """Fetch news articles and compute sentiment."""
        articles = await self._fetch_articles(symbols, start, end)
        records: List[AltDataRecord] = []

        for article in articles:
            score, label = self._sentiment.analyze(article.title + " " + article.text[:200])
            for symbol in article.symbols:
                records.append(AltDataRecord(
                    source=AltDataSource.NEWS,
                    timestamp=int(article.published_at.timestamp() * 1e9),
                    symbol=symbol,
                    data_type="sentiment",
                    value=score,
                    confidence=article.relevance_score,
                    raw_text=article.title,
                    metadata={
                        "source": article.source,
                        "url": article.url,
                        "label": label.name,
                    },
                ))

        logger.info(f"Fetched {len(records)} news records for {len(symbols)} symbols")
        return records

    async def _fetch_articles(
        self, symbols: List[str], start: datetime, end: datetime,
    ) -> List[NewsArticle]:
        if not self._api_key:
            logger.warning("NewsConnector: no api_key configured — skipping fetch")
            return []

        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed — cannot fetch news. pip install aiohttp")
            return []

        articles: List[NewsArticle] = []
        query = " OR ".join(symbols)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": self._api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        logger.error(f"NewsAPI returned {resp.status}: {await resp.text()}")
                        return []
                    data = await resp.json()

            for item in data.get("articles", []):
                published = datetime.fromisoformat(
                    item["publishedAt"].replace("Z", "+00:00")
                )
                matched_symbols = [s for s in symbols if s.lower() in (item.get("title", "") + item.get("description", "")).lower()]
                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    text=item.get("description", "") or "",
                    source=item.get("source", {}).get("name", ""),
                    url=item.get("url", ""),
                    published_at=published,
                    symbols=matched_symbols or symbols[:1],
                ))
        except Exception as exc:
            logger.error(f"NewsConnector fetch error: {exc}")

        return articles

    async def stream(
        self, symbols: List[str], callback: Callable[[AltDataRecord], None],
    ) -> None:
        """Stream real-time news (WebSocket or polling)."""
        import asyncio
        logger.info(f"Streaming news for {len(symbols)} symbols")
        while True:
            records = await self.fetch(
                symbols,
                datetime.now(tz=timezone.utc),
                datetime.now(tz=timezone.utc),
            )
            for r in records:
                callback(r)
            await asyncio.sleep(30)


class SECFilingsConnector(AltDataConnector):
    """SEC EDGAR filing ingestion (10-K, 10-Q, 8-K)."""

    def __init__(self, user_agent: str = "HRT Research research@example.com") -> None:
        self._user_agent = user_agent
        self._base_url = "https://efts.sec.gov/LATEST/search-index"

    async def fetch(
        self, symbols: List[str], start: datetime, end: datetime,
    ) -> List[AltDataRecord]:
        """Fetch and parse SEC filings."""
        records: List[AltDataRecord] = []

        for symbol in symbols:
            filings = await self._fetch_filings(symbol, start, end)
            for filing in filings:
                # Generate multiple signals per filing
                records.append(AltDataRecord(
                    source=AltDataSource.SEC_FILINGS,
                    timestamp=int(filing.filed_date.timestamp() * 1e9),
                    symbol=symbol,
                    data_type=f"filing_{filing.filing_type.lower().replace('-', '')}",
                    value=filing.sentiment_score,
                    metadata={
                        "filing_type": filing.filing_type,
                        "accession_number": filing.accession_number,
                        "risk_factors_count": len(filing.risk_factors),
                        "key_metrics": filing.key_metrics,
                    },
                ))

        return records

    async def _fetch_filings(
        self, symbol: str, start: datetime, end: datetime,
    ) -> List[SECFiling]:
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed — cannot fetch EDGAR filings")
            return []

        filings: List[SECFiling] = []
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f"\"{symbol}\"",
            "dateRange": "custom",
            "startdt": start.strftime("%Y-%m-%d"),
            "enddt": end.strftime("%Y-%m-%d"),
            "forms": "10-K,10-Q,8-K",
        }
        headers = {"User-Agent": self._user_agent}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"EDGAR returned {resp.status}")
                        return []
                    data = await resp.json()

            for hit in data.get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                filed = datetime.strptime(src.get("file_date", "2000-01-01"), "%Y-%m-%d")
                filings.append(SECFiling(
                    ticker=symbol,
                    cik=src.get("entity_id", ""),
                    filing_type=src.get("form_type", ""),
                    filed_date=filed,
                    accepted_date=filed,
                    accession_number=src.get("accession_no", ""),
                    url=f"https://www.sec.gov/Archives/edgar/data/{src.get('entity_id', '')}/{src.get('accession_no', '')}",
                    text_content=src.get("file_description", ""),
                ))
        except Exception as exc:
            logger.error(f"SECFilingsConnector fetch error: {exc}")

        return filings

    async def stream(
        self, symbols: List[str], callback: Callable[[AltDataRecord], None],
    ) -> None:
        import asyncio
        while True:
            records = await self.fetch(
                symbols,
                datetime.now(tz=timezone.utc),
                datetime.now(tz=timezone.utc),
            )
            for r in records:
                callback(r)
            await asyncio.sleep(300)  # Check every 5 min


class OptionsFlowConnector(AltDataConnector):
    """Unusual options activity detection (sweeps, blocks, vol/OI spikes)."""

    def __init__(
        self,
        vol_oi_threshold: float = 3.0,
        min_premium: float = 50_000,
    ) -> None:
        self._vol_oi_threshold = vol_oi_threshold
        self._min_premium = min_premium

    async def fetch(
        self, symbols: List[str], start: datetime, end: datetime,
    ) -> List[AltDataRecord]:
        """Fetch options flow and flag unusual activity. Requires a paid data provider."""
        raw_flow = await self._fetch_raw_flow(symbols, start, end)
        unusual = self.detect_unusual_activity(raw_flow)

        records: List[AltDataRecord] = []
        for rec in unusual:
            records.append(AltDataRecord(
                source=AltDataSource.OPTIONS_FLOW,
                timestamp=rec.timestamp,
                symbol=rec.symbol,
                data_type="unusual_options",
                value=rec.vol_oi_ratio,
                confidence=min(rec.vol_oi_ratio / self._vol_oi_threshold, 1.0),
                metadata={
                    "strike": rec.strike,
                    "expiry": rec.expiry,
                    "option_type": rec.option_type,
                    "premium": rec.premium,
                    "is_sweep": rec.is_sweep,
                    "block_trade": rec.block_trade,
                },
            ))
        return records

    async def _fetch_raw_flow(
        self, symbols: List[str], start: datetime, end: datetime,
    ) -> List[OptionsFlowRecord]:
        """Override this in a subclass to plug in CBOE / Tradier / Polygon data."""
        return []

    async def stream(
        self, symbols: List[str], callback: Callable[[AltDataRecord], None],
    ) -> None:
        import asyncio
        while True:
            records = await self.fetch(
                symbols,
                datetime.now(tz=timezone.utc),
                datetime.now(tz=timezone.utc),
            )
            for r in records:
                callback(r)
            await asyncio.sleep(10)

    def detect_unusual_activity(
        self, flow_records: List[OptionsFlowRecord],
    ) -> List[OptionsFlowRecord]:
        """Flag unusual options activity."""
        unusual = []
        for rec in flow_records:
            rec.vol_oi_ratio = (
                rec.volume / rec.open_interest
                if rec.open_interest > 0 else 0
            )
            is_unusual = (
                rec.vol_oi_ratio > self._vol_oi_threshold
                or rec.premium > self._min_premium
                or rec.is_sweep
                or rec.block_trade
            )
            if is_unusual:
                rec.is_unusual = True
                unusual.append(rec)

        return unusual


# --------------------------------------------------------------------------- #
#  Alt Data Pipeline Manager                                                   #
# --------------------------------------------------------------------------- #

class AltDataPipelineManager:
    """Orchestrates alt data sources and aggregates signals into features."""

    def __init__(self) -> None:
        self._connectors: Dict[AltDataSource, AltDataConnector] = {}
        self._records_buffer: List[AltDataRecord] = []
        self._signal_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def register_connector(self, source: AltDataSource, connector: AltDataConnector) -> None:
        self._connectors[source] = connector

    async def fetch_all(
        self, symbols: List[str], start: datetime, end: datetime,
    ) -> List[AltDataRecord]:
        """Fetch from all registered sources."""
        import asyncio
        tasks = [
            connector.fetch(symbols, start, end)
            for connector in self._connectors.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_records = []
        for result in results:
            if isinstance(result, list):
                all_records.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Alt data fetch failed: {result}")

        self._records_buffer.extend(all_records)
        return all_records

    def compute_composite_signal(
        self, symbol: str, records: List[AltDataRecord],
    ) -> float:
        """Confidence- and recency-weighted composite signal for a symbol."""
        if not records:
            return 0.0

        symbol_records = [r for r in records if r.symbol == symbol]
        if not symbol_records:
            return 0.0

        now = time.time_ns()
        weighted_sum = 0.0
        total_weight = 0.0

        for rec in symbol_records:
            # Time decay: half-life of 24 hours
            age_hours = (now - rec.timestamp) / 3.6e12
            time_weight = np.exp(-0.693 * age_hours / 24)

            weight = rec.confidence * time_weight
            weighted_sum += rec.value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def to_features(self, records: List[AltDataRecord]) -> Dict[str, Dict[str, float]]:
        """Convert alt data records to feature dictionaries, keyed by symbol."""
        features: Dict[str, Dict[str, float]] = defaultdict(dict)

        for rec in records:
            feature_name = f"alt_{rec.source.value}_{rec.data_type}"
            features[rec.symbol][feature_name] = rec.value
            features[rec.symbol][f"{feature_name}_confidence"] = rec.confidence

        # Add composite signals
        symbols = set(r.symbol for r in records)
        for symbol in symbols:
            features[symbol]["alt_composite_signal"] = self.compute_composite_signal(
                symbol, records
            )

        return dict(features)
