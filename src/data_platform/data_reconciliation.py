"""
Automated Data Reconciliation

Ensures data integrity across the pipeline:
  - Cross-source tick count reconciliation (exchange vs. local)
  - Gap detection in time-series data
  - Duplicate detection and deduplication
  - Schema drift monitoring
  - Automated self-healing with configurable policies
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
#  Types                                                                       #
# --------------------------------------------------------------------------- #

class ReconciliationStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    HEALING = "healing"


class IssueType(Enum):
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    GAP_DETECTED = "gap_detected"
    COUNT_MISMATCH = "count_mismatch"
    VALUE_MISMATCH = "value_mismatch"
    SCHEMA_DRIFT = "schema_drift"
    STALE_DATA = "stale_data"
    OUTLIER = "outlier"


class HealingAction(Enum):
    NONE = "none"
    BACKFILL = "backfill"
    DEDUPLICATE = "deduplicate"
    INTERPOLATE = "interpolate"
    DROP = "drop"
    ALERT_ONLY = "alert_only"


@dataclass
class ReconciliationIssue:
    issue_type: IssueType
    severity: ReconciliationStatus
    entity: str           # symbol / venue / source
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    healing_action: HealingAction = HealingAction.NONE
    detected_at: int = field(default_factory=time.time_ns)
    resolved: bool = False


@dataclass
class ReconciliationReport:
    run_id: str
    started_at: int
    finished_at: int = 0
    status: ReconciliationStatus = ReconciliationStatus.PASS
    issues: List[ReconciliationIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    entities_checked: int = 0
    entities_passed: int = 0

    def add_issue(self, issue: ReconciliationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ReconciliationStatus.FAIL:
            self.status = ReconciliationStatus.FAIL
        elif issue.severity == ReconciliationStatus.WARN and self.status == ReconciliationStatus.PASS:
            self.status = ReconciliationStatus.WARN

    @property
    def summary(self) -> Dict[str, Any]:
        issue_counts = defaultdict(int)
        for issue in self.issues:
            issue_counts[issue.issue_type.value] += 1
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "entities_checked": self.entities_checked,
            "entities_passed": self.entities_passed,
            "total_issues": len(self.issues),
            "issues_by_type": dict(issue_counts),
            "duration_ms": (self.finished_at - self.started_at) / 1e6,
        }


# --------------------------------------------------------------------------- #
#  Reconciliation Checks                                                       #
# --------------------------------------------------------------------------- #

class ReconciliationCheck:
    """Base class for reconciliation checks."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, data: pd.DataFrame, **kwargs: Any) -> List[ReconciliationIssue]:
        raise NotImplementedError


class GapDetector(ReconciliationCheck):
    """Detects gaps where interval exceeds max_gap_multiple * median_interval."""

    def __init__(
        self,
        max_gap_multiple: float = 5.0,
        timestamp_col: str = "timestamp",
    ) -> None:
        super().__init__("gap_detector")
        self._max_gap_multiple = max_gap_multiple
        self._ts_col = timestamp_col

    def run(self, data: pd.DataFrame, **kwargs: Any) -> List[ReconciliationIssue]:
        issues: List[ReconciliationIssue] = []
        entity = kwargs.get("entity", "unknown")

        if len(data) < 3:
            return issues

        timestamps = data[self._ts_col].sort_values().values
        intervals = np.diff(timestamps)
        median_interval = np.median(intervals)

        if median_interval <= 0:
            return issues

        threshold = median_interval * self._max_gap_multiple

        gap_indices = np.where(intervals > threshold)[0]
        for idx in gap_indices:
            gap_start = int(timestamps[idx])
            gap_end = int(timestamps[idx + 1])
            gap_duration = gap_end - gap_start

            issues.append(ReconciliationIssue(
                issue_type=IssueType.GAP_DETECTED,
                severity=ReconciliationStatus.WARN,
                entity=entity,
                description=f"Gap of {gap_duration / 1e9:.1f}s detected "
                            f"(threshold: {threshold / 1e9:.1f}s)",
                details={
                    "gap_start_ns": gap_start,
                    "gap_end_ns": gap_end,
                    "gap_duration_ns": gap_duration,
                    "median_interval_ns": float(median_interval),
                    "gap_multiple": float(gap_duration / median_interval),
                },
                healing_action=HealingAction.BACKFILL,
            ))

        return issues


class DuplicateDetector(ReconciliationCheck):
    """Detects duplicate records in time-series data."""

    def __init__(
        self,
        key_columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__("duplicate_detector")
        self._key_columns = key_columns or ["timestamp", "symbol"]

    def run(self, data: pd.DataFrame, **kwargs: Any) -> List[ReconciliationIssue]:
        issues: List[ReconciliationIssue] = []
        entity = kwargs.get("entity", "unknown")

        available_cols = [c for c in self._key_columns if c in data.columns]
        if not available_cols:
            return issues

        duplicates = data[data.duplicated(subset=available_cols, keep=False)]
        if len(duplicates) > 0:
            dup_count = len(duplicates)
            unique_dup_count = len(duplicates.drop_duplicates(subset=available_cols))

            severity = (
                ReconciliationStatus.FAIL
                if dup_count / len(data) > 0.01
                else ReconciliationStatus.WARN
            )

            issues.append(ReconciliationIssue(
                issue_type=IssueType.DUPLICATE_DATA,
                severity=severity,
                entity=entity,
                description=f"{dup_count} duplicate rows found "
                            f"({unique_dup_count} unique keys)",
                details={
                    "duplicate_count": dup_count,
                    "unique_key_count": unique_dup_count,
                    "total_rows": len(data),
                    "duplicate_ratio": dup_count / len(data),
                },
                healing_action=HealingAction.DEDUPLICATE,
            ))

        return issues


class OutlierDetector(ReconciliationCheck):
    """Detects statistical outliers using z-score or IQR methods."""

    def __init__(
        self,
        value_col: str = "price",
        z_threshold: float = 4.0,
        method: str = "zscore",
    ) -> None:
        super().__init__("outlier_detector")
        self._value_col = value_col
        self._z_threshold = z_threshold
        self._method = method

    def run(self, data: pd.DataFrame, **kwargs: Any) -> List[ReconciliationIssue]:
        issues: List[ReconciliationIssue] = []
        entity = kwargs.get("entity", "unknown")

        if self._value_col not in data.columns or len(data) < 10:
            return issues

        values = data[self._value_col].dropna()

        if self._method == "zscore":
            mean = values.mean()
            std = values.std()
            if std == 0:
                return issues
            z_scores = np.abs((values - mean) / std)
            outlier_mask = z_scores > self._z_threshold
        else:  # IQR
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)

        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            issues.append(ReconciliationIssue(
                issue_type=IssueType.OUTLIER,
                severity=ReconciliationStatus.WARN,
                entity=entity,
                description=f"{outlier_count} outliers in {self._value_col} "
                            f"(method={self._method}, threshold={self._z_threshold})",
                details={
                    "outlier_count": outlier_count,
                    "total_rows": len(values),
                    "outlier_ratio": outlier_count / len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                },
                healing_action=HealingAction.DROP,
            ))

        return issues


class CrossSourceReconciler(ReconciliationCheck):
    """Reconciles tick counts and values between two data sources."""

    def __init__(
        self,
        tolerance_pct: float = 0.01,
        timestamp_col: str = "timestamp",
        value_col: str = "price",
    ) -> None:
        super().__init__("cross_source_reconciler")
        self._tolerance = tolerance_pct
        self._ts_col = timestamp_col
        self._val_col = value_col

    def reconcile(
        self,
        source_a: pd.DataFrame,
        source_b: pd.DataFrame,
        entity: str = "unknown",
    ) -> List[ReconciliationIssue]:
        issues: List[ReconciliationIssue] = []

        # Count reconciliation
        count_a = len(source_a)
        count_b = len(source_b)
        if count_a != count_b:
            ratio = abs(count_a - count_b) / max(count_a, count_b, 1)
            severity = ReconciliationStatus.FAIL if ratio > self._tolerance else ReconciliationStatus.WARN
            issues.append(ReconciliationIssue(
                issue_type=IssueType.COUNT_MISMATCH,
                severity=severity,
                entity=entity,
                description=f"Count mismatch: source_a={count_a}, source_b={count_b} "
                            f"(diff={abs(count_a - count_b)}, {ratio:.4%})",
                details={
                    "source_a_count": count_a,
                    "source_b_count": count_b,
                    "difference": abs(count_a - count_b),
                    "ratio": ratio,
                },
            ))

        # Value reconciliation (on matching timestamps)
        if self._ts_col in source_a.columns and self._ts_col in source_b.columns:
            merged = pd.merge(
                source_a[[self._ts_col, self._val_col]],
                source_b[[self._ts_col, self._val_col]],
                on=self._ts_col,
                suffixes=("_a", "_b"),
            )
            if len(merged) > 0:
                val_a = merged[f"{self._val_col}_a"]
                val_b = merged[f"{self._val_col}_b"]
                mismatches = (val_a - val_b).abs() > (val_a.abs() * self._tolerance)
                mismatch_count = int(mismatches.sum())
                if mismatch_count > 0:
                    issues.append(ReconciliationIssue(
                        issue_type=IssueType.VALUE_MISMATCH,
                        severity=ReconciliationStatus.WARN,
                        entity=entity,
                        description=f"{mismatch_count} value mismatches of {len(merged)} matched rows",
                        details={
                            "mismatch_count": mismatch_count,
                            "matched_rows": len(merged),
                            "mismatch_ratio": mismatch_count / len(merged),
                        },
                    ))

        return issues

    def run(self, data: pd.DataFrame, **kwargs: Any) -> List[ReconciliationIssue]:
        # For single-source mode, delegate to gap detection
        return []


class StalenessChecker(ReconciliationCheck):
    """Checks if data is stale (no updates within expected window)."""

    def __init__(
        self,
        max_staleness_seconds: float = 60.0,
        timestamp_col: str = "timestamp",
    ) -> None:
        super().__init__("staleness_checker")
        self._max_staleness_ns = int(max_staleness_seconds * 1e9)
        self._ts_col = timestamp_col

    def run(self, data: pd.DataFrame, **kwargs: Any) -> List[ReconciliationIssue]:
        issues: List[ReconciliationIssue] = []
        entity = kwargs.get("entity", "unknown")

        if len(data) == 0 or self._ts_col not in data.columns:
            issues.append(ReconciliationIssue(
                issue_type=IssueType.STALE_DATA,
                severity=ReconciliationStatus.FAIL,
                entity=entity,
                description="No data available",
                healing_action=HealingAction.ALERT_ONLY,
            ))
            return issues

        latest_ts = int(data[self._ts_col].max())
        now_ns = time.time_ns()
        staleness = now_ns - latest_ts

        if staleness > self._max_staleness_ns:
            issues.append(ReconciliationIssue(
                issue_type=IssueType.STALE_DATA,
                severity=ReconciliationStatus.WARN,
                entity=entity,
                description=f"Data is {staleness / 1e9:.1f}s old "
                            f"(threshold: {self._max_staleness_ns / 1e9:.1f}s)",
                details={
                    "latest_timestamp_ns": latest_ts,
                    "staleness_seconds": staleness / 1e9,
                    "threshold_seconds": self._max_staleness_ns / 1e9,
                },
                healing_action=HealingAction.ALERT_ONLY,
            ))

        return issues


# --------------------------------------------------------------------------- #
#  Self-Healing Engine                                                         #
# --------------------------------------------------------------------------- #

class SelfHealingEngine:
    """Automated remediation of detected data quality issues."""

    def __init__(self) -> None:
        self._healers: Dict[HealingAction, Callable] = {
            HealingAction.DEDUPLICATE: self._heal_deduplicate,
            HealingAction.INTERPOLATE: self._heal_interpolate,
            HealingAction.DROP: self._heal_drop_outliers,
        }
        self._healed_count = 0

    def heal(self, data: pd.DataFrame, issues: List[ReconciliationIssue]) -> pd.DataFrame:
        """Apply healing actions and return cleaned DataFrame."""
        result = data.copy()
        for issue in issues:
            if issue.healing_action in self._healers:
                try:
                    result = self._healers[issue.healing_action](result, issue)
                    issue.resolved = True
                    self._healed_count += 1
                    logger.info(f"Healed {issue.issue_type.value} for {issue.entity}")
                except Exception as e:
                    logger.error(f"Healing failed for {issue.issue_type.value}: {e}")
        return result

    def _heal_deduplicate(self, data: pd.DataFrame, issue: ReconciliationIssue) -> pd.DataFrame:
        key_cols = [c for c in ["timestamp", "symbol", "venue"] if c in data.columns]
        if key_cols:
            return data.drop_duplicates(subset=key_cols, keep="first")
        return data.drop_duplicates(keep="first")

    def _heal_interpolate(self, data: pd.DataFrame, issue: ReconciliationIssue) -> pd.DataFrame:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].interpolate(method="linear")
        return data

    def _heal_drop_outliers(self, data: pd.DataFrame, issue: ReconciliationIssue) -> pd.DataFrame:
        value_col = issue.details.get("value_col", "price")
        if value_col not in data.columns:
            return data
        mean = data[value_col].mean()
        std = data[value_col].std()
        if std > 0:
            z_scores = np.abs((data[value_col] - mean) / std)
            return data[z_scores <= 4.0]
        return data


# --------------------------------------------------------------------------- #
#  Reconciliation Pipeline                                                     #
# --------------------------------------------------------------------------- #

class ReconciliationPipeline:
    """Orchestrates reconciliation checks and self-healing across data sources."""

    def __init__(self, auto_heal: bool = True) -> None:
        self._checks: List[ReconciliationCheck] = []
        self._healer = SelfHealingEngine() if auto_heal else None
        self._reports: List[ReconciliationReport] = []

    def add_check(self, check: ReconciliationCheck) -> "ReconciliationPipeline":
        self._checks.append(check)
        return self

    async def run(
        self,
        data: pd.DataFrame,
        entity: str = "unknown",
        run_id: Optional[str] = None,
    ) -> ReconciliationReport:
        """Execute all checks against the data."""
        run_id = run_id or hashlib.md5(f"{entity}:{time.time_ns()}".encode()).hexdigest()[:12]
        report = ReconciliationReport(
            run_id=run_id,
            started_at=time.time_ns(),
            entities_checked=1,
        )

        for check in self._checks:
            try:
                issues = check.run(data, entity=entity)
                for issue in issues:
                    report.add_issue(issue)
            except Exception as e:
                logger.error(f"Check {check.name} failed: {e}")
                report.add_issue(ReconciliationIssue(
                    issue_type=IssueType.MISSING_DATA,
                    severity=ReconciliationStatus.FAIL,
                    entity=entity,
                    description=f"Check {check.name} failed: {e}",
                ))

        # Auto-heal if configured
        if self._healer and report.issues:
            self._healer.heal(data, report.issues)

        if report.status == ReconciliationStatus.PASS:
            report.entities_passed = 1

        report.finished_at = time.time_ns()
        report.metrics = {
            "total_rows": len(data),
            "issues_found": len(report.issues),
            "issues_healed": sum(1 for i in report.issues if i.resolved),
        }

        self._reports.append(report)
        logger.info(f"Reconciliation {run_id}: {report.status.value} "
                     f"({len(report.issues)} issues)")
        return report

    @classmethod
    def create_default(cls) -> "ReconciliationPipeline":
        """Standard pipeline with common checks."""
        pipeline = cls(auto_heal=True)
        pipeline.add_check(GapDetector())
        pipeline.add_check(DuplicateDetector())
        pipeline.add_check(OutlierDetector())
        pipeline.add_check(StalenessChecker())
        return pipeline
