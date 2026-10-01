"""Structured JSON logging and CloudWatch Embedded Metric Format (EMF).

On ECS the awslogs driver ships stdout to CloudWatch Logs, which extracts EMF lines into
metrics, so no agent or PutMetricData calls are needed.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, ClassVar

NAMESPACE = "AgenticRag"


class JsonFormatter(logging.Formatter):
    _RESERVED: ClassVar[frozenset[str]] = frozenset(vars(logging.makeLogRecord({})))

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in self._RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def emit_metrics(
    metrics: dict[str, float],
    dimensions: dict[str, str] | None = None,
    units: dict[str, str] | None = None,
    properties: dict[str, Any] | None = None,
) -> None:
    dimensions = dimensions or {"Service": "api"}
    units = units or {}
    document: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [list(dimensions)],
                    "Metrics": [
                        {"Name": name, "Unit": units.get(name, "None")} for name in metrics
                    ],
                }
            ],
        },
        **dimensions,
        **metrics,
        **(properties or {}),
    }
    sys.stdout.write(json.dumps(document) + "\n")
    sys.stdout.flush()
