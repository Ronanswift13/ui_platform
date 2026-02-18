"""
Base adapter definition.

Provides an abstract base class for hardware/data adapters that plugins
can use to connect to cameras, sensors, and other data sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class AdapterStatus(str, Enum):
    """Adapter connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    SIMULATED = "simulated"


class BaseAdapter(ABC):
    """
    Base adapter for hardware/data source connections.

    Adapters wrap cameras, sensors, or data feeds so plugins can operate
    against real hardware or simulated sources interchangeably.
    """

    def __init__(self) -> None:
        self._status: AdapterStatus = AdapterStatus.DISCONNECTED
        self._error_message: str = ""

    @property
    def status(self) -> AdapterStatus:
        """Current adapter status."""
        return self._status

    @property
    def is_simulated(self) -> bool:
        """Whether this adapter is running in simulation mode."""
        return self._status == AdapterStatus.SIMULATED

    @abstractmethod
    async def connect(self, config: dict[str, Any] | None = None) -> bool:
        """
        Connect to the data source.

        Args:
            config: Optional connection configuration.

        Returns:
            Whether the connection succeeded.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass

    def get_error(self) -> str:
        """Return the last error message."""
        return self._error_message

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self._status.value}]>"
