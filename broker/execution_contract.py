"""Broker execution contract.

This defines the interface a broker implementation must satisfy. Today the app
ships a paper/simulated implementation. A real broker can be added later by
implementing this contract.

IMPORTANT: this app does not claim to be a live ECN execution system. The
execution layer is paper until a real broker is connected and configured.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BrokerExecution(Protocol):
    """Interface for order placement, cancellation, and position/price lookup.

    Implementations may be paper/simulated or connected to a real broker.
    """

    name: str

    def get_live_quote(self, symbol: str) -> dict | None:
        """Return the latest available quote for *symbol*.

        Expected keys: ``symbol``, ``price``, ``bid``, ``ask``, ``timestamp``.
        May return ``None`` if no quote is available.
        """

    def get_live_price(self, symbol: str) -> float | None:
        """Return a single live price for *symbol*, or ``None``."""

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "market", price: float | None = None,
                    time_in_force: str = "day") -> dict:
        """Place an order and return an order result.

        Expected result keys: ``order_id``, ``status``, ``symbol``, ``side``,
        ``quantity``, ``filled_quantity``, ``avg_price``, ``state``.
        """

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order by id and return a cancel result.

        Expected result keys: ``order_id``, ``status``, ``cancelled``.
        """

    def get_position(self, symbol: str) -> dict | None:
        """Return the current position for *symbol*, or ``None``.

        Expected keys: ``symbol``, ``quantity``, ``avg_entry_price``.
        """


def is_paper(broker: BrokerExecution) -> bool:
    """Return True if *broker* is the paper/simulated implementation.

    Extension hooks for a real broker should override this or be clearly
    distinct from the paper implementation.
    """
    return getattr(broker, "name", "").lower() == "paper"