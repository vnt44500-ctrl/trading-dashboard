"""Broker adapter layer for future automated trading.

Defines a pluggable interface. Concrete adapters (IBKR, Alpaca, Binance,
etc.) implement `BrokerAdapter`. A `PaperBroker` simulates execution for
testing. Swap `active_broker` in config.py or .env to change brokers.
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BrokerAdapter(ABC):
    """Interface all broker adapters must implement."""

    name = "base"

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection; return success."""

    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "market") -> dict:
        """Place an order. side in {'buy','sell'}."""

    @abstractmethod
    def get_positions(self) -> list:
        """Return open positions."""

    @abstractmethod
    def get_account(self) -> dict:
        """Return account summary."""


class PaperBroker(BrokerAdapter):
    """Simulated broker for testing the automation flow end-to-end.

    ``cash`` and realised P&L are now updated by fills; previously the account
    summary always reported a hard-coded 100,000 balance that no order changed.
    """

    name = "paper"

    def __init__(self, starting_cash: float = 100000.0):
        self._orders = []
        self._positions = {}
        self._cash = float(starting_cash)
        self._realized_pnl = 0.0
        self._last_price = {}
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "market",
                    price: float | None = None) -> dict:
        symbol = symbol.upper()
        if quantity <= 0:
            return {"status": "rejected", "symbol": symbol, "detail": "quantity must be positive"}
        fill_price = float(price) if price else float(self._last_price.get(symbol, 0.0))
        if fill_price <= 0:
            return {"status": "rejected", "symbol": symbol,
                    "detail": "a positive reference price is required for a paper fill"}

        signed = quantity if side == "buy" else -quantity
        previous = self._positions.get(symbol, 0.0)
        order = {
            "id": len(self._orders) + 1,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "fill_price": round(fill_price, 6),
            "order_type": order_type,
            "status": "filled",
        }

        # Realise P&L when the fill reduces or flips an existing position.
        if previous != 0 and (previous > 0) != (signed > 0):
            closing = min(abs(signed), abs(previous))
            direction = 1 if previous > 0 else -1
            entry = self._last_price.get(symbol, fill_price)
            self._realized_pnl += (fill_price - entry) * closing * direction

        self._positions[symbol] = previous + signed
        self._cash -= signed * fill_price
        self._last_price[symbol] = fill_price
        self._orders.append(order)
        if self._positions[symbol] == 0:
            self._positions.pop(symbol, None)
        return order

    def mark_price(self, symbol: str, price: float):
        """Record a reference price so subsequent paper fills are realistic."""
        self._last_price[symbol.upper()] = float(price)

    def get_positions(self) -> list:
        return [{"symbol": symbol, "quantity": quantity, "last_price": self._last_price.get(symbol)}
                for symbol, quantity in self._positions.items() if quantity != 0]

    def get_account(self) -> dict:
        market_value = sum(
            quantity * self._last_price.get(symbol, 0.0)
            for symbol, quantity in self._positions.items()
        )
        return {
            "cash": round(self._cash, 2),
            "market_value": round(market_value, 2),
            "equity": round(self._cash + market_value, 2),
            "realized_pnl": round(self._realized_pnl, 2),
            "orders": len(self._orders),
            "positions": self.get_positions(),
        }


class AlpacaBroker(BrokerAdapter):
    """Alpaca adapter (stub — requires alpaca-py and API keys)."""

    name = "alpaca"

    def __init__(self, api_key: str = "", secret_key: str = "", paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._api = None

    def connect(self) -> bool:
        try:
            from alpaca.trading.client import TradingClient  # optional dependency
            self._api = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            return True
        except Exception:
            logger.exception("Alpaca connection failed.")
            return False

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "market") -> dict:
        if self._api is None:
            return {"status": "not_connected", "symbol": symbol, "side": side,
                    "detail": "Call connect() with valid Alpaca credentials first."}
        return {"status": "not_implemented", "symbol": symbol, "side": side}

    def get_positions(self) -> list:
        if self._api is None:
            return []
        try:
            return [str(position) for position in self._api.get_all_positions()]
        except Exception:
            logger.exception("Failed to read Alpaca positions.")
            return []

    def get_account(self) -> dict:
        if self._api is None:
            return {}
        try:
            account = self._api.get_account()
            return {"cash": account.cash, "equity": account.equity, "buying_power": account.buying_power}
        except Exception:
            logger.exception("Failed to read Alpaca account.")
            return {}


def get_broker(name: str = None):
    """Return a broker adapter instance.

    The Alpaca branch previously passed ``news_api_key`` and
    ``alpha_vantage_key`` as the trading credentials, so a configured Alpaca
    account could never authenticate.
    """
    from config import config
    name = name or config.active_broker
    if name == "alpaca":
        return AlpacaBroker(config.alpaca_api_key, config.alpaca_api_secret)
    return PaperBroker()