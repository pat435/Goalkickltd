"""
Trading pairs configuration module for the Goalkick Ltd Trading Bot.
Defines the tradable assets with their specific parameters.
"""

class TradingPair:
    """Class representing a trading pair with its specific parameters."""
    
    def __init__(
        self,
        symbol,
        base_currency,
        quote_currency,
        min_order_size,
        price_precision,
        quantity_precision,
        min_notional,
        is_active=True,
        timeframes=None,
        lot_size_filter=None,
        price_filter=None,
        market_session_weights=None,
    ):
        self.symbol = symbol
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.min_order_size = min_order_size
        self.price_precision = price_precision
        self.quantity_precision = quantity_precision
        self.min_notional = min_notional
        self.is_active = is_active
        self.timeframes = timeframes or ["5m", "15m", "1h", "4h", "1d"]
        self.lot_size_filter = lot_size_filter or {}
        self.price_filter = price_filter or {}
        # Market session weights for trading (Asian, European, US)
        self.market_session_weights = market_session_weights or {
            "Asian": 1.0,
            "European": 1.0,
            "US": 1.0
        }
    
    def __str__(self):
        return f"{self.symbol} ({self.base_currency}/{self.quote_currency})"
    
    def to_dict(self):
        """Convert the trading pair to a dictionary."""
        return {
            "symbol": self.symbol,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "min_order_size": self.min_order_size,
            "price_precision": self.price_precision,
            "quantity_precision": self.quantity_precision,
            "min_notional": self.min_notional,
            "is_active": self.is_active,
            "timeframes": self.timeframes,
            "lot_size_filter": self.lot_size_filter,
            "price_filter": self.price_filter,
            "market_session_weights": self.market_session_weights,
        }
    
    @classmethod
    def from_exchange_info(cls, symbol_info):
        """Create a TradingPair instance from exchange symbol information."""
        # This would be implemented according to the specific exchange API
        # For now, returning a basic implementation
        return cls(
            symbol=symbol_info.get("symbol"),
            base_currency=symbol_info.get("baseAsset"),
            quote_currency=symbol_info.get("quoteAsset"),
            min_order_size=symbol_info.get("minOrderQty", 0.001),
            price_precision=symbol_info.get("pricePrecision", 2),
            quantity_precision=symbol_info.get("quantityPrecision", 3),
            min_notional=symbol_info.get("minNotional", 10),
            lot_size_filter=symbol_info.get("lot_size_filter", {}),
            price_filter=symbol_info.get("price_filter", {})
        )


# Default trading pairs for Bybit
BYBIT_TRADING_PAIRS = {
    "BTCUSDT": TradingPair(
        symbol="BTCUSDT",
        base_currency="BTC",
        quote_currency="USDT",
        min_order_size=0.001,
        price_precision=2,
        quantity_precision=3,
        min_notional=10,
        lot_size_filter={
            "min_trading_qty": 0.001,
            "max_trading_qty": 100,
            "qty_step": 0.001
        },
        price_filter={
            "min_price": 0.5,
            "max_price": 999999,
            "tick_size": 0.5
        }
    ),
    "ETHUSDT": TradingPair(
        symbol="ETHUSDT",
        base_currency="ETH",
        quote_currency="USDT",
        min_order_size=0.01,
        price_precision=2,
        quantity_precision=3,
        min_notional=10,
        lot_size_filter={
            "min_trading_qty": 0.01,
            "max_trading_qty": 1000,
            "qty_step": 0.01
        },
        price_filter={
            "min_price": 0.05,
            "max_price": 999999,
            "tick_size": 0.05
        }
    ),
    "SOLUSDT": TradingPair(
        symbol="SOLUSDT",
        base_currency="SOL",
        quote_currency="USDT",
        min_order_size=0.1,
        price_precision=3,
        quantity_precision=1,
        min_notional=10,
        lot_size_filter={
            "min_trading_qty": 0.1,
            "max_trading_qty": 10000,
            "qty_step": 0.1
        },
        price_filter={
            "min_price": 0.001,
            "max_price": 999999,
            "tick_size": 0.001
        }
    ),
    "BNBUSDT": TradingPair(
        symbol="BNBUSDT",
        base_currency="BNB",
        quote_currency="USDT",
        min_order_size=0.01,
        price_precision=2,
        quantity_precision=2,
        min_notional=10,
        lot_size_filter={
            "min_trading_qty": 0.01,
            "max_trading_qty": 1000,
            "qty_step": 0.01
        },
        price_filter={
            "min_price": 0.01,
            "max_price": 999999,
            "tick_size": 0.01
        }
    ),
    "ADAUSDT": TradingPair(
        symbol="ADAUSDT",
        base_currency="ADA",
        quote_currency="USDT",
        min_order_size=1,
        price_precision=4,
        quantity_precision=0,
        min_notional=10,
        lot_size_filter={
            "min_trading_qty": 1,
            "max_trading_qty": 1000000,
            "qty_step": 1
        },
        price_filter={
            "min_price": 0.0001,
            "max_price": 999999,
            "tick_size": 0.0001
        }
    ),
}

# Define active trading pairs
ACTIVE_TRADING_PAIRS = {
    symbol: pair for symbol, pair in BYBIT_TRADING_PAIRS.items() 
    if pair.is_active
}

def get_trading_pair(symbol):
    """Get a trading pair by symbol."""
    return ACTIVE_TRADING_PAIRS.get(symbol)

def get_all_active_symbols():
    """Get all active trading symbols."""
    return list(ACTIVE_TRADING_PAIRS.keys())

def get_base_currencies():
    """Get all unique base currencies."""
    return list(set(pair.base_currency for pair in ACTIVE_TRADING_PAIRS.values()))

def get_quote_currencies():
    """Get all unique quote currencies."""
    return list(set(pair.quote_currency for pair in ACTIVE_TRADING_PAIRS.values()))