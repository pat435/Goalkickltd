# Goalkick Ltd Trading Bot Documentation

## Overview
The Goalkick Ltd Trading Bot is an automated cryptocurrency trading bot designed to trade on the Bybit exchange. It leverages technical indicators, risk management strategies, and API integration to execute trades systematically. The bot is implemented in Python and relies on the `pybit` library for API interactions.

## Project Structure
```
Goalkickltd/
│── .env
│── .gitignore
│── README.md
│── requirements.txt
│
├── sandbox/
│   ├── funds.py
│
├── src/
│   ├── main.py
│   ├── create_signal_check.py
│
├── test/
│   ├── bot_connection.py
│   ├── test.py
│   ├── test2.py
```

### File Descriptions
- **`.env`**: Stores environment variables such as API keys.
- **`requirements.txt`**: Lists the required dependencies.
- **`README.md`**: Provides a general overview of the project.
- **`sandbox/funds.py`**: Contains financial calculations or experiments.
- **`src/main.py`**: The core trading bot implementation.
- **`src/create_signal_check.py`**: Handles signal generation for trades.
- **`test/`**: Contains testing scripts to verify bot functionality.

---

## `main.py` - Core Trading Logic

### Initialization
The bot is implemented as a class named `AdvancedTradingBot`. When instantiated, it:
- Initializes an API session using Bybit API keys stored in the `.env` file.
- Sets up trading parameters, such as `symbol`, risk per trade, and trade tracking variables.
- Calls `setup_strategies()` to define technical analysis indicators.

```python
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()

class AdvancedTradingBot:
    def __init__(self):
        self.session = HTTP(
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
            testnet=True
        )
        self.symbol = "ETHUSDT"
        self.risk_per_trade = 0.02
        self.setup_strategies()
```

### Equity Check
The `get_equity()` function retrieves the current balance from the exchange and handles errors gracefully.

```python
def get_equity(self):
    try:
        response = self.session.get_wallet_balance(accountType="UNIFIED")
        return response["result"]["totalEquity"]
    except Exception as e:
        print("Error fetching equity:", str(e))
        return 0
```

### Strategy Setup
The `setup_strategies()` method configures technical indicators such as moving averages and RSI.

```python
def setup_strategies(self):
    self.strategy = {
        "rsi": 14,
        "ema_short": 9,
        "ema_long": 21
    }
```

### Trade Execution
The bot places orders using the `execute_trade()` function, which considers risk per trade and strategy signals.

```python
def execute_trade(self, direction):
    if direction == "BUY":
        order = self.session.place_order(symbol=self.symbol, side="Buy", qty=1, orderType="Market")
    elif direction == "SELL":
        order = self.session.place_order(symbol=self.symbol, side="Sell", qty=1, orderType="Market")
```

### Scheduled Execution
The bot runs periodically using the `schedule` library.

```python
import schedule

def run(self):
    schedule.every(5).minutes.do(self.check_market)
    while True:
        schedule.run_pending()
        time.sleep(1)
```

---

## Running the Bot
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a `.env` file and add:
```ini
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
```

### 3. Start the Bot
```bash
python src/main.py
```

---

## Conclusion
This trading bot automates crypto trading by leveraging Bybit API, technical analysis, and scheduled execution. Future improvements could include backtesting, advanced risk management, and machine learning-based strategies.

