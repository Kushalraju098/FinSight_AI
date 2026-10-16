from fastmcp import FastMCP
import yfinance as yf
import feedparser

mcp= FastMCP("FinSight MCP")

@mcp.tool
def get_stock_price(symbl:str)->dict:
    """
    Get the latest stock price for a stock symbol.
    Args:
        symbl: Stock ticker symbol, e.g. AAPL, MSFT, NVDA.
    """
    ticker= yf.Ticker(symbl)
    live_price= ticker.fast_info["last_price"]
    return {symbl:float(live_price)}

@mcp.tool
def get_stock_price_history(symbl:str,period:str="1mo",interval:str="1d")->dict:
    """
    Get historical stock prices.
    Args:
        symbl: Stock ticker, e.g. AAPL.
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, etc.
        interval: 1m, 5m, 15m, 1h, 1d, 1wk, etc.
    """
    ticker=yf.Ticker(symbl)
    data=ticker.history(period=period,interval=interval)
    if data.empty:
        return {
            "success": False,
            "error": f"No historical data found for {symbl}"
        }

    data = data.reset_index()

    records = []

    for _, row in data.iterrows():
        records.append({
            "date": str(row["Datetime"] if "Datetime" in row else row["Date"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })

    return {
        "symbol": symbl.upper(),
        "period": period,
        "interval": interval,
        "data": records
    }


@mcp.tool
def get_stock_news(symbl: str) -> dict:
    """
    Get the latest news headlines and summaries for a publicly
    traded company's stock from Yahoo Finance RSS.

    Use this tool when the user asks for:
    - latest news
    - recent news
    - current news
    - relevant news
    - news headlines
    - recent company developments

    This tool should be used when the question requires
    current company news in addition to other market data.

    Args:
        symbl: Stock ticker symbol, e.g. AAPL, MSFT, NVDA.
    """

    try:
        symbol = symbl.upper()

        url = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={symbol}"
            "&region=US"
            "&lang=en-US"
        )

        feed = feedparser.parse(url)

        news = []

        for entry in feed.entries[:10]:
            news.append({
                "title": entry.get("title"),
                "summary": entry.get("summary"),
                "published": entry.get("published"),
                "url": entry.get("link")
            })

        return {
            "success": True,
            "symbol": symbol,
            "source": "Yahoo Finance RSS",
            "news": news
        }

    except Exception as e:
        return {
            "success": False,
            "symbol": symbl.upper(),
            "source": "Yahoo Finance RSS",
            "news": [],
            "error": str(e)
        }

if __name__=="__main__":
    mcp.run()
