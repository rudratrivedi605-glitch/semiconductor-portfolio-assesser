"""
Semiconductor Portfolio Assesser
---------------------------------
Built this to analyze my own semiconductor holdings and get a clearer picture
of concentration risk, diversification, and performance across sub-sectors.

Notes:
- yfinance occasionally returns stale or missing data for tickers, added
  basic error handling around each fetch
- Sharpe here uses annualized daily returns vs 0 risk-free rate — simplified
  but good enough for relative comparison between holdings
- Max drawdown is calculated from the rolling peak, not calendar year
- Sector map is manually maintained, could pull from yfinance info but
  the categories it returns are too broad for semiconductor sub-sectors

TODO: add correlation matrix between holdings
TODO: pull actual portfolio cost basis to calculate unrealized gain/loss
TODO: add comparison against SMH/SOXX as benchmark
TODO: flag if two holdings are highly correlated (redundant exposure)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")


# edit this with your actual holdings — ticker: number of shares
PORTFOLIO = {
    "NVDA":  3,
    "AMD":   8,
    "INTC":  20,
    "SOXX":  4,
    "AVGO":  2,
    "MU":    15,
    "AMAT":  5,
    "SMH":   6,
}

# manually maintained — yfinance sector tags are too broad for this use case
SECTOR_MAP = {
    "NVDA": "AI/GPU",
    "AMD":  "CPU/GPU",
    "INTC": "CPU/Foundry",
    "AVGO": "Networking",
    "QCOM": "Mobile/RF",
    "MU":   "Memory",
    "TXN":  "Analog",
    "MRVL": "Data Infra",
    "MCHP": "Microcontrollers",
    "ON":   "Power/Sensing",
    "SWKS": "RF",
    "MPWR": "Power ICs",
    "WOLF": "SiC/Power",
    "LSCC": "FPGAs",
    "AMAT": "Equipment",
    "LRCX": "Equipment",
    "KLAC": "Equipment",
    "ASML": "Equipment",
    "ONTO": "Equipment",
    "SOXX": "ETF (Broad)",
    "SMH":  "ETF (Broad)",
    "PSI":  "ETF (Broad)",
    "SOXQ": "ETF (Broad)",
    "USD":  "ETF (Leveraged)",
    "SOXS": "ETF (Inverse)",
    "SOXL": "ETF (Leveraged)",
}

ETFS      = {"SOXX", "SMH", "PSI", "SOXQ", "USD", "SOXS", "SOXL"}
LEVERAGED = {"USD", "SOXS", "SOXL"}
INVERSE   = {"SOXS"}

# tickers we recognize as semiconductor-related
KNOWN_SEMIS = {
    "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "MRVL",
    "MCHP", "ON", "SWKS", "MPWR", "WOLF", "LSCC", "FORM",
    "AMAT", "LRCX", "KLAC", "ASML", "ONTO",
    "SOXX", "SMH", "PSI", "SOXQ", "USD", "SOXS", "SOXL"
}


def fetch_portfolio_data(portfolio):
    print("Fetching market data...\n")
    data = {}
    prices = {}

    for ticker in portfolio:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            if len(hist) == 0:
                print(f"  {ticker} — no data returned, skipping")
                continue
            data[ticker] = {"hist": hist, "info": t.info}
            prices[ticker] = hist["Close"].iloc[-1]
            print(f"  {ticker:<6} ${prices[ticker]:.2f}")
        except Exception as e:
            print(f"  {ticker} — fetch error: {e}")

    return data, prices


def calculate_metrics(portfolio, data, prices):
    metrics = {}
    total_value = 0

    for ticker, shares in portfolio.items():
        if ticker not in data:
            continue

        price = prices[ticker]
        value = price * shares
        hist  = data[ticker]["hist"]["Close"]

        ret_1y = (hist.iloc[-1] / hist.iloc[0] - 1) * 100

        daily = hist.pct_change().dropna()
        vol   = daily.std() * np.sqrt(252) * 100  # annualized

        # simplified sharpe — no risk-free rate subtracted
        sharpe = (daily.mean() * 252) / (daily.std() * np.sqrt(252)) if daily.std() > 0 else 0

        # max drawdown from rolling peak
        drawdown = ((hist / hist.cummax()) - 1).min() * 100

        metrics[ticker] = {
            "shares":       shares,
            "price":        price,
            "value":        value,
            "return_1y":    ret_1y,
            "volatility":   vol,
            "sharpe":       sharpe,
            "max_drawdown": drawdown,
            "sector":       SECTOR_MAP.get(ticker, "Other"),
            "is_etf":       ticker in ETFS,
            "is_leveraged": ticker in LEVERAGED,
            "is_inverse":   ticker in INVERSE,
        }
        total_value += value

    for ticker in metrics:
        metrics[ticker]["weight"] = (metrics[ticker]["value"] / total_value) * 100

    return metrics, total_value


def generate_feedback(metrics, total_value):
    positives = []
    notes     = []
    warnings  = []

    tickers    = list(metrics.keys())
    weights    = {t: metrics[t]["weight"] for t in tickers}
    etf_weight = sum(weights[t] for t in tickers if metrics[t]["is_etf"])
    stock_weight = 100 - etf_weight

    # concentration check — anything over 40% is a red flag
    for ticker, w in weights.items():
        if w > 40:
            warnings.append(
                f"HIGH CONCENTRATION: {ticker} makes up {w:.1f}% of your portfolio. "
                f"Consider trimming if this is unintentional."
            )
        elif w > 25:
            warnings.append(
                f"MODERATE CONCENTRATION: {ticker} is {w:.1f}% of your portfolio."
            )

    # leveraged/inverse products are not buy-and-hold instruments
    for ticker in tickers:
        if metrics[ticker]["is_leveraged"]:
            warnings.append(
                f"LEVERAGED ETF: {ticker} is a leveraged or inverse product. "
                f"These decay over time and are designed for short-term trading, not long-term holding."
            )

    # sector concentration
    sector_weights = {}
    for ticker in tickers:
        s = metrics[ticker]["sector"]
        sector_weights[s] = sector_weights.get(s, 0) + weights[ticker]

    dominant = max(sector_weights, key=sector_weights.get)
    if sector_weights[dominant] > 50:
        warnings.append(
            f"SECTOR CONCENTRATION: {sector_weights[dominant]:.1f}% of your portfolio "
            f"is in {dominant}. Consider spreading across more semiconductor sub-sectors."
        )

    # etf vs stock balance
    if etf_weight > 70:
        notes.append(
            f"Portfolio is {etf_weight:.1f}% ETFs. Broad exposure with lower risk "
            f"but limits upside from individual stock picks."
        )
    elif stock_weight > 90:
        warnings.append(
            f"Portfolio is almost entirely individual stocks ({stock_weight:.1f}%). "
            f"Adding a broad ETF like SOXX or SMH could reduce single-stock risk."
        )
    else:
        positives.append(
            f"Good balance: {stock_weight:.1f}% individual stocks and {etf_weight:.1f}% ETFs."
        )

    # performance
    by_return = sorted(tickers, key=lambda t: metrics[t]["return_1y"], reverse=True)
    best  = by_return[0]
    worst = by_return[-1]

    positives.append(
        f"Best performer: {best} at {metrics[best]['return_1y']:+.1f}% over the past year."
    )

    if metrics[worst]["return_1y"] < -20:
        warnings.append(
            f"Underperformer: {worst} is down {metrics[worst]['return_1y']:.1f}% over the past year. "
            f"Review your thesis for holding this position."
        )
    else:
        notes.append(
            f"Weakest performer: {worst} at {metrics[worst]['return_1y']:+.1f}% over the past year."
        )

    # volatility
    avg_vol = np.mean([metrics[t]["volatility"] for t in tickers])
    if avg_vol > 60:
        warnings.append(
            f"HIGH VOLATILITY: Average annualized volatility is {avg_vol:.1f}%. "
            f"Semiconductors are cyclical — be prepared for large swings."
        )
    else:
        positives.append(
            f"Portfolio volatility is {avg_vol:.1f}% annualized — reasonable for the sector."
        )

    # number of holdings
    n = len(tickers)
    if n < 3:
        warnings.append(
            f"Only {n} holding(s). Consider adding more positions to reduce single-stock risk."
        )
    elif n > 15:
        notes.append(
            f"{n} positions is a lot for a sector-focused portfolio. "
            f"Make sure each holding has a clear thesis."
        )
    else:
        positives.append(f"{n} holdings — focused but reasonably diversified for a sector portfolio.")

    # flag unrecognized tickers
    unknown = [t for t in tickers if t not in KNOWN_SEMIS]
    if unknown:
        warnings.append(
            f"Unrecognized tickers: {', '.join(unknown)}. "
            f"Verify these are semiconductor-related holdings."
        )

    return positives, notes, warnings


def print_report(metrics, total_value, positives, notes, warnings):
    print("\n" + "=" * 62)
    print("  SEMICONDUCTOR PORTFOLIO ASSESSMENT")
    print("=" * 62)
    print(f"\n  Total Value:    ${total_value:,.2f}")
    print(f"  Holdings:       {len(metrics)}\n")

    print(f"  {'Ticker':<6} {'Shares':>6} {'Price':>8} {'Value':>10} "
          f"{'Weight':>7} {'1Y Ret':>8} {'Vol':>8} {'Sharpe':>7}")
    print("  " + "-" * 60)

    for ticker, m in sorted(metrics.items(), key=lambda x: x[1]["value"], reverse=True):
        print(f"  {ticker:<6} {m['shares']:>6} ${m['price']:>7.2f} "
              f"${m['value']:>9,.2f} {m['weight']:>6.1f}% "
              f"{m['return_1y']:>+7.1f}% {m['volatility']:>6.1f}%  {m['sharpe']:>6.2f}")

    print("\n  POSITIVES")
    print("  " + "-" * 40)
    for p in positives:
        print(f"  + {p}")

    if notes:
        print("\n  NOTES")
        print("  " + "-" * 40)
        for n in notes:
            print(f"  > {n}")

    if warnings:
        print("\n  WARNINGS")
        print("  " + "-" * 40)
        for w in warnings:
            print(f"  ! {w}")

    print("\n" + "=" * 62)
    print("  For educational purposes only. Not financial advice.")
    print("=" * 62 + "\n")


def plot_portfolio(metrics, data):
    fig = plt.figure(figsize=(16, 18))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    BG    = "#161b22"
    SPINE = "#30363d"
    TEXT  = "#e6edf3"
    MUTED = "#8b949e"
    PALETTE = [
        "#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
        "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb86c",
        "#39d353", "#f0883e"
    ]

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)
        ax.title.set_color(TEXT)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    tickers = list(metrics.keys())
    values  = [metrics[t]["value"]      for t in tickers]
    weights = [metrics[t]["weight"]     for t in tickers]
    returns = [metrics[t]["return_1y"]  for t in tickers]
    vols    = [metrics[t]["volatility"] for t in tickers]
    sharpes = [metrics[t]["sharpe"]     for t in tickers]
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(tickers))]

    # allocation pie
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    wedges, texts, autotexts = ax1.pie(
        values, labels=tickers, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"color": TEXT, "fontsize": 8},
        wedgeprops={"linewidth": 1.5, "edgecolor": "#0d1117"}
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color(MUTED)
    ax1.set_title("Portfolio Allocation", fontsize=12, fontweight="bold", pad=15)

    # 1y returns bar
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    bars = ax2.bar(tickers, returns, color=colors, alpha=0.85, edgecolor=SPINE)
    ax2.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (1 if val >= 0 else -3),
                 f"{val:+.0f}%", ha="center", va="bottom", color=TEXT, fontsize=7)
    ax2.set_title("1-Year Returns", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Return (%)")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # risk vs return scatter — bubble size = portfolio weight
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    for i, ticker in enumerate(tickers):
        ax3.scatter(vols[i], returns[i], color=colors[i],
                    s=weights[i] * 8, alpha=0.85,
                    edgecolors=SPINE, linewidth=0.8, zorder=3)
        ax3.annotate(ticker, (vols[i], returns[i]),
                     textcoords="offset points", xytext=(6, 4),
                     color=TEXT, fontsize=8)
    ax3.axhline(0, color=MUTED, linewidth=0.6, linestyle="--")
    ax3.set_title("Risk vs Return  (bubble = portfolio weight)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Annualized Volatility (%)")
    ax3.set_ylabel("1Y Return (%)")

    # sharpe ratio horizontal bar
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    idx = np.argsort(sharpes)[::-1]
    s_tickers = [tickers[i] for i in idx]
    s_sharpes = [sharpes[i] for i in idx]
    s_colors  = [colors[i]  for i in idx]
    bars = ax4.barh(s_tickers, s_sharpes, color=s_colors, alpha=0.85, edgecolor=SPINE)
    ax4.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax4.axvline(1, color="#3fb950", linewidth=0.8, linestyle=":", alpha=0.6)
    for bar, val in zip(bars, s_sharpes):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", color=TEXT, fontsize=8)
    ax4.set_title("Sharpe Ratio  (green line = 1.0)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("Sharpe Ratio")

    # normalized price performance
    ax5 = fig.add_subplot(gs[2, :])
    style_ax(ax5)
    for i, ticker in enumerate(tickers):
        try:
            hist = data[ticker]["hist"]["Close"]
            norm = hist / hist.iloc[0] * 100
            ax5.plot(norm.index, norm.values, color=colors[i],
                     linewidth=1.2, label=ticker, alpha=0.85)
        except Exception:
            pass
    ax5.axhline(100, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax5.set_title("Normalized Price Performance — Past 1 Year (base = 100)",
                  fontsize=12, fontweight="bold")
    ax5.set_ylabel("Normalized Price")
    ax5.legend(facecolor=BG, edgecolor=SPINE, labelcolor=TEXT,
               fontsize=8, ncol=5, loc="upper left")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax5.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("SEMICONDUCTOR PORTFOLIO ASSESSMENT",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.98)

    plt.savefig("semiconductor_portfolio_report.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print("Chart saved to semiconductor_portfolio_report.png")
    plt.show()


if __name__ == "__main__":
    data, prices = fetch_portfolio_data(PORTFOLIO)

    if not data:
        print("No data fetched. Check your tickers and internet connection.")
        exit()

    metrics, total_value = calculate_metrics(PORTFOLIO, data, prices)
    positives, notes, warnings = generate_feedback(metrics, total_value)
    print_report(metrics, total_value, positives, notes, warnings)
    plot_portfolio(metrics, data)
