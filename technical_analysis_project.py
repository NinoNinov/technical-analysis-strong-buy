import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from mplfinance.original_flavor import candlestick_ohlc
from sqlalchemy import create_engine

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


def create_engine_from_config():
    """
    Create a SQLAlchemy engine using credentials from config.py.
    """
    connection_string = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def fetch_strong_buy_stocks(engine, min_market_cap: float, rec_key: str):
    """
    Fetch stocks from the Stocks table filtered by Rec_Key and Market_cap.
    """
    # Basic sanitization to avoid breaking the SQL if user types a quote
    rec_key_safe = rec_key.replace("'", "''")

    query_filtered = f"""
    SELECT 
        symbol,
        Target_LP,
        Target_Mean_P,
        Sector,
        Anlsts,
        Rec_Mean,
        Market_Cap,
        `MTD Change`,
        `YTD Change`
    FROM Stocks 
    WHERE Rec_Key = 'strong_buy'
      AND Country = 'United States'
      AND Market_cap > {min_market_cap}
    
    """

    df = pd.read_sql(query_filtered, engine)
    return df


def generate_technical_analysis_pdf(df: pd.DataFrame, output_path: str, rec_key: str):
    """
    Generate a multi-page PDF with technical analysis charts
    for each symbol in df (filtered by the chosen Rec_Key).
    """
    if df is None or df.empty:
        print("No data returned for Rec_Key = 'strong_buy'. Nothing to plot.")
        return

    list_of_stocks = df["symbol"].tolist()

    warnings.filterwarnings("ignore")  # match notebook behaviour

    with PdfPages(output_path) as pdf:
        for ticker in list_of_stocks:
            try:
                # Get the value metrics for this stock
                stock_row = df[df["symbol"] == ticker].iloc[0]
                target_lp = stock_row["Target_LP"]
                target_mean_p = stock_row["Target_Mean_P"]
                sector = stock_row["Sector"]
                anlsts = stock_row["Anlsts"]
                rec_mean = stock_row["Rec_Mean"]
                market_cap = stock_row["Market_Cap"]
                # Convert MTD and YTD changes to float, handling None or string values
                try:
                    mtd_change = float(stock_row["MTD Change"]) if pd.notna(stock_row["MTD Change"]) else 0.0
                except (ValueError, TypeError):
                    mtd_change = 0.0
                try:
                    ytd_change = float(stock_row["YTD Change"]) if pd.notna(stock_row["YTD Change"]) else 0.0
                except (ValueError, TypeError):
                    ytd_change = 0.0

                # Fetch stock data from Yahoo Finance
                stock_data = yf.download(
                    ticker, start="2024-01-01", progress=False, auto_adjust=False
                )

                if stock_data.empty:
                    print(f"Skipping {ticker}: no price data from Yahoo Finance.")
                    continue

                # Calculate moving averages
                stock_data["200_MA"] = stock_data["Close"].rolling(
                    window=200, min_periods=1
                ).mean()
                stock_data["50_MA"] = stock_data["Close"].rolling(
                    window=50, min_periods=1
                ).mean()

                # Calculate RSI (Relative Strength Index)
                delta = stock_data["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                stock_data["RSI"] = 100 - (100 / (1 + rs))

                # Convert index to matplotlib date format
                stock_data_reset = stock_data.reset_index()
                stock_data_reset["Date"] = stock_data_reset["Date"].apply(
                    mdates.date2num
                )

                # Create figure with subplots: price chart and RSI
                fig = plt.figure(figsize=(24, 12))
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

                # Main price chart with candlesticks and moving averages
                ax1 = fig.add_subplot(gs[0])

                # Plot candlestick chart
                candlestick_ohlc(
                    ax1,
                    stock_data_reset[["Date", "Open", "High", "Low", "Close"]].values,
                    width=0.6,
                    colorup="g",
                    colordown="r",
                    alpha=0.8,
                )

                # Plot moving averages
                ax1.plot(
                    stock_data_reset["Date"],
                    stock_data_reset["200_MA"],
                    color="red",
                    label="200-Day MA",
                    linewidth=2,
                    alpha=0.7,
                )
                ax1.plot(
                    stock_data_reset["Date"],
                    stock_data_reset["50_MA"],
                    color="blue",
                    label="50-Day MA",
                    linewidth=2,
                    alpha=0.7,
                )

                # Add PCT metrics as horizontal reference lines
                ax1.axhline(
                    y=target_lp,
                    color="#FF6B6B",
                    linestyle="--",
                    label=f"Target_LP: {target_lp:.1f}%",
                    linewidth=2,
                    alpha=0.7,
                )
                ax1.axhline(
                    y=target_mean_p,
                    color="#4ECDC4",
                    linestyle="--",
                    label=f"Target_Mean_P: {target_mean_p:.1f}%",
                    linewidth=2,
                    alpha=0.7,
                )

                # Updated title to include sector, Rec_Key, Anlsts, Rec_Mean, Market_Cap, MTD Change, and YTD Change
                ax1.set_title(
                    f"{ticker} - {sector} - ({rec_key}) - Anlsts: {anlsts}, Rec_Mean: {rec_mean}, Market_Cap: {market_cap}, MTD: {mtd_change:.2f}%, YTD: {ytd_change:.2f}%",
                    fontsize=18,
                    fontweight="bold",
                    pad=20,
                )
                ax1.set_ylabel("Price ($)", fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc="upper left", fontsize=10)

                # Format x-axis (show diagonal dates on price chart)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax1.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

                # RSI subplot
                ax3 = fig.add_subplot(gs[1], sharex=ax1)
                ax3.plot(
                    stock_data_reset["Date"],
                    stock_data_reset["RSI"],
                    color="purple",
                    linewidth=1.5,
                )
                ax3.axhline(
                    y=70,
                    color="r",
                    linestyle="--",
                    alpha=0.5,
                    label="Overbought (70)",
                )
                ax3.axhline(
                    y=30,
                    color="g",
                    linestyle="--",
                    alpha=0.5,
                    label="Oversold (30)",
                )
                ax3.fill_between(
                    stock_data_reset["Date"],
                    30,
                    70,
                    alpha=0.1,
                    color="gray",
                )
                ax3.set_ylabel("RSI", fontsize=12)
                ax3.set_ylim(0, 100)
                ax3.set_xlabel("Date", fontsize=12)
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc="upper right", fontsize=9)

                # Format bottom x-axis
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax3.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

                # Get current price info safely as scalars
                close_series = stock_data["Close"]
                ma_50_series = stock_data["50_MA"]
                ma_200_series = stock_data["200_MA"]
                rsi_series = stock_data["RSI"]

                current_price = float(close_series.iloc[-1])
                price_change = float(close_series.iloc[-1] - close_series.iloc[-2])
                price_change_pct = float(
                    (price_change / close_series.iloc[-2]) * 100
                )
                ma_50 = float(ma_50_series.iloc[-1])
                ma_200 = float(ma_200_series.iloc[-1])
                rsi_value = float(rsi_series.iloc[-1])

                # Add text box with current stats
                stats_text = (
                    f"Current: ${current_price:.2f} ({price_change_pct:+.2f}%)\n"
                )
                stats_text += f"50 MA: ${ma_50:.2f}\n"
                stats_text += f"200 MA: ${ma_200:.2f}\n"
                stats_text += f"RSI: {rsi_value:.1f}"
                ax1.text(
                    0.01,
                    0.78,
                    stats_text,
                    transform=ax1.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

                plt.tight_layout()

                # Save current figure as a new page in the PDF and close it
                pdf.savefig(fig)
                plt.close(fig)

                print(f"Added {ticker} page to {output_path}")

            except PermissionError as e:
                print(f"Error processing {ticker}: Permission denied. Please close the PDF file '{output_path}' if it's open in another program.")
                print(f"Full error: {e}")
                continue
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

    print(f"PDF report created: {output_path}")


def main(output_pdf: str = None):
    # Ask the user which Rec_Key to use (default: 'strong_buy')
    rec_key_input = input(
        "Enter Rec_Key filter (press Enter for default 'strong_buy'): "
    ).strip()
    rec_key = rec_key_input if rec_key_input else "strong_buy"

    # Generate default filename with current date if not provided
    if output_pdf is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_pdf = f"strong_buy_{rec_key}_{date_str}.pdf"

    # Ask the user for the minimum market cap instead of using a fixed value (20)
    try:
        user_input = input(
            "Enter minimum Market_cap (e.g. 20 for 20 billion): "
        ).strip()
        min_market_cap = float(user_input) if user_input else 20.0
    except ValueError:
        print("Invalid input. Falling back to default Market_cap > 20.")
        min_market_cap = 20.0

    engine = create_engine_from_config()
    df = fetch_strong_buy_stocks(engine, min_market_cap, rec_key)
    generate_technical_analysis_pdf(df, output_pdf, rec_key)


if __name__ == "__main__":
    main()

