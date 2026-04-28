# Semiconductor Portfolio Assesser

A Python tool that analyzes a portfolio of semiconductor stocks and ETFs and generates a detailed assessment report with feedback, warnings, and charts.

## Overview

This project fetches real-time data from Yahoo Finance for any semiconductor holdings you input, calculates key performance metrics, and gives written feedback on the health and composition of your portfolio. It flags risks like over-concentration, leveraged ETFs, and poor diversification while highlighting your best performers.

## What It Analyzes

- Total portfolio value and weight of each holding
- 1-year return per stock and ETF
- Annualized volatility and Sharpe ratio
- Maximum drawdown
- Sector diversification across sub-sectors like AI/GPU, Memory, Equipment, and Networking
- ETF vs individual stock balance
- Concentration warnings
- Leveraged and inverse ETF warnings

## Supported Holdings

Supports all major semiconductor stocks including NVDA, AMD, INTC, AVGO, QCOM, MU, AMAT, LRCX, KLAC, ASML, TXN, MRVL and more, as well as ETFs including SOXX, SMH, PSI, and SOXQ.

## Setup

Install dependencies:

pip install yfinance pandas numpy matplotlib

## How To Use

Open semiconductor_portfolio.py and edit the portfolio section at the top of the file with your actual holdings:

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

Then run:

python3 semiconductor_portfolio.py

## Viewing The Text Report

When you run the script a full written report prints in the terminal before the charts load. If you only see graphs scroll all the way up in the terminal to find the report. It will show three sections:

POSITIVES  — what you are doing well
NOTES      — neutral observations
WARNINGS   — things to consider changing

To save the terminal report to a text file run this instead:

python3 semiconductor_portfolio.py 2>&1 | tee evaluation.txt

This saves the report as evaluation.txt in your project folder.

## Output

- Full assessment report printed in terminal
- Chart saved as semiconductor_portfolio_report.png containing:
  - Portfolio allocation pie chart
  - 1-year returns bar chart
  - Risk vs return scatter plot
  - Sharpe ratio ranking
  - Normalized 1-year price performance for all holdings
- Text report saved as evaluation.txt

## Tech Stack

- Python
- yfinance
- pandas
- numpy
- matplotlib

## Notes

This tool is for educational purposes only and is not financial advice. Always do your own research before making investment decisions.
