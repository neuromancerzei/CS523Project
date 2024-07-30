# TWS Trading Simulation

## Project Overview
This project develops a trading simulation system integrated with Interactive Brokers Trader Workstation (TWS). It leverages sentiment analysis derived from financial news to inform trading decisions, providing a robust environment to simulate and analyze potential trading outcomes without financial risk.

## Introduction
This project aims to harness the power of machine learning to analyze sentiments from financial news and apply this insight to develop and test trading strategies in a simulated environment using TWS.

## Data Collection
Data collection involves:
- **Historical Market Data**: Collecting price and volume data.
- **News Headlines**: Fetching financial news headlines for sentiment analysis.

## Feature Extraction and Preprocessing
Processes include:
- **Text Preprocessing**: HTML tag removal, tokenization, and stopword removal.
- **Feature Extraction**: Using Word2Vec for generating dense vector representations of words.

## Model Development
A convolutional neural network (CNN) model processes the textual data, structured to capture the semantic relationships essential for accurate sentiment analysis.

## Strategy Simulation
We simulate trading strategies based on model outputs to evaluate potential trading performance:
- **Backtesting**: Using historical data to test strategy effectiveness.
- **Risk Management**: Implementing protocols to mitigate financial risk during simulations.

## Integration with TWS
Using the `ib_insync` library, the system connects to TWS for executing trades based on the simulated strategies in a controlled environment.

## Installation
To set up the project environment:
```bash
git clone https://github.com/your-repository/tws-trading-simulation
cd tws-trading-simulation
pip install -r requirements.txt
