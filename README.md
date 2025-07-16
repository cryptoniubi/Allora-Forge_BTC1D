# BTC/USD 1-Day Log-Return Prediction

Predicts daily BTC/USD log-returns using historical price data from CoinGecko, technical indicators (MACD, RSI, Bollinger Bands), and news sentiment from CryptoPanic. Built with LightGBM.

## How to run

1. Install requirements:
   ```
   pip install pandas numpy requests ta lightgbm scikit-learn joblib
   ```
2. Get a free CryptoPanic API key: https://cryptopanic.com/developers/api/
3. Add your API key to the script in the `CRYPTOPANIC_API_KEY` variable.
4. Run:
   ```
   python btc_logreturn_predictor.py
   ```

Results and the trained model are saved in the `outputs/` directory.

## Outputs

- `outputs/predictions.csv` — True & predicted log-returns for the test set
- `outputs/lgbm_model.pkl` — Trained LightGBM model

## Credits

Script inspired by arcxteam/train-models and customized for the Allora Network BTC/USD prediction challenge.