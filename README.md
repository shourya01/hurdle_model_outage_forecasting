# INFORMS 2025 Data Mining Challenge Submission

This repository contains our solution to the INFORMS 2025 Data Mining Challenge. It scores a RMSE of 189 on the competition leaderboard.

### Authors
- Shourya Bose, Tomas Kaljevic, Yu Zhang  
  Department of Electrical and Computer Engineering, University of California, Santa Cruz

### Quick Start
1. Ensure `train.nc` is located in this directory.
2. Generate weather forecasts:
   ```bash
   python train_weather.py
   ```
3. Train the outage model using the generated forecasts:
   ```bash
   python train_outage.py results/weather/weather_forecast.csv
   ```

For a detailed description of the competition dataset (`\train.nc`) schema, see `TRAIN_NC.md`.
