import argparse
import time
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from timesnet import TimesNet

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "train.nc"
DEFAULT_RESULTS_SUBDIR = Path("results") / "weather"
DEFAULT_OUTPUT_NAME = "weather_forecast.csv"

# Model
DEFAULT_LOOKBACK = 24 * 5
DEFAULT_LOOKAHEAD = 48
DEFAULT_BATCH = 256 * 8
DEFAULT_LR = 5e-6
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 10
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- HELPER METHODS ----------
def train_val_test_split(data, lookback):
    T = len(data.timestamp)

    # Use last LOOKBACK hours to predict future weather data
    test_size = lookback
    val_size  = 20 * 24

    total_holdout = val_size + test_size

    train_data = data.isel(timestamp=slice(0, T - total_holdout))
    val_data = data.isel(timestamp=slice(T - total_holdout, T - test_size))
    test_data = data.isel(timestamp=slice(T - test_size, T))
        
    return train_data, val_data, test_data


def create_model(lookback, lookahead, feat_size):
    enc_in = dec_in = c_out = feat_size
    pred_len = lookahead
    seq_len = lookback

    return TimesNet(
        enc_in = enc_in,
        dec_in = dec_in,
        c_out = c_out, 
        pred_len = pred_len,
        seq_len = seq_len,
        data_idx = list(range(feat_size)),
        time_idx = [],
        # everythin below is tunable
        d_model=64,
        d_ff=256,
        e_layers=3,
        top_k=5,
        num_kernels=4,
        dropout=0.1,
    )
    

def build_windows(weather_data, lookback, lookahead):
    T, L, _ = weather_data.shape
    
    N = T - lookback - lookahead + 1
    
    inputs, targets = [], []
    for loc in range(L):
        for t in range(N):
            x = weather_data[t:t + lookback, loc, :]                        # (lookback, F)
            y = weather_data[t + lookback:t + lookback + lookahead, loc, :] # (lookahead, F)
            inputs.append(x)
            targets.append(y)
            
    X = np.stack(inputs)  # (N*L, lookback, F)
    Y = np.stack(targets) # (N*L, lookahead, F)
    return X, Y


def prepare_training(train_data, val_data, lookback, lookahead):
    w = train_data.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    T, L, F = w.shape
    
    w_val = val_data.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T_val, L, F)
    T_val = w_val.shape[0]
    
    w_scaler = StandardScaler().fit(w.reshape(-1, F))
    w_sc = w_scaler.transform(w.reshape(-1, F)).reshape(T, L, F)
    w_val_sc = w_scaler.transform(w_val.reshape(-1, F)).reshape(T_val, L, F)
    
    X, Y = build_windows(w_sc, lookback=lookback, lookahead=lookahead)
    X_val, Y_val = build_windows(w_val_sc, lookback=lookback, lookahead=lookahead)
    
    return (X, Y), (X_val, Y_val), w_scaler


def build_dl(X, Y, batch_size):
    ds = TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return dl

# ---------- MODEL TRAINING AND INFERENCE ----------
def train_model(
    train,
    val,
    lookback,
    lookahead,
    batch_size,
    feat_size,
    device,
    *,
    test_data=None,
    scaler=None,
    results_dir=None,
    lr=DEFAULT_LR,
    epochs=DEFAULT_EPOCHS,
    patience=DEFAULT_PATIENCE,
    save_intermediate=True,
):
    X, Y = train
    X_val, Y_val = val
    
    dl_train = build_dl(X, Y, batch_size=batch_size)
    dl_val = build_dl(X_val, Y_val, batch_size=batch_size)
    
    model = create_model(
        lookback=lookback, lookahead=lookahead, feat_size=feat_size
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loss_dict = {}
    val_loss_dict = {}
    
    best_val_loss = float("inf")
    best_model_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        # ----- training -----
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(dl_train.dataset)
        
        # ----- validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
                
        val_loss /= len(dl_val.dataset)

        train_loss_dict[epoch+1] = train_loss
        val_loss_dict[epoch+1] = val_loss
        
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.2f}s")
        
        # ----- run inference -----
        if save_intermediate and results_dir is not None and test_data is not None and scaler is not None:
            df_pred = predict(test_data, model, scaler, device=device)
            out = results_dir / f"weather_pred_epoch{epoch+1}.csv"
            df_pred.to_csv(out, index=False)
            # print(f"Saved inference predictions for epoch {epoch+1}: {out}")
        
        # ----- early stopping -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if results_dir is not None:
        torch.save(model.state_dict(), results_dir / "best_checkpoint.ckpt")
        
    # ----- save losses -----
    df_losses = pd.DataFrame({
        "epoch": list(train_loss_dict.keys()),
        "train_loss": list(train_loss_dict.values()),
        "val_loss": list(val_loss_dict.values())
    })
    if results_dir is not None:
        out_losses = results_dir / "losses.csv"
        df_losses.to_csv(out_losses, index=False)
        
    return model


def predict(test_data, model, scaler, device):
    locs = list(map(str, test_data.location.values))
    features = list(map(str, test_data.feature.values))
    ts_future = pd.date_range(
        start="2023-06-30T01:00:00.000000000",
        end="2023-07-02T00:00:00.000000000",
        freq="1h",
    )
    
    w = test_data.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    T, L, F = w.shape
    
    w_sc = scaler.transform(w.reshape(-1, F)).reshape(T, L, F)
    
    preds_all = []
    
    model.eval()
    with torch.no_grad():
        for li, county in enumerate(locs):
            w_loc = w_sc[:, li, :] # (lookback, F)
            X_loc = torch.from_numpy(w_loc).unsqueeze(0).float().to(device) # (1, lookback, F)
            
            pred_sc = model(X_loc) # (1, lookahead, F)
            pred_sc = pred_sc.squeeze(0).cpu().numpy() # (lookahead, F)
            
            pred = scaler.inverse_transform(pred_sc)
            preds_all.append(pred)
            
    preds_all = np.stack(preds_all) # (L, lookahead, F)
    
    df_list = []
    for li, county in enumerate(locs):
        df_loc = pd.DataFrame(
            preds_all[li],
            index=ts_future,
            columns=features
        ).reset_index().rename(columns={"index": "timestamp"})
        
        df_loc.insert(1, "county", county)
        
        df_list.append(df_loc)
    
    df = pd.concat(df_list, ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train the TimesNet weather forecasting model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to the train.nc NetCDF file.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_SUBDIR, help="Directory (relative to script) where artifacts will be stored.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_NAME), help="Forecast CSV name or path. Relative paths are resolved inside results dir.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="History window length in hours.")
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD, help="Forecast horizon in hours.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE, help="Device to use for training.")
    parser.add_argument("--skip-intermediate", action="store_true", help="Disable saving intermediate forecasts after each epoch.")
    args = parser.parse_args()

    data_path = args.data_path
    if not data_path.is_absolute():
        data_path = (BASE_DIR / data_path).resolve()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = (BASE_DIR / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = results_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    data = xr.open_dataset(data_path)
    
    lookback = int(args.lookback)
    lookahead = int(args.lookahead)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    patience = max(int(args.patience), 1)
    lr = float(args.lr)
    save_intermediate = not args.skip_intermediate

    train_data, val_data, test_data = train_val_test_split(data, lookback=lookback)
    
    feat_size = int(data["weather"].shape[-1])
    train, val, scaler = prepare_training(
        train_data, val_data, lookback=lookback, lookahead=lookahead
    )
    model = train_model(
        train,
        val,
        lookback=lookback,
        lookahead=lookahead,
        batch_size=batch_size,
        feat_size=feat_size,
        device=device,
        test_data=test_data,
        scaler=scaler,
        results_dir=results_dir,
        lr=lr,
        epochs=epochs,
        patience=patience,
        save_intermediate=save_intermediate,
    )
    df = predict(test_data, model, scaler, device=device)
    df.to_csv(output_path, index=False)
    
    print("Saved predictions:")
    print(f"  {output_path}")
    
    data.close(); train_data.close(); val_data.close(); test_data.close()
    

if __name__ == "__main__":
    main()
