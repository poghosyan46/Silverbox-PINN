import os
import urllib.request
import zipfile
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. ՏՎՅԱԼՆԵՐԻ ՆԵՐԲԵՌՆՈՒՄ ԵՎ ՆԱԽԱՄՇԱԿՈՒՄ
# ==========================================
def load_and_preprocess_data():
    mat_filename = "SNLS80mV.mat"
    zip_filename = "SilverboxFiles.zip"
    
    # Եթե ֆայլը չկա, ավտոմատ ներբեռնում ենք
    if not os.path.exists(mat_filename):
        print("Ներբեռնվում է Silverbox dataset-ը...")
        url = "https://www.nonlinearbenchmark.org/files/benchmarks/SilverboxFiles.zip"
        urllib.request.urlretrieve(url, zip_filename)
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    mat_data = sio.loadmat(mat_filename)
    u_full = mat_data['V1'].squeeze()
    y_full = mat_data['V2'].squeeze()
    
    # [42650 : 46746] հատվածի կտրում (4096 sample)
    start_idx = 42650
    end_idx = 42650 + 4096
    u_excerpt = u_full[start_idx:end_idx]
    y_excerpt = y_full[start_idx:end_idx]
    
    # Կենտրոնացում (subtract mean)
    u_centered = u_excerpt - np.mean(u_excerpt)
    y_centered = y_excerpt - np.mean(y_excerpt)
    
    return u_centered, y_centered

# ==========================================
# 2. MODEL ԵՎ RK4 INTEGRATOR
# ==========================================
class AccelerationNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super(AccelerationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, y, v, u):
        inputs = torch.cat([y, v, u], dim=-1)
        return self.net(inputs)

def rk4_step(model, y_t, v_t, u_t, u_next, dt):
    u_mid = 0.5 * (u_t + u_next)

    k1 = v_t
    m1 = model(y_t, v_t, u_t)

    k2 = v_t + 0.5 * dt * m1
    m2 = model(y_t + 0.5 * dt * k1, v_t + 0.5 * dt * m1, u_mid)

    k3 = v_t + 0.5 * dt * m2
    m3 = model(y_t + 0.5 * dt * k2, v_t + 0.5 * dt * m2, u_mid)

    k4 = v_t + dt * m3
    m4 = model(y_t + dt * k3, v_t + dt * m3, u_next)

    y_next = y_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    v_next = v_t + (dt / 6.0) * (m1 + 2*m2 + 2*m3 + m4)

    return y_next, v_next

# ==========================================
# 3. MAIN RUN & TRAINING
# ==========================================
def main():
    u, y = load_and_preprocess_data()
    
    fs = 610.35
    dt = 1.0 / fs
    
    # Տվյալների վերածում PyTorch Tensor-ների
    u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Սկզբնական արագություն v0 (finite-difference-ով)
    v0 = (y[1] - y[0]) / dt
    
    model = AccelerationNet(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Սկսվում է PINN մոդելի մարզումը...")
    
    # Training loop (Rollout-ով 3072 train նմուշների վրա)
    epochs = 500
    rollout_len = 64  # sequence length for training
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Պատահական սկզբնակետ ընտրում ենք train record-ի միջից
        idx = np.random.randint(0, 3072 - rollout_len)
        
        # Սկզբնական վիճակ
        y_curr = y_tensor[idx:idx+1]
        v_curr = torch.tensor([[(y[idx+1] - y[idx]) / dt]], dtype=torch.float32)
        
        y_pred_list = []
        for t in range(rollout_len):
            y_pred_list.append(y_curr)
            u_t = u_tensor[idx + t : idx + t + 1]
            u_next = u_tensor[idx + t + 1 : idx + t + 2]
            y_curr, v_curr = rk4_step(model, y_curr, v_curr, u_t, u_next, dt)
            
        y_pred_seq = torch.cat(y_pred_list, dim=0)
        y_true_seq = y_tensor[idx : idx + rollout_len]
        
        loss = nn.MSELoss()(y_pred_seq, y_true_seq)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Rollout Loss: {loss.item():.6f}")

    # ==========================================
    # 4. FREE-RUN EVALUATION (Ազատ ընթացք)
    # ==========================================
    print("\nԿատարվում է Free-Run Evaluation բոլոր 4096 կետերի վրա...")
    
    model.eval()
    y_preds = []
    
    y_curr = y_tensor[0:1]
    v_curr = torch.tensor([[v0]], dtype=torch.float32)
    
    with torch.no_grad():
        for t in range(len(u) - 1):
            y_preds.append(y_curr.item())
            u_t = u_tensor[t:t+1]
            u_next = u_tensor[t+1:t+2]
            y_curr, v_curr = rk4_step(model, y_curr, v_curr, u_t, u_next, dt)
        y_preds.append(y_curr.item())
    
    y_preds = np.array(y_preds)
    
    # Train RMSE (առաջին 3072 նմուշները)
    train_rmse = np.sqrt(np.mean((y_preds[:3072] - y[:3072])**2))
    
    # Test RMSE (վերջին 1024 նմուշները)
    test_rmse = np.sqrt(np.mean((y_preds[3072:] - y[3072:])**2))
    
    print("------------------------------------------")
    print(f"Free-run Train RMSE: {train_rmse:.6f} V")
    print(f"Free-run Test RMSE:  {test_rmse:.6f} V")
    print("------------------------------------------")
    
    # ==========================================
    # 5. ԳՐԱՖԻԿԻ ՊԱՀՊԱՆՈՒՄ PDF-Ի ՀԱՄԱՐ
    # ==========================================
    time_test = np.arange(1024) * dt
    plt.figure(figsize=(10, 4))
    plt.plot(time_test, y[3072:], label='Measured $y_{test}$', color='black', alpha=0.8)
    plt.plot(time_test, y_preds[3072:], label=r'Predicted $\hat{y}_{test}$ (PINN)', color='red', linestyle='--')
    plt.title(f'Silverbox Test Free-Run (Test RMSE = {test_rmse:.5f} V)')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_plot.png')
    plt.show()
    print("Գրաֆիկը պահպանվեց որպես 'test_plot.png'!")

if __name__ == "__main__":
    main()