import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

print("Loading SNLS80mV.mat data...")
mat = sio.loadmat('SNLS80mV.mat')

V1 = mat['V1'].flatten()
V2 = mat['V2'].flatten()

u_raw = V1[42650:46746].astype(np.float32)
y_raw = V2[42650:46746].astype(np.float32)
dt = 1.0 / 8000.0

u = u_raw - np.mean(u_raw)
y = y_raw - np.mean(y_raw)
v = np.gradient(y, dt).astype(np.float32)

n_train = 3072
u_train, y_train, v_train = u[:n_train], y[:n_train], v[:n_train]
u_test, y_test, v_test = u[n_train:], y[n_train:], v[n_train:]

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, y_val, v_val, u_val):
        x = torch.cat([y_val, v_val, u_val], dim=-1)
        return self.net(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def rk4_step(model, z, u_curr, dt):
    def f(z_in, u_in):
        y_curr = z_in[:, 0:1]
        v_curr = z_in[:, 1:2]
        acc = model(y_curr, v_curr, u_in)
        return torch.cat([v_curr, acc], dim=-1)

    k1 = f(z, u_curr)
    k2 = f(z + 0.5 * dt * k1, u_curr)
    k3 = f(z + 0.5 * dt * k2, u_curr)
    k4 = f(z + dt * k3, u_curr)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

u_tr_t = torch.tensor(u_train, dtype=torch.float32).unsqueeze(-1)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
v_tr_t = torch.tensor(v_train, dtype=torch.float32).unsqueeze(-1)

epochs = 200
rollout_len = 32

print("Training PINN model...")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss = 0.0
    indices = np.random.randint(0, n_train - rollout_len, size=32)
    for idx in indices:
        z = torch.cat([y_tr_t[idx:idx+1], v_tr_t[idx:idx+1]], dim=-1)
        for step in range(rollout_len):
            u_c = u_tr_t[idx + step : idx + step + 1]
            z = rk4_step(model, z, u_c, dt)
            target_y = y_tr_t[idx + step + 1 : idx + step + 2]
            loss += torch.mean((z[:, 0:1] - target_y) ** 2)

    loss = loss / (len(indices) * rollout_len)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    z_tr = torch.tensor([[y_train[0], v_train[0]]], dtype=torch.float32)
    y_pred_train = []
    for i in range(len(u_train)):
        y_pred_train.append(z_tr[0, 0].item())
        z_tr = rk4_step(model, z_tr, torch.tensor([[u_train[i]]], dtype=torch.float32), dt)
    train_rmse = np.sqrt(np.mean((np.array(y_pred_train) - y_train) ** 2))

    z_te = torch.tensor([[y_test[0], v_test[0]]], dtype=torch.float32)
    y_pred_test = []
    for i in range(len(u_test)):
        y_pred_test.append(z_te[0, 0].item())
        z_te = rk4_step(model, z_te, torch.tensor([[u_test[i]]], dtype=torch.float32), dt)
    test_rmse = np.sqrt(np.mean((np.array(y_pred_test) - y_test) ** 2))

print(f"\nFree-run Train RMSE: {train_rmse:.6f} V")
print(f"Free-run Test RMSE: {test_rmse:.6f} V")

plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True Test Signal', color='blue', alpha=0.7)
plt.plot(y_pred_test, label='PINN Free-run Prediction', color='red', linestyle='--')
plt.title(f'Silverbox Oscillator Free-run Test Prediction (RMSE: {test_rmse:.6f} V)')
plt.xlabel('Sample')
plt.ylabel('Voltage V2 (V)')
plt.legend()
plt.grid(True)
plt.savefig('test_plot.png', dpi=300)
plt.close()
print("Plot saved as test_plot.png")
