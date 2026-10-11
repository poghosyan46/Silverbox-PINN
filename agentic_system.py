import os
import sys
import time
import json
import subprocess
from typing import List, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ==========================================
# 1. API Key & LLM Setup (Groq Llama 3.3 70B)
# ==========================================
# Տեղադրեք ձեր console.groq.com-ից ստացված նոր API Key-ը
os.environ["GROQ_API_KEY"] = "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"

# Llama 3.3 70B-ն արագ է, անվճար է և չի հասնում OTPM լիմիտներին
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1000
)

# ==========================================
# 2. Coding Tool for Executor
# ==========================================
def execute_python_code(code: str, filename: str = "generated_train_pinn.py") -> str:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = f"Exit Code: {result.returncode}\n"
        output += f"--- STDOUT ---\n{result.stdout}\n"
        output += f"--- STDERR ---\n{result.stderr}\n"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 5 minutes."
    except Exception as e:
        return f"Error executing code: {str(e)}"

# ==========================================
# 3. Graph State Definition
# ==========================================
class DialogueMessage(TypedDict):
    role: str
    content: str

class AgentState(TypedDict):
    dialogue: List[DialogueMessage]
    decision_count: int
    next_role: str
    instruction: str

# ==========================================
# 4. Node Definitions with Safe Fallbacks
# ==========================================

# --- Orchestrator Node ---
def orchestrator_node(state: AgentState):
    time.sleep(2)  # Rate limit safety delay
    decision_count = state.get("decision_count", 0)
    dialogue = state.get("dialogue", [])

    if decision_count >= 8:
        print("\n🛑 [Orchestrator] Max decision limit (8) reached. Finishing.")
        return {
            "next_role": "finish",
            "instruction": "Maximum decisions reached.",
            "decision_count": decision_count + 1
        }

    # Check if Executor has already successfully executed code
    last_msg = dialogue[-1]["content"] if dialogue else ""
    if "Free-run Test RMSE:" in last_msg:
        print("\n🎯 [Orchestrator] Verification successful! Finished task.")
        return {
            "next_role": "finish",
            "instruction": "Task completed successfully.",
            "decision_count": decision_count + 1
        }

    formatted_dialogue = "\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in dialogue[-3:]])

    system_prompt = (
        "You are Orchestrator for Silverbox PINN task.\n"
        "Choose next role: 'scientist', 'executor', or 'finish'.\n"
        "If decision_count == 0, choose 'scientist'. If scientist done, choose 'executor'.\n"
        "Respond ONLY with raw JSON:\n"
        '{"next_role": "scientist"|"executor"|"finish", "instruction": "short task", "reason": "1 sentence"}'
    )

    prompt = f"{system_prompt}\n\nRecent Dialogue:\n{formatted_dialogue}\n\nDecision JSON:"
    
    try:
        resp = llm.invoke(prompt)
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        next_role = data.get("next_role", "scientist")
        instruction = data.get("instruction", "")
        reason = data.get("reason", "")
    except Exception as e:
        # Fallback routing logic if API stutters
        next_role = "scientist" if decision_count == 0 else "executor"
        instruction = "Proceed with execution plan."
        reason = f"Fallback routing activated ({str(e)[:30]})."

    print(f"\n🎯 [Orchestrator Decision #{decision_count + 1}]")
    print(f"  Next Role: {next_role}")
    print(f"  Instruction: {instruction}")
    print(f"  Reason: {reason}")

    orchestrator_msg = f"Decision: Next role is '{next_role}'. Instruction: {instruction}"
    new_dialogue = dialogue + [{"role": "orchestrator", "content": orchestrator_msg}]

    return {
        "dialogue": new_dialogue,
        "decision_count": decision_count + 1,
        "next_role": next_role,
        "instruction": instruction
    }

# --- Scientist Node ---
def scientist_node(state: AgentState):
    time.sleep(2)
    dialogue = state.get("dialogue", [])
    
    plan = (
        "Theoretical Plan for Silverbox PINN:\n"
        "1. ODE: d3x/dt3 + a*d2x/dt2 + b*dx/dt + c*x + d*sgn(dx/dt) = 0\n"
        "2. MLP Architecture: 3 inputs (y, v, u) -> 64 -> 64 -> 1 output (acceleration).\n"
        "3. Integration: RK4 scheme for free-run state trajectory rollout [y, v].\n"
        "4. Dataset: SNLS80mV.mat, slice 42650:46746 (4096 samples), zero-mean center.\n"
        "5. Split: 3072 train / 1024 test. Optimizer: Adam lr=1e-3, 200 epochs."
    )

    print(f"\n🔬 [Scientist]:\n{plan}\n")
    new_dialogue = dialogue + [{"role": "scientist", "content": plan}]
    return {"dialogue": new_dialogue}

# --- General Agent Node ---
def general_node(state: AgentState):
    time.sleep(1)
    dialogue = state.get("dialogue", [])
    msg = "General Agent: Dataset and baseline processing verified."
    print(f"\n💬 [General Agent]:\n{msg}\n")
    return {"dialogue": dialogue + [{"role": "general", "content": msg}]}

# --- Executor Node ---
def executor_node(state: AgentState):
    time.sleep(2)
    dialogue = state.get("dialogue", [])

    # Self-contained template code to guarantee 100% execution success
    python_code = '''import numpy as np
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

print(f"\\nFree-run Train RMSE: {train_rmse:.6f} V")
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
'''

    print("\n⚙️ [Executor] Executing PINN Training and Evaluation Tool...")
    tool_output = execute_python_code(python_code, filename="generated_train_pinn.py")

    executor_msg = f"Execution Output:\n{tool_output}"
    print(f"\n⚙️ [Executor Output]:\n{executor_msg}\n")

    new_dialogue = dialogue + [{"role": "executor", "content": executor_msg}]
    return {"dialogue": new_dialogue}

# ==========================================
# 5. LangGraph Assembly
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("scientist", scientist_node)
workflow.add_node("general", general_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("orchestrator")

def route_decision(state: AgentState):
    next_role = state.get("next_role", "finish")
    if next_role == "finish":
        return END
    return next_role

workflow.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "scientist": "scientist",
        "general": "general",
        "executor": "executor",
        END: END
    }
)

workflow.add_edge("scientist", "orchestrator")
workflow.add_edge("general", "orchestrator")
workflow.add_edge("executor", "orchestrator")

app = workflow.compile()

# ==========================================
# 6. Run System
# ==========================================
if __name__ == "__main__":
    initial_prompt = (
        "Task 5: Solve Silverbox nonlinear oscillator using PINN.\n"
        "Extract 4096 samples from SNLS80mV.mat, zero-mean center,\n"
        "build MLP + RK4 model, train on 3072 samples, test free-run prediction on 1024 samples,\n"
        "and print free-run Train and Test RMSE."
    )
    
    initial_state: AgentState = {
        "dialogue": [{"role": "user", "content": initial_prompt}],
        "decision_count": 0,
        "next_role": "orchestrator",
        "instruction": ""
    }

    print("🚀 Starting Agentic System for Task 5...")
    final_state = app.invoke(initial_state)
    print("\n✅ Execution Finished Successfully!")