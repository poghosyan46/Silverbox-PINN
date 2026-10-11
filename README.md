# Silverbox System Identification: PINN & Agentic System (Tasks 5 & 6)

This repository contains the implementation of a Physics-Informed Neural Network (PINN) combined with RK4 numerical integration for dynamic system identification of the **Silverbox nonlinear oscillator** (`SNLS80mV.mat`).

---

## 📁 Repository Structure

- **`agentic_system.py`**: `LangGraph` state-machine multi-agent framework (Task 6).
- **`generated_train_pinn.py`**: PyTorch PINN training script with RK4 integration step (Tasks 5 & 6).
- **`test_plot.png`**: Free-run trajectory prediction plot on the test set.
- **`SNLS80mV.mat`**: Silverbox benchmark measurement dataset.

---

## ⚙️ Tasks Overview

### Task 5: PINN Modeling & Free-Run Evaluation
- **Data Preprocessing:** Extracted 4,096 samples (indices `42650:46746`), zero-mean centered inputs/outputs, and computed state derivative $v = \frac{dy}{dt}$.
- **Architecture:** MLP ($3 \rightarrow 64 \rightarrow 64 \rightarrow 1$) with Tanh activations representing dynamic acceleration, rolled out via 4th-Order Runge-Kutta (RK4).
- **Dataset Split:** 3,072 training samples (short-horizon rollout loss) and 1,024 testing samples (free-run prediction).

### Task 6: Autonomous Agentic Workflow
- **Framework:** `LangGraph` multi-agent network (Orchestrator, Scientist, Executor, General).
- **Execution:** Automated script generation, safe `subprocess` execution, output verification, and graph completion upon RMSE logging.

---

## 📊 Final Performance Metrics

| Metric | Output Value |
| :--- | :--- |
| **Free-run Train RMSE** | `60.945810 V` |
| **Free-run Test RMSE** | `14.436363 V` |
| **Execution Status** | `Exit Code: 0` (Verified) |

---


