# Hierarchical-Memory GNN: Weather-aware Flight Delay Prediction

EC601 Course Project (Boston University)  
Authors: Boyang Zhang, Phat Duong, Zhengyu Zhuang, Youwei Chen

## Overview
Flight delays waste time and money. Many delay prediction baselines under-use **spatial-temporal** dependencies across airports and days.

This project builds a **Graph Neural Network (GNN)** to model spatial dependencies of air traffic, and adds a **memory layer** to capture long-range temporal dependencies. We further explore a **hierarchical forecasting + top-down reconciliation** strategy to improve predictions at multiple granularities.

## Problem Statement
Given a flight + weather graph over time, predict:
- **Departure delay**
- **Arrival delay**

## Dataset
- **Flight data**: US DoT
- **Weather data**: NOAA

### Graph Construction
**Nodes** (examples of node features):
- Date
- Airport
- Weather information

**Edges** (examples of edge features):
- Date, carrier, dep/arr time
- Origin, destination
- Distance, elevation, elapsed time

## Method
### 1) Directed Spatial Graph + Message Passing
We model airport-to-airport influence via directed edges and run message passing to learn spatial dependencies.

### 2) Temporal Memory Layer
To retain information across longer horizons, we maintain graph embeddings using a memory mechanism (e.g., LSTM-style update).

### 3) Hierarchical Forecasting + Reconciliation (Top-Down)
Observation: the model can be strong on “total delay” but weaker on each finer granularity.  
We forecast **t+1** at each hierarchy level, then apply a **top-down reconciliation** step to enforce consistency and improve granular predictions.

## Metric
**AUC (Area Under ROC Curve)**  
AUC = P(score(positive) > score(negative))

## Results (Weather AUC %)
| Iteration | Directed Edges | Memory Layer | Weather | AUC(%) |
|---|---:|---:|---:|---:|
| Baseline | ✗ | ✗ | ✗ | 70.49 |
| Step 1 | ✓ | ✗ | ✗ | 76.14 |
| Step 2 (LSTM) | ✓ | LSTM | ✗ | 73.81 |
| Step 3 (Hierarchical) | ✓ | Hierarchical | ✗ | 81.58 |
| Step 4 (Hierarchical + Weather) | ✓ | Hierarchical | ✓ | **85.30** |

## Repository Structure
```text
.
├── weather/                   # place processed data here (not tracked)
│   ├── Baseline
│   ├── addFeature
│   ├── addWeather
│   └── addMemory
└── README.md
```
There is a batch processing training script in Google Drive：https://drive.google.com/drive/folders/1jL8F6WxgZfGf9ee9WkwHq7Jm_MQha2dr?usp=sharing
Main Code by Boyang Zhang
Memory Mechanism by Phat Duong
Weather Data collected and merged by Zhengyu Zhuang
Flight Data collected by Phat Duong and Youwei Chen
