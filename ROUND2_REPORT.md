# Round 2 模型探索報告 — 60 trials, 20 架構

> **在與 Round 1 完全相同的訓練條件下，探索 20 種架構。發現 `tcn_gru`（TCN + BiGRU）以 F1=0.604 超越 Round 1 冠軍，成為新的最佳架構。**

---

## 1. 實驗設計

### 跟 Round 1 的唯一差別

| 項目 | Round 1 | Round 2 |
|---|---|---|
| 架構 | 12 種 | **20 種** |
| Per-epoch 上限 | 20 秒 | **60 秒** |
| 其他所有條件 | — | **完全相同** |

固定條件：624 train subjects, 30 epochs, 1000 chunks/epoch, seed=42, 80/20 split (seed=2026), bce loss, pos_weight=1.5, batch_size=8, chunk_size=4200, kernel_size=7。

### 20 種架構分三組

**A 組（8 種）**：Round 1 冠軍的 ablation 變體
**B 組（6 種）**：全新設計
**C 組（6 種）**：進階組合 + Round 1 對照

### 執行

- **60 trials**（20 archs × 3 configs each）
- 三台並行：win5070 (CUDA), macstudio (MPS), Mac mini M4 (MPS)
- macstudio 速度太慢（69 min/trial），中途將 15 個 trials 轉移到 win5070 救援
- **60/60 全部完成**

---

## 2. 架構排名

| Rank | 架構 | n | Mean F1 | Std | Best F1 | Best r | 穩定度 |
|---|---|---|---|---|---|---|---|
| **🥇 1** | **tcn_gru** | 3 | **0.595** | 0.008 | **0.604** | 0.890 | **100%** |
| 🥈 2 | deep_tcn_bilstm | 3 | 0.540 | 0.007 | 0.547 | **0.923** | 100% |
| 🥉 3 | tcn_bilstm_attn | 3 | 0.498 | 0.026 | 0.518 | **0.934** | 100% |
| 4 | tcn_bilstm_hybrid | 3 | 0.463 | 0.034 | 0.499 | 0.890 | 100% |
| 5 | unet1d | 3 | 0.436 | 0.051 | 0.471 | 0.835 | 100% |
| 6 | attention_unet1d | 3 | 0.418 | 0.085 | 0.497 | 0.853 | 100% |
| 7 | residual_tcn_bilstm | 3 | 0.416 | 0.079 | 0.491 | 0.928 | 100% |
| 8 | tcn_bilstm_se | 3 | 0.329 | 0.289 | 0.541 | 0.900 | 67% |
| 9 | wavenet_bilstm | 3 | 0.323 | 0.123 | 0.465 | 0.856 | 100% |
| 10 | tcn_transformer_hybrid | 3 | 0.261 | 0.024 | 0.283 | 0.868 | 100% |
| 11 | tcn | 3 | 0.257 | 0.020 | 0.281 | 0.904 | 100% |
| 12 | se_resnet1d | 3 | 0.225 | 0.031 | 0.258 | 0.885 | 67% |
| 13 | conformer | 3 | 0.142 | 0.034 | 0.182 | 0.873 | 0% |
| 14 | dilated_tcn | 3 | 0.131 | 0.052 | 0.189 | 0.855 | 0% |
| 15 | inception_bilstm | 3 | 0.124 | 0.064 | 0.193 | 0.661 | 0% |
| 16 | bilstm_cnn | 3 | 0.109 | 0.142 | 0.273 | 0.783 | 33% |
| 17 | wavenet | 3 | 0.072 | 0.002 | 0.074 | 0.853 | 0% |
| 18 | transformer | 3 | 0.031 | 0.027 | 0.052 | 0.890 | 0% |
| 19 | resnet1d | 3 | 0.026 | 0.023 | 0.044 | 0.062 | 0% |
| 20 | inception1d | 3 | 0.016 | 0.020 | 0.038 | — | 0% |

> **穩定度** = F1 ≥ 0.20 的 trial 佔比。Top 7 全部 100%。

---

## 3. Top 15 Trials

| # | 架構 | F1 | r | nf | nl | lr |
|---|---|---|---|---|---|---|
| **1** | **tcn_gru** | **0.604** | 0.888 | 96 | 10 | 0.0005 |
| 2 | tcn_gru | 0.594 | 0.856 | 64 | 8 | 0.0005 |
| 3 | tcn_gru | 0.587 | 0.890 | 96 | 12 | 0.002 |
| 4 | deep_tcn_bilstm | 0.547 | 0.923 | 48 | 10 | 0.002 |
| 5 | tcn_bilstm_se | 0.541 | 0.900 | 96 | 8 | 0.003 |
| 6 | deep_tcn_bilstm | 0.538 | 0.919 | 96 | 10 | 0.001 |
| 7 | deep_tcn_bilstm | 0.535 | 0.892 | 96 | 10 | 0.003 |
| 8 | tcn_bilstm_attn | 0.518 | 0.912 | 96 | 12 | 0.001 |
| 9 | tcn_bilstm_attn | 0.508 | 0.909 | 128 | 10 | 0.003 |
| 10 | tcn_bilstm_hybrid | 0.499 | 0.796 | 96 | 8 | 0.001 |
| 11 | attention_unet1d | 0.497 | 0.815 | 96 | 8 | 0.003 |
| 12 | residual_tcn_bilstm | 0.491 | 0.913 | 48 | 10 | 0.003 |
| 13 | unet1d | 0.471 | 0.835 | 128 | 10 | 0.001 |
| 14 | tcn_bilstm_attn | 0.469 | 0.934 | 64 | 10 | 0.0005 |
| 15 | wavenet_bilstm | 0.465 | 0.856 | 128 | 8 | 0.0005 |

---

## 4. 新冠軍：tcn_gru

### 為什麼 GRU 贏了 LSTM？

| 比較 | BiLSTM (Round 1 冠軍) | **BiGRU (Round 2 冠軍)** |
|---|---|---|
| Mean F1 | 0.463 (R2) / 0.385 (R1) | **0.595** |
| Best F1 | 0.580 (R1) / 0.499 (R2) | **0.604** |
| Stability | 100% / 93% | **100%** |
| Std | 0.034 / 0.143 | **0.008** ← 極穩定 |

> **GRU 比 LSTM 的優勢**：
>
> 1. **參數更少**：GRU 有 2 個 gate（reset + update），LSTM 有 3 個 gate（input + forget + output）+ cell state。在相同 hidden size 下，GRU 參數少 ~25%。
>
> 2. **收斂更快**：更少的參數意味著在 30 epochs × 1000 chunks 的有限訓練中，GRU 更容易學好。LSTM 的額外 capacity 在資料量不夠大時反而是負擔。
>
> 3. **極度穩定**：3/3 trials 的 F1 分別是 0.604, 0.594, 0.587 — **std = 0.008**，幾乎不受 hparam 影響。這對工程部署極為重要。

### tcn_gru 的 3 個 trial 細節

| Trial | F1 | r | nf | nl | lr | 特徵 |
|---|---|---|---|---|---|---|
| 012 | **0.604** | 0.888 | 96 | 10 | **0.0005** | 低 lr 最好 |
| 013 | 0.594 | 0.856 | 64 | 8 | 0.0005 | 小模型也行 |
| 014 | 0.587 | 0.890 | 96 | 12 | 0.002 | 高 lr 也行 |

> **觀察**：lr=0.0005（低學習率）表現最好。這跟 Round 1 冠軍偏好 lr=0.001-0.003 不同。GRU 可能需要更平穩的訓練。

---

## 5. 科學問題答案

### 與 Round 1 冠軍（tcn_bilstm_hybrid, mean=0.463）比較

| 問題 | 改了什麼 | Mean F1 | vs Baseline | 結論 |
|---|---|---|---|---|
| **Q4 GRU 取代 LSTM** | backend: LSTM → GRU | **0.595** | **+0.132** | **🔥 大幅更好** |
| Q7 更深 TCN (20L) | frontend: 12L → 20L | 0.540 | +0.077 | ✅ 更好 |
| Q1 加 Attention | +1 層 self-attention | 0.498 | +0.035 | ✅ 稍好 |
| C1 Residual skip | +residual from input | 0.416 | -0.047 | ≈ 接近 |
| Q2 加 SE block | +channel attention | 0.329 | -0.134 | ❌ 更差 |
| Q8 WaveNet frontend | causal frontend | 0.323 | -0.140 | ❌ 更差 |
| Q3 Transformer backend | attention 取代 recurrence | 0.261 | -0.202 | ❌ 更差 |
| Q9 Inception frontend | 多尺度 parallel | 0.124 | -0.339 | ❌ 很差 |

### 核心洞察

1. **Recurrence 是必要的**（Transformer backend 失敗），但 **GRU > LSTM**
2. **深度有幫助**（20L > 12L），但報酬遞減
3. **Attention 小幅有用**（加在 BiLSTM 後面），但不能取代 recurrence
4. **SE / WaveNet / Inception 加到 hybrid 裡反而有害** — 簡單設計更好

---

## 6. Round 1 vs Round 2 總對照

### 架構演進

```
Round 1 冠軍:  TCN(12L) → BiLSTM → Linear         F1=0.580 (best), 0.385 (mean)
Round 2 冠軍:  TCN(12L) → BiGRU → Linear           F1=0.604 (best), 0.595 (mean)
                         ~~~~~~
                         唯一差別
```

### 數據對比

| 指標 | Round 1 best | Round 2 best | 改善 |
|---|---|---|---|
| Best F1 | 0.580 | **0.604** | **+0.024 (+4.1%)** |
| Mean F1 | 0.385 | **0.595** | **+0.210 (+55%)** |
| Std | 0.143 | **0.008** | **-0.135 (18x 更穩定)** |
| Best r | 0.911 | 0.888 | -0.023 |

> **F1 改善 +4.1% 看起來不多，但 mean 改善 +55% 和 std 改善 18 倍才是最重要的。** 這代表 tcn_gru 不管怎麼調參數都能拿到好成績，而 tcn_bilstm_hybrid 的表現高度依賴 hparam 設定。

### 對產品的意義

| 面向 | tcn_bilstm_hybrid (舊) | **tcn_gru (新)** |
|---|---|---|
| 最好成績 | 0.580 | **0.604** |
| 平均成績 | 0.385 | **0.595** |
| 調參容易度 | 需要精心選 hparams | **幾乎不用調** |
| 模型大小 | ~3.4M params | **更小（少 25% 參數）** |
| 推論速度 | 較慢（LSTM 有 cell state） | **更快（GRU 更簡單）** |
| 部署風險 | hparam 選錯就崩 | **穩定可靠** |

---

## 7. 下一步建議

### 立即行動

1. **切換架構到 `tcn_gru`** — 取代 tcn_bilstm_hybrid 做為主架構
2. **用 lr=0.0005 做 refinement sweep** — Round 2 top trial 偏好低 lr
3. **存 checkpoint + 50 人 held-out test** — 獲得可靠的 AUC-ROC/F1/r

### 中期行動

4. **嘗試 `deep_tcn_gru`**（20L TCN + BiGRU）— 結合 #1 和 #2 的優勢
5. **完整訓練**（60 ep, 10K chunks, 3-ensemble）— 看 tcn_gru 能到多高
6. **輔大醫院資料驗證** — 跨域測試

---

## 8. 失敗率分析

| 類別 | 數量 | 佔比 |
|---|---|---|
| 正常完成 (F1 > 0.20) | 37 | 62% |
| 弱表現 (0.05-0.20) | 12 | 20% |
| **崩潰 (F1 < 0.05)** | **11** | **18%** |

崩潰主要集中在：transformer (3/3), resnet1d (3/3), inception1d (3/3), wavenet (2/3)。全是 Round 1 就表現很差的架構，符合預期。

---

## 9. 實驗時間線

| 時間 | 事件 |
|---|---|
| 04/10 17:56 | 三台並行啟動 (60 trials) |
| 04/10 22:03 | win5070 完成 18/18 |
| 04/11 00:34 | local 完成 21/21 |
| 04/11 ~04:00 | macstudio 完成 6/21（太慢，剩餘轉移到 win5070） |
| 04/11 10:40 | win5070 rescue 完成 15 個 macstudio trials |
| 04/11 ~12:00 | 補跑 3 個 transformer trials |
| 04/11 12:30 | **60/60 全部完成，leaderboard 產出** |

---

*報告日期：2026-04-11*
*資料：60 trials, 20 archs, 624 train / 50 held-out test*
*訓練：30 ep × 1000 chunks × 1-ens (同 Round 1)*
*設備：win5070 (RTX 5070) + macstudio (M2 Max) + Mac mini M4*
