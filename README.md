# 🛸 無人機人群流量分析系統 (UAV Crowd Counting)

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.3-green.svg)

## 📌 專案簡介
本專案專為 **無人機 (UAV)** 高空視角設計。不同於一般影像辨識主要針對人臉或全景，無人機在高空拍攝的照片多為 **「俯視人頭」** 視角，且行人物件極為微小且密集。

本系統基於 **MobileCount** 深度學習模型，採用 **「密度估計 (Density Estimation)」** 技術而非傳統畫框偵測。此方法能有效解決傳統 YOLO 偵測器在處理「極小人頭」與「高度擁擠」時容易漏抓或誤判的問題，進而精準計算人群總數並分析移動趨勢。

---

## 💻 環境配置 (Environment)

### 核心安裝指令 (GPU 加速版本)
請確保您的 `(venv)` 虛擬環境已啟動，並依照以下順序安裝：

```powershell
# 安裝特定版本的 PyTorch 以支援 GPU 運算
pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)

# 升級必要工具包
pip install --upgrade numpy scipy hdbscan
```
硬體測試
測試環境：本專案已在 NVIDIA GeForce RTX 4060 上測試成功。

效能預估：處理 1080x1920 影像之速度約為 1.94 FPS。

🚀 核心執行指令 (Usage)
1. 硬體效能與 GPU 測試
用於確認環境是否能讀取顯卡，並查看處理速度。

指令：
```python test_gpu.py```

2. 正式運行預測模式
對指定的圖片資料夾或影片進行分析。

指令範例：
```python run.py --path images256 --callbacks count_callback display_callback```

## 🛠️ 功能選項說明 (Callbacks)

執行 `run.py` 時，可自由組合以下功能（Callbacks）來決定輸出結果：

| 功能名稱 (Callback) | 輸出效果描述 | 最佳適用場景 |
| :--- | :--- | :--- |
| **`count_callback`** | 在終端機即時印出每張影像的預估總人數。 | 快速數據統計、分析日誌。 |
| **`display_callback`** | 彈出視窗對照「原始圖」與「彩色密度圖」。 | 單張影像分析、效果演示。 |
| **`save_callback`** | 自動將預測的彩色密度圖存成 `.png` 檔案。 | 大量圖片自動化批量處理。 |
| **`video_callback`** | 使用 OpenCV 窗口即時播放預測畫面。 | 分析動態無人機影片檔案。 |
| **`track_callback`** | 導出標準化的 `.h5` 密度矩陣數據。 | 進階時空流量與位移研究。 |

開發者調試備忘錄 (Dev Notes)
路徑修正：檢查 config.py 中的 PRE_TRAINED 變數，已改為自動偵測的相對路徑。

模型初始化：CrowdCounter 必須包含參數初始化：

```cc = CrowdCounter(gpus='0', model_name='MobileCount')```

視窗暫停：使用 display_callback 時必須手動關閉圖片視窗，程式才會繼續處理下一張。
