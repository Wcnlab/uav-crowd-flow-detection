# 🛸 無人機人群流量分析系統 (UAV Crowd Counting)

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.3-green.svg)

## 📌 專案簡介
本專案基於 **MobileCount** 深度學習模型，專為無人機拍攝的高空影像設計。它利用「密度估計 (Density Estimation)」技術，能精準計算極其微小且擁擠的人群數量，並分析人群移動趨勢。

相比於傳統的物件偵測（如 YOLO），本系統在處理高空、高密度、小像素人頭時具有更強的魯棒性，且適合部署於 Edge 端進行即時推論。

---

## 💻 環境配置 (Environment)

### 核心安裝指令 (GPU 加速版本)
請確保您的 `(venv)` 虛擬環境已啟動，並依照以下順序安裝：

```powershell
# 安裝特定版本的 PyTorch 以支援 GPU 運算
pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)

# 升級必要工具包
pip install --upgrade numpy scipy hdbscan
硬體測試測試環境：本專案已在 NVIDIA GeForce RTX 4060 上測試成功。效能預估：處理 1080x1920 影像之速度約為 1.94 FPS。🚀 核心執行指令 (Usage)1. 硬體效能與 GPU 測試用於確認環境是否能正確讀取顯卡，並查看當前硬體的處理效能。指令：python test_gpu.py關鍵數據：Device: 顯示當前使用的顯卡型號。mean (fps): 每秒處理張數。2. 正式運行預測模式對指定的資料夾圖片或影片進行分析。指令格式：PowerShellpython run.py --path <資料夾或影片路徑> --callbacks <功能1> <功能2>
範例：PowerShellpython run.py --path images256 --callbacks count_callback display_callback
🛠️ 功能選項說明 (Callbacks)您可以根據需求，在執行 run.py 時組合多種功能：功能名稱 (Callback)輸出效果描述最佳適用場景count_callback在終端機即時印出每張影像的預估總人數。快速數據統計、分析日誌。display_callback彈出視窗對照「原始圖」與「彩色密度圖」。單張影像分析、效果演示。save_callback自動將密度預測圖存成 .png 檔案。大量圖片自動化後處理。video_callback使用 OpenCV 視窗即時播放與預測畫面。分析動態無人機影片檔。track_callback導出標準化的 .h5 密度數據。進階時空流量與位移研究。⚠️ 開發者調試備忘錄 (Dev Notes)1. 路徑自動偵測 (去 "seven" 化)本專案已優化路徑處理邏輯，請避免使用絕對路徑：config.py: 已改為使用 os.path.abspath 自動獲取 BASE_DIR。所有權重路徑將自動定位於根目錄的 /exp 資料夾下。run.py: 確保載入模型的路徑與 config.py 的設定保持一致。2. 模型初始化參數在自定義啟動腳本時，CrowdCounter 類別必須明確傳入顯卡編號與模型名稱：Pythoncc = CrowdCounter(gpus='0', model_name='MobileCount')
3. 操作小提醒視窗暫停：使用 display_callback 時，程式會因為等待使用者查看圖片而暫停。必須手動關閉當前圖片視窗，Python 才會繼續處理下一張影像。縮排注意：修改 run.py 時請確保 if __name__ == '__main__': 內的縮排整齊，避免 IndentationError。
