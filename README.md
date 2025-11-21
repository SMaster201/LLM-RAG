# Qwen RAG 系統 — 使用說明

這份文件說明如何在 Windows（PowerShell）上建立環境、安裝所需套件、下載模型，並啟動系統或執行效能測試。假設你已把工作資料夾放在 `C:\Users\<you>\OneDrive\Desktop\專題\VLM+rag`。

**重點檔案**
- `download_models.py` — 用來下載/快取 Qwen 模型到本地 `model/` 資料夾（支援 CLI `--all` / `--model`，也保留互動式選單）。
- `rag_performance_test.py` — 針對多個模型執行 LLM+RAG 的效能測試（會載入 `PDF/` 中的 PDF、對問題做檢索+生成、儲存 JSON/Excel 報告）。會嘗試先從 `test_questions.json` 讀取題目，找不到則使用內建 fallback。
- `qwen_rag_system.py` — 系統核心，提供 `QwenRAGSystem` 類（載模型、建立向量庫、檢索、生成、互動式問答）。可直接執行進入互動問答模式。
- `download_selected_models.py` — 簡單的非互動批次下載到 HF cache（次要）。
- `test_questions.json` — 問題集合（RAG 測試可讀取）。
- `PDF/` — 放入要做 RAG 的 PDF 文件（`rag_performance_test.py` 會掃描此資料夾）。
- `model/` — 本地模型快取（`download_models.py` 會存放於此）。


**1) 建置環境（建議：使用 Python 3.10+）**

1. 開啟 PowerShell，切換到專案資料夾：
```powershell
Set-Location -Path "C:\Users\SMaster201\OneDrive\Desktop\專題\VLM+rag"
```

2. 建立虛擬環境並啟動（Windows PowerShell）：
```powershell
python -m venv .\venv
& .\venv\Scripts\Activate.ps1
```

3. 更新 pip、wheel：
```powershell
python -m pip install --upgrade pip wheel setuptools
```

4. 安裝主要相依套件（使用專案中的 `requirements.txt`）：
```powershell
pip install -r .\requirements.txt
```

5. 額外需要安裝的套件（`requirements.txt` 可能未列出但腳本會用到）：
```powershell
pip install huggingface_hub pandas openpyxl psutil nvidia-ml-py3
```
- `huggingface_hub`：下載模型時用到
- `pandas`, `openpyxl`：生成 Excel 報告
- `psutil`：效能監控
- `nvidia-ml-py3`：若要監控 GPU VRAM（可選，無 NVIDIA 時程式會退回監控 RAM）

6. 如果你要在 GPU 上使用大型模型並希望使用 `bitsandbytes`，請依作業系統/硬體安裝對應版本的 `bitsandbytes`。


**2) 下載模型 — 使用哪個檔案？**

- 推薦使用 `download_models.py`（互動 + CLI）。

常見用法：
- 下載全部（注意空間巨大）：
```powershell
python .\download_models.py --all
```
- 下載指定模型（可以重複 `--model`）：
```powershell
python .\download_models.py --model Qwen/Qwen2-0.5B-Instruct
```
- 互動模式（沒有參數）：
```powershell
python .\download_models.py
```

下載後，模型會被存到 `model/` 資料夾，路徑命名會把 `/` 換成 `__`，例如 `model\Qwen__Qwen2-0.5B-Instruct`。

備註：`download_selected_models.py` 也能批次下載（會存到 Hugging Face 的 cache），但 `download_models.py` 會把模型放在 `./model`，較方便離線使用。


**3) 下載完成後要執行哪個檔案才能使用功能？**

- 想直接互動、根據你放的 PDF 或 TXT 做 RAG 問答：
  - 執行 `qwen_rag_system.py`：
  ```powershell
  python .\qwen_rag_system.py
  ```
  - 會啟動互動模式，程式會提示你輸入問題；若要使用文件，先在程式中選擇載入文件或在碼中呼叫 `load_documents(...)`。

- 想跑完整的七個模型效能測試（批次載入 `PDF/` 中文件、對多題做檢索 + 生成、產生 JSON/Excel 報告）：
  - 放好要測試的 PDF 到 `PDF/` 資料夾，然後執行：
  ```powershell
  python .\rag_performance_test.py
  ```
  - 程式會：
    - 依 `MODELS` 列表（檔案內定義）逐一載入模型（若模型未放在 `model/`，`qwen_rag_system` 會嘗試從 HF 下載），
    - 載入 `PDF/` 所有 PDF，建立向量庫（Chroma）並進行 RAG，
    - 對 `test_questions.json` 中的問題或 fallback 問題做檢索 + 生成，
    - 將結果寫到 `rag_performance_test_results.json` 與 `RAG效能測試報告.xlsx`。

- 快速 smoke-test（小型示例）：
  - `example_usage.py`：展示如何使用 `QwenRAGSystem` 的不同模式（含不使用 RAG 的純模型對話）。
  ```powershell
  python .\example_usage.py
  ```


**常見問題與提醒**
- 確保 `PDF/` 中有 PDF 檔，否則 `rag_performance_test.py` 會停止並顯示錯誤。
- 若要節省下載流量，先用 `download_models.py --model <repo>` 下載你要測試的模型到 `model/`。
- 若使用 GPU，請確認驅動與 CUDA 相容，且 `torch` 已安裝對應的 CUDA 版本；大型模型可能需要大量 VRAM。
- 若在 Windows 上出現 symlink 問題，程式會自動設定 `HF_HUB_DISABLE_SYMLINKS=1`。


如果你希望我：
- 把 `rag_performance_test.py` 加上 CLI（例如 `--pdf-folder`, `--questions-file`, `--models`）以便自動化執行；或
- 把這個 README 的內容同步到 `新手執行手冊.md`，或建立更詳細的安裝腳本（PowerShell） — 我可以接著幫你做。 

