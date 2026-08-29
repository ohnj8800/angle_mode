# FBG 角度狀態與 VMD 模態分析

本專案分析 FBG CH1 波長訊號，利用 VMD、IOVMD、AVMD、SVMD 與 STVMD 拆解不同頻率模態，並比較模態與角度移動、穩定角度及不同循環之間的關係。

程式會找出值得後續檢查的頻率與時間區段，但輸出的「異常候選」不等同於設備故障。

## 環境需求

- Windows 10/11
- Python 3
- CSV 資料至少包含時間與 `CH1` 欄位

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 輸入資料

預設資料放在 `data/raw/`：

```text
angle_10deg_43deg_repeat.csv
angle_10deg_stable.csv
```

更換資料時，需要確認：

- 時間欄位及 `CH1` 欄位名稱正確
- 時間值遞增且沒有大量缺值
- 取樣率與程式設定一致
- 實驗包含可辨識的穩定角度與角度切換區段

## 執行方式

在專案根目錄執行全部分析：

```powershell
python .\run_all_analysis.py
```

執行指定範圍：

```powershell
python .\run_all_analysis.py --from-step 10 --to-step 13
```

略過選配的第 04 步：

```powershell
python .\run_all_analysis.py --skip-step 4
```

## 分析流程

| 步驟 | 功能 |
| ---: | --- |
| 01 | 檢查輸入資料與取樣資訊。 |
| 02 | 估測角度並標記穩定與移動狀態。 |
| 03 | 執行五種 VMD 類方法並輸出模態。 |
| 04 | 補充比較角度移動與穩定期間的模態反應；不是後續步驟的必要輸入。 |
| 05 | 擷取滑動視窗的 RMS、包絡與局部頻率等特徵。 |
| 06 | 依角度狀態建立穩定基準。 |
| 07 | 找出偏離基準的候選視窗。 |
| 08 | 比較不同方法是否支持相同候選。 |
| 09 | 繪製候選事件的訊號、角度與模態證據圖。 |
| 10 | 合併相鄰候選並判斷是否與角度移動有關。 |
| 11 | 使用未經 VMD 的原始 Welch 頻譜驗證候選。 |
| 12 | 比較相同角度的不同穩定循環。 |
| 13 | 建立跨方法的模態頻率與行為總表。 |

## 結果位置與判讀

全部輸出位於 `results/`。建議依序查看：

1. `mode_behavior_atlas/`：各方法共同出現的頻率，以及頻率與角度／循環的關係。
2. `physical_episodes/`：候選發生時間、頻率和事件分類。
3. `raw_spectral_validation/`：候選是否也獲得原始頻譜支持。
4. `stable_cycle_comparison/`：相同角度不同循環的能量與頻率變化。
5. `consensus_event_evidence/`：候選事件圖，供人工檢查。

主要欄位：

- `supporting_methods`：支持候選的方法；方法越多，結果通常越不受單一演算法影響。
- `raw_spectrum_supported`：原始頻譜是否支持候選，但 `True` 仍不代表故障。
- `maximum_consecutive_candidate_windows`：候選連續出現的視窗數，持續性通常比單一高分更重要。
- `episode_to_baseline_rms_ratio`：相對同角度基準的能量比例；大於 1 為增加，小於 1 為降低。
- `episode_classification`：區分與角度移動相關的動態反應，以及穩定角度下的異常候選。

## 注意事項

- 角度為訊號估測結果，不是獨立角度感測器的真值。
- 多種 VMD 方法得到相同結果，只代表候選較穩健，不代表故障機率。
- 若要判定故障或零件來源，仍需健康／故障標籤、重複實驗、轉速、負載、傳動比及零件規格等資料。
- 新資料若具有不同欄位、取樣率、角度流程或感測器頻寬，需要先調整設定。
