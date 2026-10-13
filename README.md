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

### 固定角度、多次重複量測

`run_batch_static_analysis.py` 用於像「無負載10度 (1).csv」這類固定角度、
多次重複量測的資料。這條流程不執行角度切換偵測，而是比較資料品質、
平均波長、AC RMS、Welch PSD及五種VMD方法的模態結果。

將CSV放入 `data/raw/no_load_30hz/` 後執行：

```powershell
python .\run_batch_static_analysis.py
```

若資料位於其他資料夾：

```powershell
python .\run_batch_static_analysis.py --input-dir "D:\data\無負載"
```

只先執行資料品質與Welch PSD：

```powershell
python .\run_batch_static_analysis.py --basic-only
```

批次結果預設輸出至 `results/batch_no_load_30hz/`。程式接受原專案的
`time`、`CH1` 欄位，也接受解調器匯出的 `時間戳記`、`Ch1` 欄位。

批次流程會自動輸出下列圖表：

- 基本特徵逐次比較與角度分布圖。
- Welch PSD中位數、每筆PSD熱圖、主頻逐次變化與43°／10° PSD比值。
- 25–32、50–65、78–88與90–99 Hz頻帶功率分布。
- 五種方法的模態頻率比較、重建品質、運算時間／模態數與跨重複量測頻率覆蓋率。

其中90–99 Hz接近200 Hz取樣率的Nyquist邊界，只應視為候選高頻成分。

### 260720整包資料：先前處理，再執行AVMD

將解壓後的原始資料放在：

```text
data/raw/260720/
```

先只執行168個CSV的資料品質檢查與前處理對照：

```powershell
python .\run_archive_avmd_analysis.py --preprocess-only
```

確認 `results/archive_avmd/preprocessing_before_after.csv` 沒有失敗後，
執行整包AVMD：

```powershell
python .\run_archive_avmd_analysis.py
```

新版流程預設將AVMD輸入限制在0.5–40 Hz，先由代表檔選擇初始K與alpha，
再對每個CSV執行adaptive-K迴圈。adaptive-K會比較重建誤差、模態頻譜重疊、
模態相關性、Welch PSD支持度與模態數複雜度，因此不同檔案可以得到不同K。

先用3個檔案測試新版adaptive-K：

```powershell
python .\run_archive_avmd_analysis.py --max-files 3 --skip-per-file-plots
```

若要重現舊版「全部檔案共用K」：

```powershell
python .\run_archive_avmd_analysis.py --disable-adaptive-k
```

若先測試流程，可限制前3個檔案並減少AVMD參數候選：

```powershell
python .\run_archive_avmd_analysis.py --max-files 3 --avmd-k 4 --avmd-alpha 500
```

這個入口預設只執行AVMD，但VMD、IOVMD、SVMD與STVMD都保留。例如：

```powershell
python .\run_archive_avmd_analysis.py --methods VMD IOVMD AVMD SVMD STVMD
```

前處理原則如下：

- 接受 `時間戳記/Ch1`、`time/CH1` 與 `ts/wl` 欄名。
- 移除完全重複列並記錄數量，不把偏心或高振動資料當成雜訊刪除。
- 依已知200 Hz取樣率建立均勻時間軸。
- 原始波長保留供角度與平均波長比較。
- 0.1 Hz低通分量供慢變趨勢觀察；0.5 Hz高通訊號供AVMD分析。
- 每個CSV輸出前處理前後時域、Welch PSD及AVMD模態圖。

主要總覽結果：

- `01_data_quality_overview.png`
- `02_angle_vs_mean_wavelength.png`
- `03_angle_vs_rms_before_after.png`
- `04_angle_vs_frequency_before_after.png`
- `05_angle_vs_avmd_dominant_frequency.png`
- `06_angle_vs_avmd_energy_ratio.png`
- `07_avmd_reconstruction_quality.png`
- `08_adaptive_k_by_angle.png`
- `preprocessing_before_after.csv`
- `method_file_summary.csv`
- `all_mode_metrics.csv`
- `adaptive_k_trials.csv`

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
