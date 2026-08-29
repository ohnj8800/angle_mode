# FBG角度模態與VMD動態分析

本專案使用FBG Bragg波長訊號，辨識10°與43°角度狀態、角度切換時間，
並比較IOVMD、AVMD與SVMD所分離的動態模態頻率及移動響應。

## 研究定義

- 10D代表10°，43D代表43°。
- 10°與43°都是正常角度狀態，不把43°直接視為異常。
- 沒有外部角度時間標記，角度切換由低頻波長成分自行辨識。
- 異常偵測必須等正常角度位置與正常移動模態建立後再進行。

## 分析流程

```text
原始FBG訊號
  ├─ DC與0.1 Hz以下低頻成分 → 10°/43°估計與切換事件偵測
  ├─ 0.5–100 Hz動態成分 → VMD參數與動態中心頻率選擇
  └─ 保留DC的中心化完整訊號 → IOVMD / AVMD / SVMD
                                      ↓
                   角度位置模態 + 各動態IMF頻率與能量
                                      ↓
                         移動期間 / 穩定期間比較
```

角度路徑不執行線性去趨勢或0.5 Hz高通，避免刪除角度平台。
0.5 Hz高通訊號只用來選擇動態模態的參數；最終VMD輸入仍保留
低頻角度成分，使角度位置模態與動態模態可在同一次分解中辨識。

## 資料

將下列原始CSV放入`data/raw/`：

- `angle_10deg_43deg_repeat.csv`
- `angle_10deg_stable.csv`

CSV欄位必須包含：

```text
time,CH1
```

取樣率設定為200 Hz。

## 執行

安裝套件：

```powershell
pip install -r requirements.txt
```

執行完整流程：

```powershell
python run_angle_analysis.py
```

也可以逐步執行：

```powershell
python scripts\01_check_data.py
python scripts\02_track_angle.py
python scripts\03_analyze_vmd_modes.py
```

第三步會執行三種VMD，所需時間較長。

## 主要輸出

`results/angle_tracking/`：

- `01_angle_component.png`：原始波長與角度低頻成分
- `02_estimated_angle_timeline.png`：估計角度時間序列
- `03_angle_transition_detection.png`：角度變化速度與事件區段
- `angle_transitions.csv`：切換開始、結束、持續時間及方向
- `angle_state_summary.csv`：各角度及移動狀態統計

`results/mode_analysis/`：

- 每種方法的模態頻率、能量及切換響應
- `mode_frequency_comparison.csv`：跨方法頻率比較
- `method_summary.csv`：三種方法品質摘要
- `angle_position_mode_comparison.png`：三種方法之角度位置模態比較
- `cross_method_mode_comparison.png`：頻率與角度移動敏感度比較
- `ANALYSIS_SUMMARY.md`：目前可下結論與限制的中文摘要

## 結果解釋原則

- 不以IMF編號直接跨方法配對，應以峰值頻率配對。
- 移動期間能量增加的模態是「角度移動響應候選」，不直接稱為故障。
- 只有在相同角度與相同移動條件下偏離正常模態，才可進一步定義異常。
- 目前只有10°與43°兩個校正點，不能宣稱可精確估計任意角度。
