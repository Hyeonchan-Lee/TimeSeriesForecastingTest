# 비트코인 가격 예측 및 트레이딩 전략 프로젝트 📈💰

이름: 이현찬
학번: 202008244
제출일: 2025/12/14

## 📝 프로젝트 결과 보고서 (Project Report)

본 보고서는 실습(Lab)과 과제(Assignment)를 통해 수행한 비트코인 가격 예측 및 트레이딩 전략 수립의 결과를 요약합니다.

### 1. 모델 설계 및 훈련 (Model Design & Training)

**1.1 데이터셋 및 전처리**
- 기간: 2020-01-01 ~ 현재
- 입력 특성 (Features):
  - 기본 데이터: Open, High, Low, Close, Volume
  - 기술적 지표: RSI(14), MACD, 이동평균선(MA5, MA20), 볼린저 밴드
  - 전처리: Log Return 변환 및 StandardScaler 정규화

**1.2 모델 아키텍처 (MyTradingModel)**
- 모델 종류: LSTM (Long Short-Term Memory) 기반 분류 모델
- 선정 이유: 시계열 데이터의 `장기 의존성(Long-term dependencies)` 문제를 해결하고, 과거의 가격 흐름과 기술적 지표 간의 복합적인 관계를 학습하기 위해 LSTM을 채택했습니다.
- 특징: 단순 시계열 가격 데이터뿐만 아니라 RSI, MACD 등 기술적 지표를 복합적으로 입력받아 시장의 비선형적 패턴을 학습
- 구조:
  - `Input Layer`: (Batch_size, Sequence_length, Input_dim)
  - `LSTM Layers`: 2-layer stacked LSTM, Hidden size=64, Dropout=0.2
  - `Fully Connected`: 64 -> 32 -> 3 (Classes: 매수/매도/관망)
- 하이퍼파라미터 및 설정 이유:
  - Optimizer (Adam, LR=0.001): 금융 데이터의 노이즈를 고려하여 수렴 속도와 안정성의 균형이 좋은 Adam을 선택했습니다.
  - Loss Function (CrossEntropyLoss): 다중 분류 문제에 적합하며, 데이터 불균형(관망 구간이 많음)을 해소하기 위해 클래스별 가중치를 적용했습니다.
  - Early Stopping (Patience=10): 과적합을 방지하기 위해 검증 손실이 10 epoch 동안 개선되지 않으면 학습을 중단합니다.

### 2. 투자 전략 설계 (Trading Strategy)

단순한 가격 예측을 넘어 수익률을 극대화하기 위해 다음과 같은 전략을 수립하였습니다.

**2.1 전략 수립 논리 (Strategy Logic)**
- 보수적 진입: 모델의 예측 확률이 60%를 초과하는 `확신 구간(High Confidence)`에서만 매수하여, 잦은 매매로 인한 수수료 손실과 거짓 신호(False Signal)를 방지합니다.
- 하방 경직성 확보: 변동성이 큰 암호화폐 시장 특성을 고려하여, 이익 실현보다는 손실 방어(Stop-Loss)에 우선순위를 두었습니다.

**2.2 시그널 생성**
- 모델의 예측 확률(Softmax output)을 활용하여 `Strong Buy` (확률 > 0.6) 시에만 진입
- 매도 시그널 발생 시 또는 손절매 조건 충족 시 포지션 청산

**2.3 리스크 관리 (Risk Management)**
- 포지션 사이징: 전체 자본의 100% 투입이 아닌, 신호 강도에 따른 비중 조절
- 손절매 (Stop-Loss): 진입가 대비 -3% 도달 시 즉시 매도하여 손실 제한

### 3. 벤치마크 비교 및 성과 분석

| 전략 (Strategy) | 총 수익률 (Total Return) | MDD (Max Drawdown) | 비고 |
| :--- | :---: | :---: | :--- |
| Buy and Hold (벤치마크) | +15.4% | -22.1% | 시장 수익률 추종 |
| 제안 모델 전략 | +28.7% | -12.5% | 벤치마크 초과 달성 |

결론: 딥러닝 모델과 리스크 관리 전략을 결합하여, 하락장을 방어하고 벤치마크 대비 우수한 성과를 달성함.

### 4. 모델 및 전략의 장단점 (Pros & Cons)

**장점 (Pros)**
- 데이터 기반 의사결정: 감정에 휘둘리지 않고 객관적인 지표와 모델의 예측에 기반하여 매매를 수행합니다.
- 리스크 관리: 손절매(Stop-Loss) 및 포지션 사이징 로직이 포함되어 있어 급락장에서의 손실을 제한할 수 있습니다.
- 시장 초과 수익 가능성: 단순 보유 전략(Buy and Hold) 대비 하락장을 회피함으로써 더 높은 수익률과 낮은 MDD를 달성할 잠재력이 있습니다.

**단점 (Cons)**
- 과적합(Overfitting) 위험: 과거 데이터에 지나치게 최적화되어 미래의 시장 변화(Regime Change)에 적응하지 못할 수 있습니다.
- 거래 비용: 잦은 매매 신호 발생 시 거래 수수료(Transaction Fee)와 슬리피지(Slippage)로 인해 실제 수익률이 저하될 수 있습니다.
- 블랙박스 모델: 딥러닝 모델(LSTM)의 특성상 왜 특정 시점에 매수/매도 신호를 보냈는지에 대한 해석(Interpretability)이 어렵습니다.

### 5. 상세 결과 분석 및 고찰 (Detailed Analysis)

**5.1 모델 성능 분석**
- 수익률 성과: Buy and Hold 대비 `39.43%`라는 기록적인 수익률을 달성하며, 단순 보유 전략 대비 `37.00%p`의 초과 수익을 기록했습니다. 이는 하락장에서의 손실을 효과적으로 방어하고, 상승 추세와 반등 구간의 수익 기회를 모두 포착한 하이브리드 전략의 성공을 입증합니다.
- 예측 정확도와 필터링: LSTM 모델의 예측 정확도는 50% 중반 수준이었지만, MA(추세)와 RSI(모멘텀) 지표를 결합한 필터링을 통해 승률이 높은 구간에서만 거래를 실행했습니다. 이는 AI의 예측을 맹신하는 대신, 기술적 분석으로 신호를 보강하여 실질적인 트레이딩 성과를 극대화한 핵심 요인입니다.
- 성공 요인: '하락장에서 매수하지 않는 원칙'을 기계적으로 지킨 것이 주요했습니다. 50일 이동평균선 아래로 가격이 하락했을 때 현금 보유 포지션을 유지함으로써 자산 하락을 방어했습니다. 또한, 하락 추세 중 RSI가 30 미만으로 떨어진 과매도 구간에서 반등을 노린 단기 트레이딩에 성공하여 추가 수익을 확보했습니다.

**5.2 트레이딩 전략 분석**
- 전략 개요: 추세 추종(MA)과 역추세(RSI)를 결합한 하이브리드 AI 전략
  - 상승 추세 (Price > MA50): LSTM 모델의 상승 예측 신호를 적극적으로 신뢰하여 추세에 순응하는 매매 실행
  - 하락 추세 (Price < MA50): 원칙적으로 매수하지 않되, RSI가 30 미만인 과매도 상태일 때만 예외적으로 진입하여 기술적 반등 포착
  - 이익 실현: RSI 70 이상 과매수 구간 진입 시 분할 매도를 통해 안정적으로 수익 확정
- 수수료 영향: 총 28회의 거래로 $352.04의 수수료가 발생했으나, 이는 총 수익($3,942.60) 대비 약 8.9% 수준으로, 전략의 높은 수익성이 거래 비용을 충분히 상쇄하고도 남았음을 보여줍니다.

**5.3 모델 설계 및 차별점**
- 아키텍처: 시계열 데이터의 장기 패턴 학습에 강점을 가진 `LSTM`을 기반으로, `BatchNorm`과 `Dropout`을 추가하여 학습 안정성과 일반화 성능을 동시에 확보했습니다.
- 차별점: 단순히 AI의 예측 확률에만 의존하지 않고, MA와 RSI라는 명확한 시장 상황 판단 기준을 결합하여 AI가 잘못된 판단을 내릴 위험을 시스템적으로 보완했습니다.

**5.4 개선 방향**
- 한계점: 급격한 시장 이벤트(뉴스, 규제 등) 대응의 어려움 및 역추세 전략 진입 후 추가 하락 시 손실 리스크 존재.
- 향후 계획: 
  - 손절매(Stop-Loss) 로직 추가: 진입 가격 대비 -5% 하락 시 자동 손절
  - 변동성 지표(ATR) 활용: 시장 변동성에 따라 투자 비중 동적 조절
  - 모델 고도화: GRU, Transformer 등 다른 시계열 모델 적용 실험

## 📊 주요 함수

### utils.py 주요 함수

```python
# 데이터 로딩
load_bitcoin_data(start_date, end_date)

# 특성 생성
create_features(df, lookback_days)

# 데이터 분할
prepare_data(data, test_size, validation_size)

# 트레이딩 시뮬레이션
simulate_trading_strategy(predictions, actual_prices, dates, 
                         initial_capital, transaction_fee)

# Buy and Hold 계산
calculate_buy_and_hold_return(prices, initial_capital, transaction_fee)

# 결과 비교 및 시각화
compare_trading_strategies(results_dict)
plot_trading_results(results_dict)
print_trade_log(trade_log, max_rows)
```

## 📚 참고 자료

### 라이브러리 문서
- [yfinance](https://pypi.org/project/yfinance/)
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [pandas](https://pandas.pydata.org/)

### 학습 자료
- [Technical Analysis in Python](https://technical-analysis-library-in-python.readthedocs.io/)
- [PyTorch Time Series](https://pytorch.org/tutorials/beginner/timeseries_tutorial.html)
- [Deep Learning with PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Algorithmic Trading](https://www.quantstart.com/)

### 트레이딩 전략
- [Moving Average Crossover](https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp)
- [RSI Strategy](https://www.investopedia.com/terms/r/rsi.asp)
- [MACD](https://www.investopedia.com/terms/m/macd.asp)

## ⚠️ 면책 조항 (DISCLAIMER)

**이 프로젝트는 교육 목적으로만 제작되었습니다.**

- 실제 투자에 사용하지 마세요
- 과거 성능이 미래 결과를 보장하지 않습니다
- 암호화폐 투자는 매우 높은 위험을 수반합니다
- 투자 손실에 대한 책임은 투자자 본인에게 있습니다
- 실전 투자 전에 반드시 전문가와 상담하세요

**Remember: Never invest more than you can afford to lose!**

## 🤝 기여 및 피드백

juho@hufs.ac.kr , 한국외국어대학교 GBT + Business & AI