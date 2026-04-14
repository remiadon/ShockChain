# ShockChain: Technical Specification

## Event-Driven Macro Risk Simulation for Portfolio Hedging

**Version**: 0.1 (MVP)  
**Date**: April 2026

---

## 1. Problem statement

Risk managers responsible for portfolio hedging need to rapidly assess the multi-horizon market impact of exogenous events (geopolitical shocks, policy changes, natural disasters). Current approaches rely on either manual scenario construction, which is slow and subjective, or historical lookback, which is rigid and backward-looking. ShockChain provides a learned prior over event-to-market-impact mappings, structured as an editable causal chain that the risk manager can interrogate, adjust, and defend to stakeholders.

## 2. Product definition

**User**: Portfolio hedging desk risk manager.

**Workflow**: The user enters a natural language event description (e.g., "Trump invades Venezuela"). ShockChain returns predicted impacts on a set of macroeconomic indicators across three time horizons (1-day, 5-day, 15-day), presented as a chain where each horizon's predictions feed into the next. The user can override any intermediate prediction and observe how downstream impacts change, enabling structured scenario exploration.

**Key differentiator**: The chain is not a black box. Each link is transparent, editable, and produces a narrative the user can present to a CRO or investment committee: "VIX spikes next-day, which drives USD strength at 5d, which pressures EM equities at 15d. I adjusted the USD call because ECB meets that week."

---

## 3. Architecture overview

### 3.1 High-level architecture

The diagram below illustrates the full system architecture. The model is a three-stage chain where each stage produces predictions for all target indicators at a given horizon. The output of each stage feeds into the next, and the user can override any intermediate prediction before it propagates downstream.

```
                    ┌─────────────────┐
                    │  Headline (raw)  │
                    └────────┬────────┘
                             │
                 ┌───────────┼───────────┐
                 ▼                       ▼
        ┌────────────────┐     ┌─────────────────────┐
        │  FinBERT -> Q  │     │  LLM -> S (optional) │
        └───────┬────────┘     └──────────┬───────────┘
                │                         │
                └────────┬────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │     Concat [Q, (S), X]       │◄──── Market state X
          └──────────────┬───────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────┐
    │         STAGE 1: next-day (t+1)            │
    │                                            │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
    │  │ y1  │ │ y2  │ │ y3  │ │ y4  │ │ y5  │ │ -> ŷ₁
    │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │
    └──────────────────┬─────────────────────────┘
                       │
              ŷ₁ or ŷ₁* (user override) ◄── User adjusts
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │         STAGE 2: mid-term (t+5)            │
    │                                            │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
    │  │ y1  │ │ y2  │ │ y3  │ │ y4  │ │ y5  │ │ -> ŷ₂
    │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │
    └──────────────────┬─────────────────────────┘
                       │
              ŷ₂ or ŷ₂* (user override) ◄── User adjusts
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │        STAGE 3: long-term (t+15)           │
    │                                            │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
    │  │ y1  │ │ y2  │ │ y3  │ │ y4  │ │ y5  │ │ -> ŷ₃
    │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │
    └────────────────────────────────────────────┘

    Training: noise injection on intermediate ŷ to simulate
    user overrides and model prediction error.
```

### 3.2 Formal model definition

The model is a three-stage chain:

$$
\hat{y}_1 = F_1(X, Q, S)
$$

$$
\hat{y}_2 = F_2(X, Q, S, \hat{y}_1^*)
$$

$$
\hat{y}_3 = F_3(X, Q, S, \hat{y}_2^*)
$$

Where:

- $X \in \mathbb{R}^d$ is the market state vector (current regime features).
- $Q \in \mathbb{R}^e$ is the event embedding (FinBERT output).
- $S \in \mathbb{R}^k$ is the structured event meta-features (LLM-extracted). This component is optional and may be included or excluded depending on experimental results (see section 4.3).
- $\hat{y}_i^* \in \{-2, -1, 0, +1, +2\}^n$ is the (possibly user-overridden) prediction from stage $i$, where $n$ is the number of target indicators. At training time, $\hat{y}_i^* = y_i + \epsilon$ (true label with injected noise). At inference time, $\hat{y}_i^* = \hat{y}_i$ (model prediction) unless the user overrides specific components.
- $F_i$ are multitask shallow MLPs with shared trunk and per-indicator classification heads.

### 3.3 Target variables (y)

Each stage outputs a 5-class categorical prediction for each target indicator. The number and selection of indicators is subject to discussion and experimentation. The following five are proposed as an initial working set, chosen because they span equity, volatility, FX, commodity, and rates risk, and they are all liquid enough to hedge directly with derivatives:

| Indicator | Notation | Rationale |
|-----------|----------|-----------|
| S&P 500 return | $y^{SPX}$ | Core equity risk; most hedging books reference this index |
| VIX change (abs. pts) | $y^{VIX}$ | Maps directly to options hedge cost; primary shock indicator |
| DXY change | $y^{DXY}$ | FX exposure channel; USD strength drives EM repricing |
| WTI crude return | $y^{WTI}$ | Commodity exposure; geopolitical events hit oil first |
| US 10Y yield change (bps) | $y^{10Y}$ | Rate risk and flight-to-quality signal |

Alternative or additional indicators to consider in later iterations include: IG/HY credit spreads, EM FX basket, sector-level equity indices, gold, and natural gas.

**Discretization scheme**: Each indicator's move is classified into five buckets based on trailing 60-day realized volatility ($\sigma_{60}$):

| Class | Label | Definition |
|-------|-------|------------|
| -2 | Strong negative | Move < $-2\sigma_{60}$ |
| -1 | Moderate negative | $-2\sigma_{60}$ ≤ Move < $-1\sigma_{60}$ |
| 0 | Neutral | $\lvert\text{Move}\rvert$ < $1\sigma_{60}$ |
| +1 | Moderate positive | $1\sigma_{60}$ ≤ Move < $2\sigma_{60}$ |
| +2 | Strong positive | Move ≥ $2\sigma_{60}$ |

Note: for VIX, the sign convention is inverted relative to "risk-on/risk-off" framing. A positive VIX move corresponds to a risk-off shock. The UI should present this clearly to the user.

### 3.4 Time horizons

| Stage | Horizon | Label | Measures |
|-------|---------|-------|----------|
| 1 | t+1 business day | Next-day | Immediate shock absorption |
| 2 | t+5 business days | Mid-term | Sentiment propagation, positioning adjustment |
| 3 | t+15 business days | Long-term | Structural repricing, policy response feedback |

---

## 4. Input specification

### 4.1 Market state vector X

The market state vector captures the current regime and the market's vulnerability to shocks at the time of the event. The exact feature set is subject to discussion and experimentation. The following is a proposed initial working set:

| Feature | Source | Rationale |
|---------|--------|-----------|
| VIX level | CBOE | Implied vol = market fear gauge |
| VIX percentile (1Y) | Derived | Contextualizes current VIX within regime |
| US 10Y yield | Treasury | Rate environment |
| 2Y-10Y spread | Derived | Yield curve slope = recession signal |
| IG credit spread (OAS) | ICE BofA | Corporate stress indicator |
| HY credit spread (OAS) | ICE BofA | High-yield stress = tail risk gauge |
| DXY level | ICE | Dollar strength regime |
| S&P 500 20d return | Derived | Recent equity momentum |
| S&P 500 20d realized vol | Derived | Recent turbulence |
| WTI crude price | NYMEX | Energy price regime |
| Gold price | COMEX | Safe-haven demand proxy |
| 1M cross-asset correlation (SPX, bonds, gold, oil) | Derived | Correlation regime; high = fragile |
| Put/call ratio (CBOE equity) | CBOE | Positioning / hedging demand proxy |

This yields approximately 13 to 15 features. All should be standardized (z-score using a trailing 252-day window) before model input. The guiding principle for feature selection is that $X$ should capture the market's current vulnerability to shocks, not just price levels.

### 4.2 Event embedding Q

**Model**: FinBERT (or similar financial domain sentence transformer).

**Process**: The raw headline string is tokenized and passed through the frozen (or lightly fine-tuned) encoder. The [CLS] token embedding is extracted as $Q \in \mathbb{R}^{768}$.

**Fine-tuning strategy for v0**: Freeze FinBERT entirely. The downstream MLP learns to map from a fixed embedding space. Fine-tuning is deferred to v1 when more training data is available, to avoid overfitting the encoder on a small event corpus.

### 4.3 Structured meta-features S (optional)

This component is an optional addition to the input vector, to be evaluated experimentally. The hypothesis is that explicit structured features provide interpretability and serve as a fallback signal if the FinBERT embedding is not discriminative enough on a small training set. If ablation experiments show that $S$ does not improve validation performance, it can be removed without affecting the rest of the architecture.

If included, $S$ is extracted by an LLM at data preparation time (offline, not at inference for training data; at inference time for user queries). Each headline is processed to produce:

| Feature | Type | Example values |
|---------|------|----------------|
| Event category | One-hot (6 classes) | Geopolitical, monetary policy, trade, regulatory, natural disaster, supply disruption |
| Primary asset class affected | One-hot (5 classes) | Equity, FX, commodity, rates, credit |
| Geographic scope | One-hot (4 classes) | US-domestic, developed markets, emerging markets, global |
| Escalation signal | Binary | Escalation (1) vs. de-escalation (0) |

Total: approximately 16 binary features after one-hot encoding.

---

## 5. Model specification (proposed)

The following architecture is a starting proposal. It is designed to be simple enough to train on a small dataset (under 10K samples) while expressive enough to capture non-linear interactions between event embeddings and market state. The exact layer sizes, activation functions, and regularization are subject to hyperparameter tuning.

### 5.1 Per-stage architecture

Each stage $F_i$ is a multitask MLP with shared trunk and independent heads:

```
Input: [Q (768) | S (16, optional) | X (15) | ŷ*_{i-1} (n x 5, one-hot)]

Shared trunk:
  Linear(input_dim, 256) -> BatchNorm -> ReLU -> Dropout(0.3)
  Linear(256, 128) -> BatchNorm -> ReLU -> Dropout(0.3)

Per-indicator head (x n):
  Linear(128, 64) -> ReLU -> Dropout(0.2)
  Linear(64, 5) -> Softmax
```

Output per head: probability distribution over 5 classes.

For stage 1, the input dimension is approximately 799 (or 783 without $S$). For stages 2 and 3, the previous stage's output adds $n \times 5$ dimensions (one-hot encoded predictions for each indicator).

### 5.2 Chain wiring

Stage 1 takes no previous predictions: input is $[Q \| S \| X]$.

Stages 2 and 3 additionally receive the previous stage's output, encoded as a one-hot vector per indicator (the predicted or overridden class). This is concatenated to the shared input before the trunk.

The use of one-hot encoding of the predicted class (rather than the raw softmax vector) is intentional. At inference time with user overrides, the user selects a discrete class, not a probability distribution. Training on one-hot representations ensures consistency between training-time and inference-time input distributions.

### 5.3 Noise injection during training

This section describes a training technique designed to solve two problems simultaneously: (1) preventing error compounding across the chain stages, and (2) making the model robust to user overrides at inference time.

**The core problem**: During training, if stages 2 and 3 always receive the true historical labels as intermediate inputs, they learn to expect perfect information. At inference time, they receive imperfect information (the model's own predictions, or user overrides that may differ from reality). This distribution mismatch causes performance to degrade downstream in the chain.

**The solution**: During training, we corrupt the true intermediate labels with controlled noise before feeding them to downstream stages. This teaches the model to make reasonable predictions even when its intermediate inputs are approximate.

**Mechanism**: For each training sample, the intermediate labels $y_i$ fed to stage $i+1$ are replaced with a noisy version $\hat{y}_i^*$ according to the following procedure:

$$
\hat{y}_{i,\text{train}}^* = \begin{cases} y_i & \text{with probability } p_{\text{true}} \\ \text{perturb}(y_i) & \text{with probability } 1 - p_{\text{true}} \end{cases}
$$

Where the perturbation function flips the class label to an adjacent class with high probability, or to a non-adjacent class with lower probability.

**Worked example**: Suppose the true 1-day labels for a "China imposes 40% tariff on Australian lithium" event are:

| Indicator | True class ($y_1$) | Meaning |
|-----------|---------------------|---------|
| SPX | -1 | Moderate negative |
| VIX | +2 | Strong positive (fear spike) |
| DXY | +1 | Moderate positive (USD strength) |
| WTI | -1 | Moderate negative |
| US10Y | -1 | Moderate negative (flight to quality) |

When constructing the input for stage 2 (the 5-day model) during training, we process each indicator independently with $p_{\text{true}} = 0.7$:

**SPX** ($y = -1$): Random draw = 0.45. Since 0.45 < 0.7, we keep the true label. Result: $\hat{y}^* = -1$.

**VIX** ($y = +2$): Random draw = 0.82. Since 0.82 > 0.7, we perturb. We then draw again to decide the perturbation type. With probability 0.7 among noise cases, we flip to an adjacent class: $\hat{y}^* = +1$ (one step toward neutral). With probability 0.3, we would flip further, e.g., to 0 or -1. In this example, the adjacent flip wins. Result: $\hat{y}^* = +1$.

**DXY** ($y = +1$): Random draw = 0.55. Since 0.55 < 0.7, we keep. Result: $\hat{y}^* = +1$.

**WTI** ($y = -1$): Random draw = 0.91. Since 0.91 > 0.7, we perturb. Adjacent flip gives $\hat{y}^* = 0$ (one step toward neutral). Result: $\hat{y}^* = 0$.

**US10Y** ($y = -1$): Random draw = 0.33. Since 0.33 < 0.7, we keep. Result: $\hat{y}^* = -1$.

The resulting noisy vector fed to stage 2 is $[-1, +1, +1, 0, -1]$ instead of the true $[-1, +2, +1, -1, -1]$. Stage 2 must now learn to predict 5-day outcomes given that VIX and WTI intermediate values are "wrong" (both understated by one class). This is precisely the situation the model will face at inference time in two scenarios: (a) the 1-day model's own predictions are slightly off, which is the normal case, or (b) a user overrides VIX downward because they believe the market will absorb the shock faster than the model expects.

**Why this works**: Over many training examples, the model sees intermediate inputs that are correct 70% of the time and slightly off 30% of the time. It learns to use the intermediate signal when it is informative while remaining robust to noise. The downstream stage effectively learns a mapping from "approximate intermediate state" to "outcome," rather than from "exact intermediate state" to "outcome." This is analogous to scheduled sampling in sequence-to-sequence models and to dropout's role in preventing co-adaptation of neurons.

**Recommended starting parameters** (to be tuned on validation set):

| Parameter | Value | Description |
|-----------|-------|-------------|
| $p_{\text{true}}$ | 0.7 | Probability of keeping the true label |
| Adjacent flip probability | 0.7 of noise cases | Perturb by one class step |
| Non-adjacent flip probability | 0.3 of noise cases | Perturb by two or more class steps |

If $p_{\text{true}}$ is set too high (e.g., 0.95), the model becomes brittle to imperfect inputs. If set too low (e.g., 0.3), the intermediate signal becomes too noisy to learn from. The 0.7 starting point is a reasonable prior, but this should be treated as a key hyperparameter during validation.

### 5.4 Loss function

Weighted cross-entropy per head, with higher weight on extreme classes to counteract class imbalance:

$$
\mathcal{L}_{\text{stage}} = \sum_{j=1}^{n} \lambda_j \cdot \text{CE}(p_j, y_j, w)
$$

Where:

- $j$ indexes the $n$ target indicators.
- $\lambda_j$ are per-indicator task weights (initially uniform; can be tuned).
- $w = [3.0, 1.5, 1.0, 1.5, 3.0]$ are per-class weights for classes $[-2, -1, 0, +1, +2]$.

Total loss is the sum across all three stages: $\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2 + \mathcal{L}_3$.

All three stages are trained jointly end-to-end in a single forward pass.

---

## 6. Data pipeline

### 6.1 Source

**Primary**: Benzinga news headlines via Massive.com. Historical coverage spanning multiple years of financial news.

### 6.2 Filtering

A two-stage filtering pipeline to retain only exogenous event headlines:

**Stage 1, rule-based pre-filter**: Remove headlines matching patterns for analyst opinions, price target changes, earnings reports, and market commentary (e.g., headlines containing "price target", "upgrades", "downgrades", "earnings beat/miss", "trading at").

**Stage 2, LLM classifier**: A few-shot prompted LLM (e.g., Claude Haiku for cost efficiency) classifies remaining headlines as:

- **Exogenous event** (keep): describes a real-world event such as a policy decision, geopolitical action, natural disaster, regulatory change, or supply disruption.
- **Market commentary** (discard): describes market activity, analyst opinion, fund flows, sentiment, or price action.

Target: approximately 50 hand-labeled examples for few-shot prompt. Expected yield: 5,000 to 20,000 usable event headlines from the full Benzinga corpus.

### 6.3 Timestamp alignment

This step is critical for avoiding lookahead bias.

| Headline publication time | Market reaction window starts |
|---------------------------|-------------------------------|
| Before market open (pre 9:30 ET) | Same-day open |
| During market hours (9:30-16:00 ET) | Next bar after publication (conservative: next-day open) |
| After market close (post 16:00 ET) | Next business day open |
| Weekend / holiday | Next business day open |

For v0, the conservative approach is used: all headlines published on date $t$ are aligned with market data starting at date $t+1$ open. This sacrifices some signal from intraday headlines but eliminates lookahead risk entirely.

### 6.4 Label construction

For each retained headline at time $t$:

1. Retrieve the market state vector $X_t$ (close-of-business values for all features on the day before or of the headline, depending on timing).
2. Compute forward returns/changes for each indicator at horizons $t+1$, $t+5$, $t+15$ business days.
3. Compute trailing 60-day realized volatility for each indicator as of date $t$.
4. Discretize each forward move into the 5-class scheme using the trailing vol as the normalization factor.

### 6.5 Structured feature extraction (if applicable)

If structured meta-features $S$ are included (see section 4.3), run the LLM extraction prompt on each retained headline to produce the structured features. This is done offline at data preparation time. The extraction prompt should be deterministic (temperature=0) and validated on a small hand-labeled subset.

### 6.6 Dataset summary

| Component | Specification |
|-----------|--------------|
| Raw source | Benzinga headlines via Massive |
| Post-filter volume | Approximately 5,000 to 20,000 events (estimate) |
| Features per sample | Q (768) + S (16, optional) + X (15) |
| Labels per sample | $n$ indicators x 3 horizons x 5 classes |
| Train set | Events before 2022-01-01 |
| Validation set | Events 2022-01-01 to 2023-12-31 |
| Test set | Events 2024-01-01 to 2025-12-31 |

---

## 7. Training procedure (proposed)

The following training procedure is a starting proposal and should be refined based on early experimental results.

### 7.1 Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 with cosine annealing to 1e-5 |
| Weight decay | 1e-4 |
| Batch size | 64 (or largest power of 2 that fits in memory) |
| Max epochs | 200 |
| Early stopping | Patience 20 epochs on validation macro F1 |
| Gradient clipping | Max norm 1.0 |

### 7.2 Training loop (pseudocode)

```
for each batch (headline, X, S, y_1d, y_5d, y_15d):
    Q = FinBERT(headline)                          # frozen encoder
    input_1 = concat(Q, S, X)
    ŷ_1 = F_1(input_1)                             # stage 1 forward

    ŷ_1_noisy = noise_inject(y_1d, p_true=0.7)     # corrupt TRUE labels
    input_2 = concat(Q, S, X, one_hot(ŷ_1_noisy))
    ŷ_2 = F_2(input_2)                             # stage 2 forward

    ŷ_2_noisy = noise_inject(y_5d, p_true=0.7)
    input_3 = concat(Q, S, X, one_hot(ŷ_2_noisy))
    ŷ_3 = F_3(input_3)                             # stage 3 forward

    loss = CE(ŷ_1, y_1d) + CE(ŷ_2, y_5d) + CE(ŷ_3, y_15d)
    loss.backward()
    optimizer.step()
```

During training, stages 2 and 3 receive noisy versions of the **true** labels, not the model's own predictions. This is analogous to scheduled sampling in sequence models and prevents the train/inference distribution mismatch that causes error compounding.

### 7.3 Inference loop (pseudocode)

```
Q = FinBERT(headline)
S = LLM_extract(headline)       # if structured features are used
X = get_market_state(current_date)

ŷ_1 = F_1(concat(Q, S, X))
display ŷ_1 to user -> user optionally overrides -> ŷ_1*

ŷ_2 = F_2(concat(Q, S, X, one_hot(ŷ_1*)))
display ŷ_2 to user -> user optionally overrides -> ŷ_2*

ŷ_3 = F_3(concat(Q, S, X, one_hot(ŷ_2*)))
display ŷ_3 to user
```

---

## 8. Evaluation protocol

### 8.1 Data split

Strictly temporal: train < 2022, validation 2022-2023, test 2024-2025. No shuffling, no cross-validation across time boundaries.

### 8.2 Metrics (per head, per horizon)

| Metric | What it measures | Why it matters |
|--------|------------------|----------------|
| **Accuracy** | Overall correctness | Baseline sanity check |
| **Macro F1** | Per-class F1 averaged equally | Primary metric; penalizes model that ignores rare extreme classes |
| **Directional accuracy** | Correct sign prediction (collapse to neg/neutral/pos) | For hedging, direction matters more than magnitude |
| **Calibration (ECE)** | Expected calibration error of softmax probabilities | Critical for risk tool; uncalibrated confidence destroys trust |

### 8.3 Aggregate metrics

| Metric | Definition |
|--------|------------|
| Average macro F1 | Mean of macro F1 across all heads (reported per horizon) |
| Worst-head F1 | Minimum macro F1 across heads (per horizon) |
| Horizon degradation | Ratio of 15d avg macro F1 to 1d avg macro F1; measures chain error compounding |

### 8.4 Baselines

| Baseline | Description | What it tests |
|----------|-------------|---------------|
| **Always-neutral** | Predicts class 0 for all indicators | Floor; model must beat naive majority class |
| **Sentiment-only** | FinBERT sentiment score to 3-class directional prediction (no market state) | Tests whether model adds value beyond "bad news = market down" |
| **Market-state-only** | MLP on X alone (no headline) | Tests whether event information adds value beyond current regime |
| **Independent heads** | Same architecture but each horizon trained independently (no chain) | Tests whether the chain structure helps or hurts |

**Success criterion**: ShockChain must beat all four baselines on average macro F1 for at least the 1d and 5d horizons. If it fails to beat market-state-only, the event embedding is not contributing. If it fails to beat sentiment-only, the architecture is not learning beyond shallow sentiment.

---

## 9. UX specification (web app)

### 9.1 Core interaction flow

1. **Input panel**: Text field for headline entry. Optionally, a date picker to set the "as-of" date for market state $X$ (defaults to today).

2. **Chain visualization**: Three connected panels (1d, 5d, 15d), each displaying indicator predictions as color-coded tiles (strong negative = red, neutral = gray, strong positive = green) with softmax confidence shown as a percentage or bar.

3. **Override mechanism**: The user clicks any indicator tile at any stage to cycle through classes or select from a dropdown. Overriding a tile at stage $k$ triggers re-inference of all stages > $k$ with the override applied. A visual cue (e.g., amber border) distinguishes user-overridden tiles from model predictions.

4. **Scenario comparison**: The user can save multiple scenario paths (e.g., "base case", "VIX overreaction scenario", "oil-focused scenario") and compare them side by side.

### 9.2 Technical stack (proposed)

The following stack is a starting proposal for the MVP and may be revised based on team expertise and infrastructure constraints.

| Component | Technology |
|-----------|-----------|
| Frontend | React (web app) |
| Backend | FastAPI (Python) |
| Model serving | PyTorch model loaded in FastAPI process (v0); migrate to TorchServe or ONNX Runtime for v1 |
| Embedding | FinBERT via HuggingFace Transformers |
| Structured extraction | Claude Haiku API call at inference time (if S is included) |
| Market data | Yahoo Finance API or equivalent for real-time X |

### 9.3 Latency budget (estimated)

The following latency targets are rough estimates based on typical inference times for models of this size. Actual performance will depend on hardware and implementation.

| Step | Estimated target |
|------|------------------|
| FinBERT embedding | < 200ms |
| LLM structured extraction (if included) | < 1.5s |
| Per-stage MLP inference | < 10ms |
| Full chain (3 stages) | < 50ms |
| Total end-to-end | < 2s |
| Override re-inference (stages 2+3 only) | < 30ms |

---

## 10. Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Class imbalance (neutral dominates) | High | Weighted CE loss, oversampling extreme events, macro F1 as primary metric |
| Error compounding across chain | High | Noise injection during training, independent-heads baseline comparison |
| Small training set (< 10K samples) | High | Shallow architecture, frozen encoder, structured features as fallback signal, aggressive regularization |
| Lookahead bias in labels | Critical | Conservative timestamp alignment (all headlines map to next-day reaction) |
| FinBERT not discriminative for event types | Medium | Structured meta-features S provide complementary signal; upgrade to fine-tuned encoder in v1 |
| Multi-event contamination in labels | Medium | Accept noisy labels; market state X implicitly captures concurrent environment |
| User trust and adoption | Medium | Editable chain provides transparency; calibration metrics ensure reliable confidence |

---


## 11. Development roadmap (proposed)
 
The following roadmap is calibrated for a side project at approximately 8 to 15 hours per week, with the understanding that some weeks will see zero progress due to other commitments. Durations are expressed as calendar weeks to account for this irregular cadence. The total estimated timeline for the MVP is 5 to 7 months.
 
The roadmap is deliberately structured to produce a visible, testable output at the end of each phase. This is important for side projects: long stretches of invisible infrastructure work kill motivation. Every phase ends with something you can look at, show someone, or evaluate.
 
### Phase 0: Project setup and data acquisition (weeks 1 to 3, approximately 3 weeks)
 
**Goal**: Have raw data in hand and a working development environment. This phase is deliberately short and concrete to build early momentum.
 
| Task | Estimated effort | Details |
|------|------------------|---------|
| Set up project repository and environment | 3 to 4 hours | Python environment, dependencies (PyTorch, Transformers, pandas), project structure, version control |
| Acquire Benzinga data from Massive | 4 to 6 hours | Account setup, data download, initial exploration of schema, volume, and time coverage. Understand what fields are available (timestamp, headline text, tickers, categories) |
| Exploratory data analysis | 4 to 6 hours | Headline volume over time, category distribution, timestamp granularity, headline length distribution. The goal is to understand what you are working with before building anything |
| Acquire market data | 3 to 4 hours | Download historical daily data for the target indicators (SPX, VIX, DXY, WTI, US10Y) and market state features. Yahoo Finance or FRED. Store as clean CSVs or parquet |
 
**Deliverable**: A notebook showing headline volume over time, sample headlines, and aligned market data for the target indicators. You should be able to see the raw materials of the project.
 
**Go/no-go checkpoint**: Is the Benzinga data sufficient in volume and time coverage? If fewer than 3,000 headlines survive initial inspection of the date range, consider supplementary sources before proceeding.
 
### Phase 1: Data pipeline and label construction (weeks 4 to 9, approximately 6 weeks)
 
**Goal**: A clean, labeled dataset ready for model training. This is the longest and most unglamorous phase. It is also where most side projects stall, so break it into small, testable subtasks.
 
| Task | Estimated effort | Details |
|------|------------------|---------|
| Rule-based headline pre-filter | 4 to 6 hours | Implement keyword and pattern filters to remove analyst opinions, price targets, earnings commentary. Manually review 200 random filtered/retained headlines to calibrate |
| LLM headline classifier | 8 to 12 hours | Hand-label approximately 50 headlines as exogenous event vs. market commentary. Build few-shot prompt for Claude Haiku. Run on full corpus. Manually validate a random sample of 200 classified headlines. Iterate on prompt until precision is acceptable (target > 85% on the "exogenous event" class) |
| Timestamp alignment logic | 4 to 6 hours | Implement the conservative alignment rule (all headlines on date t map to t+1 open). Handle weekends, holidays (US market calendar), and edge cases. This is fiddly but critical; get it right once |
| Market state feature engineering | 6 to 8 hours | Compute all derived features (VIX percentile, yield curve slope, realized vol, cross-asset correlation, put/call ratio). Align to headline dates. Handle missing data and backfill rules |
| Label construction and discretization | 4 to 6 hours | Compute forward returns at 1d, 5d, 15d for each indicator. Compute trailing 60-day realized vol. Discretize into 5 classes. Validate class distribution; expect 55 to 65% neutral, 15 to 20% moderate, 5 to 10% extreme |
| Structured feature extraction (if included) | 6 to 8 hours | Build LLM extraction prompt for event category, asset class, geographic scope, escalation signal. Run on full corpus. Validate on a subset |
| Final dataset assembly | 3 to 4 hours | Join all components into a single training-ready dataset. Train/val/test split by date. Save as parquet or HDF5. Write a data card documenting all decisions |
 
**Deliverable**: A clean dataset file with columns for headline, embedding (or raw text to embed later), market state features, structured features (if applicable), and discretized labels for all indicators at all horizons. A data card documenting filtering decisions, alignment rules, and class distributions.
 
**Go/no-go checkpoint**: How many usable training samples do you have (events before 2022)? If fewer than 2,000, the shallow MLP may not have enough signal. Consider simplifying to fewer indicators or coarser discretization (3 classes instead of 5).
 
### Phase 2: Embedding pipeline and baseline models (weeks 10 to 13, approximately 4 weeks)
 
**Goal**: Establish performance baselines and validate that the event embedding carries signal. This phase is where you find out whether the project is viable.
 
| Task | Estimated effort | Details |
|------|------------------|---------|
| FinBERT embedding pipeline | 4 to 6 hours | Load FinBERT from HuggingFace. Batch-embed all headlines. Store embeddings alongside the dataset. This is a one-time compute step |
| Always-neutral baseline | 1 to 2 hours | Compute accuracy and macro F1 for a model that always predicts class 0. This is your absolute floor |
| Sentiment-only baseline | 4 to 6 hours | Extract FinBERT sentiment scores. Train a simple logistic regression or small MLP mapping sentiment to 3-class directional prediction. Evaluate |
| Market-state-only baseline | 4 to 6 hours | Train a small MLP on X alone (no headline information). This tests whether the market regime already predicts what happens next, regardless of the event |
| Embedding quality analysis | 3 to 4 hours | Visualize embedding space (t-SNE or UMAP) colored by event category and by market reaction. Check whether similar events cluster together. This tells you whether FinBERT representations are useful for your task before investing in the full model |
 
**Deliverable**: A baseline comparison table showing accuracy, macro F1, and directional accuracy for all three baselines across all indicators and horizons. An embedding visualization showing whether event types cluster meaningfully.
 
**Go/no-go checkpoint**: Does the sentiment-only baseline beat always-neutral by a meaningful margin (at least 3 to 5 points of macro F1)? If not, the headline signal may be too weak for any downstream model to exploit, and you should investigate whether the labeling or filtering pipeline has issues before proceeding.
 
### Phase 3: Core model development (weeks 14 to 20, approximately 7 weeks)
 
**Goal**: Train the full ShockChain model (3-stage chain with noise injection) and evaluate against baselines. This is the core ML work.
 
| Task | Estimated effort | Details |
|------|------------------|---------|
| Stage 1 model (next-day) | 8 to 10 hours | Implement the multitask MLP with shared trunk and 5 heads. Train on 1-day prediction task only (no chain yet). This validates the architecture before adding chain complexity |
| Noise injection module | 3 to 4 hours | Implement the label corruption logic as a reusable module. Unit test with known inputs to verify the perturbation distribution matches the specification |
| Chain wiring (stages 2 and 3) | 6 to 8 hours | Extend the architecture to 3 stages with chain connections. Implement the one-hot encoding of intermediate predictions. Implement noise injection into the training loop |
| Hyperparameter tuning | 8 to 12 hours | Tune learning rate, dropout, noise injection rate ($p_{\text{true}}$), class weights. Use validation set macro F1 as the selection criterion. This will take multiple training runs; expect 10 to 20 experiments |
| Independent heads baseline | 4 to 6 hours | Train the same architecture but without chain connections (each horizon sees only Q, S, X). This directly tests whether the chain helps |
| Full evaluation | 4 to 6 hours | Evaluate ShockChain and all baselines on the test set. Compute all metrics from section 8. Generate calibration curves. Write up results |
 
**Deliverable**: A trained model with full evaluation results. A comparison table showing ShockChain vs. all four baselines. Calibration curves for each head. A clear answer to the question: does the chain add value?
 
**Go/no-go checkpoint**: Does ShockChain beat all baselines on at least the 1d horizon? If the chain underperforms independent heads, the noise injection parameters need tuning, or the chain architecture may not be justified for this dataset size. If ShockChain does not beat market-state-only, the event embedding is not contributing, and you should revisit the embedding strategy before building a UI.
 
### Phase 4: Minimal web UI (weeks 21 to 26, approximately 6 weeks)
 
**Goal**: A functional web application that demonstrates the full ShockChain workflow, from headline input to chain visualization to user override.
 
| Task | Estimated effort | Details |
|------|------------------|---------|
| Backend API (FastAPI) | 8 to 10 hours | Endpoints for headline submission (returns full chain prediction), override submission (returns re-inferred downstream stages), and current market state retrieval. Load the trained PyTorch model into the FastAPI process |
| FinBERT serving | 3 to 4 hours | Integrate FinBERT inference into the backend. Handle tokenization and embedding extraction. Consider caching embeddings for repeated queries |
| LLM extraction integration (if applicable) | 3 to 4 hours | Call Claude Haiku API for structured feature extraction at inference time. Handle API errors and latency gracefully |
| Frontend: chain visualization | 10 to 14 hours | React app with three connected panels (1d, 5d, 15d). Each panel shows indicator tiles with color coding and confidence. Animated arrows between panels to convey the chain flow |
| Frontend: override mechanism | 6 to 8 hours | Clickable indicator tiles that open a class selector. Override triggers re-inference of downstream stages. Visual distinction (amber border) for overridden tiles. This is the core UX differentiator; spend time making it feel responsive |
| Frontend: input and polish | 4 to 6 hours | Headline text input, optional date picker, loading states, error handling. Basic responsive layout for desktop |
| Integration testing | 4 to 6 hours | End-to-end testing of the full workflow. Verify that overrides propagate correctly. Test with a range of headline types |
 
**Deliverable**: A working web application where you can type a headline, see chain predictions, override intermediate values, and observe downstream changes. This is the MVP.
 
### Post-MVP: v1 improvements (no fixed timeline)
 
The following improvements are sequenced by expected impact. Work on them as motivation and time allow, in roughly this order.
 
| Improvement | Estimated effort | Priority rationale |
|-------------|------------------|---------------------|
| Scenario saving and comparison | 8 to 12 hours | Lets the user explore multiple "what if" paths side by side. High UX value, moderate engineering effort |
| Confidence thresholding and "low confidence" warnings | 3 to 4 hours | Small change with significant trust impact. Flag predictions where the model is uncertain |
| Historical event search ("show me similar past events") | 10 to 15 hours | Build a nearest-neighbor index on the FinBERT embedding space. When the user submits a headline, show the 5 most similar historical events and what actually happened. High value for user trust |
| Fine-tune FinBERT on the prediction task | 8 to 12 hours | Unfreeze the encoder and fine-tune end-to-end. Only worth attempting once you have validated that the frozen embedding carries signal and you have enough data |
| Expand indicator set | 6 to 10 hours per indicator | Add credit spreads, EM FX, sector ETFs. Each new indicator requires data sourcing, label construction, and a new head. Incremental effort |
| Continuous distributional output | 10 to 15 hours | Replace 5-class discretization with calibrated probability distributions over returns. More informative but harder to calibrate and evaluate |
| PDF/report export for CRO presentations | 6 to 10 hours | Generate a formatted scenario report that the risk manager can present. High value for adoption in a real workflow |
 
### Timeline summary
 
| Phase | Calendar weeks | Cumulative | Key output |
|-------|----------------|------------|------------|
| Phase 0: Setup and data acquisition | Weeks 1 to 3 | Month 1 | Raw data in hand, EDA notebook |
| Phase 1: Data pipeline and labels | Weeks 4 to 9 | Month 2 to 3 | Clean labeled dataset |
| Phase 2: Embeddings and baselines | Weeks 10 to 13 | Month 3 to 4 | Baseline performance table, go/no-go |
| Phase 3: Core model | Weeks 14 to 20 | Month 4 to 5 | Trained model, full evaluation |
| Phase 4: Web UI | Weeks 21 to 26 | Month 5 to 7 | Working MVP application |
| Total v0 | 26 weeks | 6 to 7 months | Complete MVP |
 
These estimates assume approximately 10 hours per week of effective work. Add 20 to 30% buffer for weeks with zero progress, debugging rabbit holes, and the inevitable data issues that only surface during model training. A realistic total is 7 to 8 months from start to a working MVP.
 
### Side project risk factors
 
There are several risks specific to the side-project format that are worth naming explicitly.
 
**Motivation cliff after Phase 1**: The data pipeline phase is long and produces no exciting output. Break it into the smallest possible subtasks and track progress visibly (e.g., a checklist). Consider building a quick "smoke test" model (logistic regression on raw sentiment) at the end of Phase 1 just to see numbers, even if they are bad.
 
**Scope creep in Phase 4**: The UI is where it becomes tempting to add "just one more feature." Define the MVP UI before starting Phase 4 and stick to it. The chain visualization with override is the only feature that matters for v0. Everything else is v1.
 
**Data issues surfacing late**: You will discover labeling problems, timestamp edge cases, and missing data during model training (Phase 3), not during data pipeline construction (Phase 1). Budget time for going back to Phase 1 during Phase 3. This is normal and expected.
 
**Hardware for training**: The shallow MLP will train in minutes on a laptop CPU. FinBERT embedding of the full corpus is the only compute-intensive step and should take under an hour on a GPU or a few hours on CPU. No cloud compute budget is required for v0.
 

---

## 12. Open questions (for discussion)

The following questions remain open and should be addressed during the early experimental phase.

1. **Embedding dimensionality reduction**: Should we apply PCA or a learned projection to reduce $Q$ from 768 to a smaller dimension before concatenation? With fewer than 10K samples, 768 input features from the embedding alone may cause overfitting even with the shallow architecture.

2. **Indicator correlation structure**: The target indicators are correlated (VIX spikes with SPX drops, flight-to-quality in 10Y). Should we model this explicitly (e.g., shared output layer with correlation penalty) or let the shared trunk learn it implicitly?

3. **Horizon-specific training weighting**: Should $\mathcal{L}_1$, $\mathcal{L}_2$, $\mathcal{L}_3$ be weighted equally, or should we upweight $\mathcal{L}_1$ given that shorter horizons are inherently more predictable and provide the foundation for the chain?

4. **Confidence thresholding at inference**: Should the UI display a "low confidence" warning when the maximum softmax probability falls below a threshold (e.g., 40%), signaling to the user that the model is uncertain and they should rely more on their own judgment?

5. **Augmentation strategy**: Given data scarcity, should we use LLM-generated synthetic headlines (variations of real events) to expand the training set? This requires careful validation to avoid introducing distribution shift.
