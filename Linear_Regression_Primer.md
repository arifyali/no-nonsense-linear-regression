# The No-Nonsense Guide to Linear Regression

## A practitioner's guide for data science interviews and beyond

## Notation Guide

**Common notation used throughout:**
- $n$ = number of observations (sample size)
- $p$ = number of predictor variables (features)
- $Y$ = response/outcome variable (vector $\in \mathbb{R}^n$)
- $X$ = design matrix ($n \times (p+1)$, including intercept column)
- $x_j$ = the $j$-th predictor variable
- $\beta$ = coefficient vector $(\beta_0, \beta_1, \ldots, \beta_p)^\top$
- $\hat{\beta}$ = estimated coefficient vector
- $\hat{Y} = X\hat{\beta}$ = fitted/predicted values
- $\varepsilon_i$ = error term for observation $i$ (unobservable)
- $e_i = Y_i - \hat{Y}_i$ = residual for observation $i$ (observable)
- $\bar{Y}$ = sample mean of $Y$
- $\sigma^2$ = error variance (population)
- $s^2$ = estimated error variance (sample): $s^2 = \frac{\text{SSE}}{n - p - 1}$
- $\lambda$ = regularization parameter (Sections 8)
- $H = X(X^\top X)^{-1}X^\top$ = hat matrix (projection matrix)

## Why This Matters

Linear regression isn't just a teaching tool — it's the foundation that logistic regression, GLMs, and neural networks are built on. It's still used in production across finance, healthcare, insurance, and econometrics, wherever stakeholders need a model they can explain to a regulator or a jury. It's also the most common topic in data science interviews, and for good reason: your ability to reason about regression reveals whether you understand modeling fundamentals. See [Section 10.1](#101-why-linear-regression-in-2026) for the full case.

## 1. The Linear Model

Everything in this section starts from one assumption: **the data-generating process is linear**. That is, we assume $Y$ really is a linear function of the predictors plus noise. The parameters $\beta_0, \beta_1, \ldots, \beta_p$ only have their clean interpretations ("change in $Y$ per unit change in $x$") *because* we're making this assumption.

If the true relationship isn't linear, OLS still gives you something — it finds the best linear approximation (the projection of $Y$ onto the column space of $X$). But the $\beta$ values become "best linear fit" coefficients, not parameters of the true data-generating process. Whether that distinction matters depends on your goal. We'll formalize these assumptions in [Section 3](#3-assumptions-of-linear-regression).

### 1.1. Simple Linear Regression

For a single predictor $x$:

$$Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad i = 1, 2, \ldots, n$$

where:
- $\beta_0$ = **intercept** — the expected value of $Y$ when $x = 0$
- $\beta_1$ = **slope** — the expected change in $Y$ for a one-unit increase in $x$
- $\varepsilon_i$ = **error term** — captures everything the model doesn't explain

**Interpretation**: $\beta_1$ answers the question: "If I increase $x$ by 1 unit, how much does $Y$ change on average, all else equal?"

### 1.2. Multiple Linear Regression

With $p$ predictors:

$$Y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i$$

In matrix form:

$$Y = X\beta + \varepsilon$$

where:

$$Y = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix}, \quad X = \begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{bmatrix}, \quad \beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{bmatrix}, \quad \varepsilon = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}$$

The column of 1s in $X$ corresponds to the intercept term $\beta_0$.

**Key interpretation**: In multiple regression, each $\beta_j$ represents the expected change in $Y$ for a one-unit increase in $x_j$, **holding all other predictors constant**. This "holding constant" part is critical — it's what separates multiple regression from running $p$ separate simple regressions.

### 1.3. What "Linear" Means

"Linear" refers to linearity **in the parameters** $\beta$, not in the predictors $x$. These are all linear regression models:

| Model | Formula | Why it's linear |
|-------|---------|-----------------|
| Polynomial | $Y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon$ | Linear in $\beta_0, \beta_1, \beta_2$; just treat $x^2$ as a new feature |
| Log-transformed predictor | $Y = \beta_0 + \beta_1 \log(x) + \varepsilon$ | Linear in $\beta$; $\log(x)$ is just another feature |
| Interaction | $Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 + \varepsilon$ | Linear in all $\beta$s |

This is **not** a linear regression model: $Y = \beta_0 e^{\beta_1 x} + \varepsilon$ (nonlinear in $\beta_1$).

## 2. Ordinary Least Squares (OLS) Estimation

### 2.1. The Objective

OLS finds the $\hat{\beta}$ that minimizes the sum of squared residuals:

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \arg\min_{\beta} \|Y - X\beta\|^2$$

**Why squared?** Squaring penalizes large errors more than small ones and makes the objective differentiable everywhere (unlike absolute value).

### 2.2. Derivation of the Normal Equations

Expand the objective:

$$\text{SSE}(\beta) = (Y - X\beta)^\top(Y - X\beta) = Y^\top Y - 2\beta^\top X^\top Y + \beta^\top X^\top X \beta$$

Take the gradient with respect to $\beta$ and set it to zero:

$$\frac{\partial \text{SSE}}{\partial \beta} = -2X^\top Y + 2X^\top X\beta = 0$$

Solving:

$$X^\top X \hat{\beta} = X^\top Y$$

$$\boxed{\hat{\beta} = (X^\top X)^{-1} X^\top Y}$$

These are the **normal equations**. They have a unique solution when $X^\top X$ is invertible, which requires:
1. $n \geq p + 1$ (at least as many observations as parameters)
2. No perfect multicollinearity (no predictor is an exact linear combination of others)

### 2.3. Geometric Interpretation

The fitted values are:

$$\hat{Y} = X\hat{\beta} = X(X^\top X)^{-1}X^\top Y = HY$$

where $H = X(X^\top X)^{-1}X^\top$ is the **hat matrix** (or projection matrix).

$H$ projects $Y$ onto the column space of $X$. The residuals $e = Y - \hat{Y} = (I - H)Y$ are orthogonal to this column space:

$$X^\top e = X^\top(Y - X\hat{\beta}) = X^\top Y - X^\top X \hat{\beta} = 0$$

**Intuition**: OLS finds the point in the "space of all possible predictions" (column space of $X$) that is closest to the observed $Y$. The residual vector is perpendicular to this space.

### 2.4. Gauss-Markov Theorem

**Statement**: Under the classical assumptions (Section 3), OLS is **BLUE** — the **B**est **L**inear **U**nbiased **E**stimator.

- **Best**: Lowest variance among all linear unbiased estimators
- **Linear**: $\hat{\beta}$ is a linear function of $Y$ (i.e., $\hat{\beta} = CY$ for some matrix $C$)
- **Unbiased**: $\mathbb{E}[\hat{\beta}] = \beta$
- **Estimator**: It's a function of the data used to estimate $\beta$

**What this means in practice**: If the assumptions hold, you cannot do better than OLS (among linear unbiased estimators). If assumptions are violated, or if you're willing to accept some bias, regularized estimators (Section 8) can have lower mean squared error.

**What Gauss-Markov does NOT say**:
- It doesn't say OLS is the best estimator period — just the best *linear unbiased* one
- It doesn't require normality (that's needed for inference, not for BLUE)
- It doesn't mean OLS has the lowest MSE — biased estimators like Ridge can beat it

### 2.5. Properties of OLS Estimators

Under the classical assumptions:

$$\mathbb{E}[\hat{\beta}] = \beta \quad \text{(unbiased)}$$

$$\text{Var}(\hat{\beta}) = \sigma^2 (X^\top X)^{-1} \quad \text{(covariance matrix)}$$

The variance of the $j$-th coefficient:

$$\text{Var}(\hat{\beta}_j) = \sigma^2 \left[(X^\top X)^{-1}\right]_{jj}$$

Estimated by substituting $s^2 = \frac{\text{SSE}}{n - p - 1}$ for $\sigma^2$.

**Key insight**: The variance of each $\hat{\beta}_j$ increases with:
- Higher error variance $\sigma^2$ (noisier data)
- Higher multicollinearity (inflates $(X^\top X)^{-1}$ entries)
- Smaller sample size $n$
- Less variation in $x_j$

## 3. Assumptions of Linear Regression

You'll often see the mnemonic **LINE** (**L**inearity, **I**ndependence, **N**ormality, **E**qual variance), plus two additional conditions. But LINE treats all assumptions as equally important — they're not. What follows is organized by how much damage a violation actually does.

**Tier 1 — Structural** (violations bias $\hat{\beta}$; your coefficients are wrong):
- Linearity (3.1), Exogeneity (3.2)

**Tier 2 — Inferential** (coefficients are still unbiased, but standard errors, p-values, and CIs are wrong):
- Independence of Errors (3.3), Homoscedasticity (3.4)

**Tier 3 — Technical / Mild** (either a computational prerequisite or largely handled by large samples):
- Normality of Errors (3.5), No Perfect Multicollinearity (3.6)

If you only have time to check two things, check Tier 1.

---

### Tier 1: Structural Assumptions

*Violations here mean your $\hat{\beta}$ values are biased — they don't estimate what you think they estimate.*

### 3.1. Linearity

**Formal statement**: The true relationship between $Y$ and $X$ is linear: $\mathbb{E}[Y \mid X] = X\beta$.

**What happens when violated**: Biased coefficient estimates. The model systematically over- or under-predicts in certain ranges of $X$.

**How to detect**:
- Residuals vs. fitted values plot: look for curved patterns
- Residuals vs. each predictor: look for non-random patterns
- Component-plus-residual plots (partial residual plots)

**How to fix**:
- Add polynomial terms ($x^2$, $x^3$)
- Apply transformations (log, square root)
- Add interaction terms
- Use a different model (GAM, tree-based)

### 3.2. Exogeneity (Zero Conditional Mean)

**Formal statement**: $\mathbb{E}[\varepsilon \mid X] = 0$. The errors are uncorrelated with the predictors.

**What happens when violated**: OLS estimates are biased. This is the most damaging violation because it undermines the fundamental interpretation of coefficients.

**Common causes**:
- Omitted variable bias: a relevant variable is left out and is correlated with an included predictor
- Simultaneity: $X$ causes $Y$ but $Y$ also causes $X$
- Measurement error in $X$

**How to fix**:
- Include the omitted variable
- Instrumental variables (IV) / two-stage least squares
- Randomized experiments (eliminates endogeneity by design)

---

### Tier 2: Inferential Assumptions

*Violations here don't bias your coefficients, but your standard errors, p-values, and confidence intervals are wrong. You might think a feature is significant when it isn't (or vice versa).*

### 3.3. Independence of Errors

**Formal statement**: $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ for all $i \neq j$, or equivalently $\text{Var}(\varepsilon) = \sigma^2 I$.

**What happens when violated**: Coefficient estimates are still unbiased, but standard errors are wrong. Confidence intervals and p-values become unreliable. Typically underestimates standard errors → inflated Type I error.

**Common violations**:
- Time series data (autocorrelation)
- Spatial data (nearby observations are correlated)
- Clustered data (students within schools, patients within hospitals)

**How to detect**:
- Durbin-Watson test (autocorrelation)
- Plot residuals vs. time/order
- ACF (autocorrelation function) of residuals

**How to fix**:
- Clustered standard errors
- Generalized Least Squares (GLS)
- Include time/spatial terms in the model
- Use time-series specific models (ARIMA)

### 3.4. Equal Variance (Homoscedasticity)

**Formal statement**: $\text{Var}(\varepsilon_i) = \sigma^2$ for all $i$ (constant variance).

**What happens when violated (heteroscedasticity)**: OLS estimates are still unbiased, but standard errors are wrong. This affects hypothesis tests and confidence intervals.

**How to detect**:
- Residuals vs. fitted values: look for a "fan" or "cone" shape
- Scale-location plot: $\sqrt{|e_i|}$ vs. fitted values
- Breusch-Pagan test, White's test

**How to fix**:
- **Robust standard errors** (HC0–HC3, also called "sandwich estimators") — the most practical fix. Doesn't change coefficients, just corrects the standard errors.
  - HC0: White's original estimator
  - HC1: Small-sample correction ($\times \frac{n}{n-p-1}$)
  - HC3: Best for small samples, recommended default
- Weighted Least Squares (WLS) if you know the variance structure
- Transform $Y$ (log transform often stabilizes variance)

---

### Tier 3: Technical / Mild Assumptions

*One is a computational prerequisite; the other is largely a non-issue with modern sample sizes.*

### 3.5. Normality of Errors

**Formal statement**: $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ for all $i$.

**What happens when violated**: OLS estimates are still unbiased and BLUE (Gauss-Markov doesn't require normality). But exact t-tests and F-tests are no longer valid. With large $n$, the Central Limit Theorem kicks in and inference is approximately valid anyway.

**How to detect**:
- Q-Q plot of residuals
- Shapiro-Wilk test (for $n < 5000$)
- Histogram of residuals

**How to fix**:
- Often not necessary with large $n$ (CLT)
- Transform $Y$ (log, Box-Cox)
- Use bootstrapped confidence intervals
- Use robust standard errors

**Interview note**: This is the *least important* assumption. Many practitioners don't even check it. With $n > 30$, CLT provides good approximations. Focus your energy on linearity and homoscedasticity.

### 3.6. No Perfect Multicollinearity

**Formal statement**: No predictor is an exact linear combination of other predictors. Equivalently, $X^\top X$ is invertible ($\text{rank}(X) = p + 1$).

**Note**: *Near* multicollinearity (high but not perfect correlation) is allowed but problematic — see Section 6.

---

### 3.7. Summary Table

| | Assumption | Violated → Effect on $\hat{\beta}$ | Violated → Effect on Inference | Fix |
|---|---|---|---|---|
| **Tier 1** | Linearity | Biased | Invalid | Transform, add terms |
| | Exogeneity | Biased | Invalid | Include omitted vars, IV |
| **Tier 2** | Independence | Unbiased | Invalid (SEs wrong) | Clustered SEs, GLS |
| | Homoscedasticity | Unbiased | Invalid (SEs wrong) | Robust SEs, WLS |
| **Tier 3** | Normality | Unbiased | Approximately valid (large $n$) | CLT, bootstrap |
| | No multicollinearity | Undefined (can't estimate) | N/A | Drop/combine variables |

## 4. Hypothesis Testing & Inference

### 4.1. Testing Individual Coefficients (t-test)

**Hypothesis**: $H_0: \beta_j = 0$ vs. $H_1: \beta_j \neq 0$

**Test statistic**:

$$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} = \frac{\hat{\beta}_j}{s \sqrt{[(X^\top X)^{-1}]_{jj}}}$$

Under $H_0$ and the classical assumptions: $t_j \sim t_{n-p-1}$.

**Decision**: Reject $H_0$ if $|t_j| > t_{\alpha/2, \, n-p-1}$, or equivalently if the p-value $< \alpha$.

**Interpretation**: A significant t-test means the predictor $x_j$ has a statistically significant linear relationship with $Y$, **after accounting for all other predictors in the model**.

**Confidence interval** for $\beta_j$:

$$\hat{\beta}_j \pm t_{\alpha/2, \, n-p-1} \cdot \text{SE}(\hat{\beta}_j)$$

### 4.2. Testing Overall Model Significance (F-test)

**Hypothesis**: $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$ (no predictor is useful)

**Test statistic**:

$$F = \frac{\text{SSR}/p}{\text{SSE}/(n-p-1)} = \frac{\text{MSR}}{\text{MSE}} = \frac{R^2 / p}{(1 - R^2)/(n-p-1)}$$

Under $H_0$: $F \sim F_{p, \, n-p-1}$.

**ANOVA Table for Regression**:

| Source | df | SS | MS | F |
|--------|---|---|---|---|
| Regression | $p$ | SSR | MSR = SSR/$p$ | MSR/MSE |
| Error | $n-p-1$ | SSE | MSE = SSE/$(n-p-1)$ | |
| Total | $n-1$ | SST | | |

**Interpretation**: If the overall F-test is not significant, the entire model provides no better prediction than just using $\bar{Y}$.

### 4.3. Partial F-test (Nested Models)

**Use case**: Test whether a *subset* of predictors improves the model.

**Setup**: Compare a full model (all $p$ predictors) to a reduced model (omitting $q$ predictors).

**Test statistic**:

$$F = \frac{(\text{SSE}_{\text{reduced}} - \text{SSE}_{\text{full}})/q}{\text{SSE}_{\text{full}}/(n - p - 1)}$$

Under $H_0$ (the $q$ omitted predictors have zero coefficients): $F \sim F_{q, \, n-p-1}$.

**Example**: You have a model with 10 predictors. You want to test whether the 3 TF-IDF features collectively add value. Compare the 10-predictor model to the 7-predictor model using the partial F-test.

### 4.4. Confidence Intervals vs. Prediction Intervals

For a new observation $x_0$:

**Confidence interval** (for the mean response $\mathbb{E}[Y \mid X = x_0]$):

$$\hat{Y}_0 \pm t_{\alpha/2, \, n-p-1} \cdot s \sqrt{x_0^\top (X^\top X)^{-1} x_0}$$

**Prediction interval** (for a single new observation $Y_0$):

$$\hat{Y}_0 \pm t_{\alpha/2, \, n-p-1} \cdot s \sqrt{1 + x_0^\top (X^\top X)^{-1} x_0}$$

**The key difference**: Prediction intervals are always wider because they account for both:
1. Uncertainty in estimating the mean (same as confidence interval)
2. Irreducible noise $\varepsilon$ in a single new observation

**Interview tip**: "A confidence interval tells you where the *average* response is for a given $x$. A prediction interval tells you where a *single new observation* will likely fall. The prediction interval is always wider because individual observations have noise."

## 5. Model Evaluation Metrics

### 5.1. Coefficient of Determination ($R^2$)

**Sum of Squares Decomposition**:

$$\underbrace{\sum_{i=1}^n (Y_i - \bar{Y})^2}_{\text{SST (Total)}} = \underbrace{\sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2}_{\text{SSR (Regression)}} + \underbrace{\sum_{i=1}^n (Y_i - \hat{Y}_i)^2}_{\text{SSE (Error)}}$$

$$R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}}$$

**Interpretation**: The proportion of variance in $Y$ explained by the model.
- $R^2 = 0$: Model explains nothing (no better than predicting $\bar{Y}$)
- $R^2 = 1$: Model explains everything (perfect fit)

**Important properties**:
- $R^2$ never decreases when you add more predictors (even useless ones)
- $R^2$ = squared correlation between $Y$ and $\hat{Y}$ (in OLS with intercept)
- A "good" $R^2$ depends entirely on context. $R^2 = 0.02$ in social science can be meaningful. $R^2 = 0.90$ in physics might be disappointing.

### 5.2. Adjusted $R^2$

Penalizes for adding unnecessary predictors:

$$R^2_{\text{adj}} = 1 - \frac{\text{SSE}/(n-p-1)}{\text{SST}/(n-1)} = 1 - \frac{n-1}{n-p-1}(1 - R^2)$$

**Key property**: $R^2_{\text{adj}}$ can decrease if a new predictor doesn't improve the model enough to justify the loss of a degree of freedom.

**Interview tip**: "Use adjusted $R^2$ when comparing models with different numbers of predictors. Regular $R^2$ always goes up when you add variables, even noise. Adjusted $R^2$ penalizes complexity."

### 5.3. RMSE, MAE, and MAPE

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(Y_i - \hat{Y}_i)^2}$ | Average error magnitude, sensitive to outliers. Same units as $Y$. |
| **MAE** | $\frac{1}{n}\sum\|Y_i - \hat{Y}_i\|$ | Average absolute error. More robust to outliers than RMSE. Same units as $Y$. |
| **MAPE** | $\frac{100}{n}\sum\left\|\frac{Y_i - \hat{Y}_i}{Y_i}\right\|$ | Percentage error. Scale-free. Undefined when $Y_i = 0$. |

**When to use which**:
- **RMSE** when large errors are disproportionately bad (e.g., predicting costs)
- **MAE** when all errors matter equally (e.g., median-like behavior desired)
- **MAPE** when you need a scale-free measure (comparing across different scales)
- Note: RMSE $\geq$ MAE always, with equality only when all errors are equal

### 5.4. AIC and BIC

For model comparison and selection:

$$\text{AIC} = n \ln(\text{SSE}/n) + 2(p+1)$$

$$\text{BIC} = n \ln(\text{SSE}/n) + \ln(n)(p+1)$$

- Lower is better for both
- BIC penalizes complexity more heavily than AIC (the $\ln(n)$ factor exceeds 2 when $n > 7$)
- BIC tends to select simpler models, AIC tends to select more complex ones
- Neither has an absolute interpretation — only useful for comparing models on the same data

### 5.5. Cross-Validation

**Why**: $R^2$ on training data is optimistic (overfitting). Cross-validation estimates out-of-sample performance.

**$k$-Fold Cross-Validation**:
1. Split data into $k$ equal folds
2. For each fold $i$: train on all folds except $i$, predict on fold $i$
3. Average the performance metric across folds

**Common choices**: $k = 5$ or $k = 10$. Leave-one-out (LOO) cross-validation is $k = n$.

**Special result for linear regression**: LOO CV can be computed without refitting the model $n$ times:

$$\text{CV}_{\text{LOO}} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{e_i}{1 - h_{ii}}\right)^2$$

where $h_{ii}$ is the $i$-th diagonal element of the hat matrix $H$. This is called the PRESS (Predicted Residual Error Sum of Squares) statistic.

## 6. Multicollinearity

### 6.1. What It Is

Multicollinearity occurs when two or more predictors are highly correlated. This makes $(X^\top X)$ nearly singular, inflating the variance of $\hat{\beta}$.

**Perfect multicollinearity**: One predictor is an exact linear combination of others. $(X^\top X)$ is singular and OLS cannot be computed. Example: Including both "temperature in Celsius" and "temperature in Fahrenheit."

**Near multicollinearity**: Predictors are highly but not perfectly correlated. OLS works but coefficients become unstable.

### 6.2. Effects

- Individual coefficient estimates become **unstable** (large standard errors)
- Small changes in data lead to large changes in coefficients
- Individual t-tests may show non-significance even when the overall F-test is significant
- **Coefficients are still unbiased** — just high-variance

### 6.3. Variance Inflation Factor (VIF)

For predictor $x_j$, regress $x_j$ on all other predictors and compute $R_j^2$:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

**Interpretation**:
- $\text{VIF}_j = 1$: No correlation with other predictors
- $\text{VIF}_j = 5$: Standard error of $\hat{\beta}_j$ is $\sqrt{5} \approx 2.24$ times larger than it would be with no multicollinearity
- $\text{VIF}_j = 10$: Commonly used as a threshold for "problematic"

**Rule of thumb**: VIF > 10 suggests serious multicollinearity. Some use VIF > 5 as a stricter threshold.

### 6.4. Remedies

1. **Drop one of the correlated predictors** (simplest)
2. **Combine correlated predictors** (e.g., create a composite score, use PCA)
3. **Ridge regression** (Section 8.1) — designed specifically for this problem
4. **Do nothing** — if you only care about prediction (not individual coefficients), multicollinearity doesn't affect $\hat{Y}$

**Interview tip**: "Multicollinearity inflates coefficient variance but doesn't bias them. If I care about prediction, it's often fine. If I care about interpreting individual coefficients, I need to fix it — either by dropping variables, using PCA, or using Ridge regression."

## 7. Feature Engineering & Variable Selection

### 7.1. Categorical Variables

Categorical variables must be encoded numerically. For a categorical variable with $k$ levels, create $k - 1$ **dummy variables** (one-hot encoding minus one level).

**Example**: Color $\in$ {Red, Green, Blue}

| Observation | $D_{\text{Green}}$ | $D_{\text{Blue}}$ |
|---|---|---|
| Red | 0 | 0 |
| Green | 1 | 0 |
| Blue | 0 | 1 |

The omitted level (Red) is the **reference category**. Coefficients are interpreted *relative to the reference*:
- $\hat{\beta}_{\text{Green}}$ = difference in predicted $Y$ between Green and Red
- $\hat{\beta}_{\text{Blue}}$ = difference in predicted $Y$ between Blue and Red

**Why $k-1$ and not $k$ dummies?** Including all $k$ creates perfect multicollinearity with the intercept (the dummy columns sum to the intercept column). This is called the **dummy variable trap**.

### 7.2. Interaction Terms

An interaction term captures how the effect of one variable depends on the level of another:

$$Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 + \varepsilon$$

**Interpretation**: The effect of $x_1$ on $Y$ is $\beta_1 + \beta_3 x_2$ — it depends on $x_2$.

**When to include**: When you believe (or discover) that the relationship between a predictor and the outcome changes depending on another variable.

**Best practice**: Always include the main effects ($x_1$ and $x_2$) when including their interaction. Omitting main effects is rarely justified and leads to misleading coefficients.

### 7.3. Polynomial Features

For non-linear relationships:

$$Y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \varepsilon$$

**Caution**: Polynomials of degree > 3 tend to overfit. They also cause multicollinearity ($x$ and $x^2$ are correlated). Centering $x$ before squaring can help.

### 7.4. Log Transformations

Log transformations change coefficient interpretation:

| Model | Formula | Interpretation of $\beta_1$ |
|-------|---------|----------------------------|
| **Linear** (level-level) | $Y = \beta_0 + \beta_1 x$ | 1-unit increase in $x$ → $\beta_1$ unit change in $Y$ |
| **Log-linear** (log-level) | $\log Y = \beta_0 + \beta_1 x$ | 1-unit increase in $x$ → approx. $100\beta_1$% change in $Y$ |
| **Log-log** | $\log Y = \beta_0 + \beta_1 \log x$ | 1% increase in $x$ → $\beta_1$% change in $Y$ (elasticity) |
| **Lin-log** (level-log) | $Y = \beta_0 + \beta_1 \log x$ | 1% increase in $x$ → $\beta_1/100$ unit change in $Y$ |

**When to log-transform**:
- **$Y$**: When the distribution is right-skewed, when the relationship is multiplicative, or to stabilize variance
- **$x$**: When the effect of $x$ diminishes as $x$ gets larger (diminishing returns)
- **Both**: When you're modeling an elasticity or proportional relationship

**Caution**: $\log$ is undefined for zero and negative values. Common workarounds: $\log(Y + 1)$, or use an inverse hyperbolic sine transform $\text{asinh}(Y)$.

### 7.5. Variable Selection

#### Stepwise Methods

- **Forward selection**: Start with no predictors, add the most significant one at a time
- **Backward elimination**: Start with all predictors, remove the least significant one at a time
- **Stepwise**: Combination of forward and backward

**Problems with stepwise**:
- P-values are invalid because of multiple testing
- Coefficient estimates are biased upward
- Results are unstable — small changes in data lead to different selected models
- Not recommended for inference; acceptable as a rough screening tool

#### Better Approaches

- **Domain knowledge**: Let theory guide which variables to include
- **Regularization** (Section 8): Lasso performs automatic variable selection
- **Information criteria**: Compare models using AIC/BIC
- **Cross-validation**: Select the model with best out-of-sample performance

## 8. Regularization

### 8.1. Ridge Regression (L2)

**Objective**:

$$\hat{\beta}^{\text{ridge}} = \arg\min_{\beta} \left\{ \|Y - X\beta\|^2 + \lambda \|\beta\|_2^2 \right\} = \arg\min_{\beta} \left\{ \sum_{i=1}^n (Y_i - X_i^\top \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}$$

**Closed-form solution**:

$$\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top Y$$

**Key properties**:
- $\lambda = 0$: Reduces to OLS
- $\lambda \to \infty$: All coefficients shrink to zero
- Coefficients are **biased** but have lower variance than OLS
- **Never sets coefficients exactly to zero** — doesn't perform variable selection
- Handles multicollinearity by stabilizing $(X^\top X + \lambda I)$

**Geometric intuition**: Ridge constrains coefficients to lie within a sphere ($\sum \beta_j^2 \leq t$). The OLS solution is "pulled" toward the origin.

**When to use**: Many correlated predictors, none of which you want to eliminate entirely. Prediction-focused problems where interpretability is less critical.

### 8.2. Lasso Regression (L1)

**Objective**:

$$\hat{\beta}^{\text{lasso}} = \arg\min_{\beta} \left\{ \|Y - X\beta\|^2 + \lambda \|\beta\|_1 \right\} = \arg\min_{\beta} \left\{ \sum_{i=1}^n (Y_i - X_i^\top \beta)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

**Key properties**:
- No closed-form solution (solved via coordinate descent or other optimization)
- **Sets some coefficients exactly to zero** — performs automatic variable selection
- Produces **sparse models** — easier to interpret
- With correlated predictors, tends to pick one and set others to zero (somewhat arbitrary choice)

**Geometric intuition**: Lasso constrains coefficients to lie within a diamond ($\sum |\beta_j| \leq t$). The diamond's corners lie on the axes, so the OLS solution often hits a corner where some $\beta_j = 0$.

**When to use**: High-dimensional data where you suspect many features are irrelevant. When you want a sparse, interpretable model.

### 8.3. Elastic Net

**Objective**: Combines L1 and L2 penalties:

$$\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \left\{ \|Y - X\beta\|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2 \right\}$$

Often parameterized with a mixing parameter $\alpha \in [0, 1]$:

$$\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \left\{ \|Y - X\beta\|^2 + \lambda \left[ \alpha \|\beta\|_1 + (1-\alpha) \|\beta\|_2^2 \right] \right\}$$

- $\alpha = 1$: Pure Lasso
- $\alpha = 0$: Pure Ridge

**When to use**: When you have correlated predictors and want variable selection. Elastic Net tends to select groups of correlated predictors together (unlike Lasso, which picks one arbitrarily).

### 8.4. The Bias-Variance Tradeoff

For any estimator $\hat{\beta}$:

$$\text{MSE}(\hat{\beta}) = \text{Bias}^2(\hat{\beta}) + \text{Var}(\hat{\beta})$$

| | OLS | Ridge/Lasso |
|---|---|---|
| **Bias** | 0 (unbiased) | > 0 (biased) |
| **Variance** | Can be high | Lower (controlled by $\lambda$) |
| **MSE** | Can be high when $p$ is large or multicollinearity exists | Often lower overall |

**Intuition**: OLS fits the training data as closely as possible (zero bias, high variance). Regularization accepts some bias to substantially reduce variance, often resulting in better predictions on new data.

**Choosing $\lambda$**: Use cross-validation to select the $\lambda$ that minimizes out-of-sample error. Common choices:
- $\lambda_{\min}$: The $\lambda$ with lowest CV error
- $\lambda_{1\text{SE}}$: The largest $\lambda$ within one standard error of the minimum (simpler model)

### 8.5. Comparison Table

| | OLS | Ridge | Lasso | Elastic Net |
|---|---|---|---|---|
| **Penalty** | None | $\lambda\sum\beta_j^2$ | $\lambda\sum\|\beta_j\|$ | $\lambda[\alpha\sum\|\beta_j\| + (1-\alpha)\sum\beta_j^2]$ |
| **Variable selection** | No | No | Yes | Yes |
| **Closed form** | Yes | Yes | No | No |
| **Multicollinearity** | Fails | Handles well | Picks one arbitrarily | Selects groups |
| **Bias** | None | Some | Some | Some |
| **Best for** | Small $p$, low collinearity | Many correlated features | Sparse solutions | Correlated + sparse |

## 9. Diagnostics

### 9.1. Residual Plots

The four standard diagnostic plots:

**1. Residuals vs. Fitted Values**
- What to look for: Random scatter around zero
- Red flag: Curved pattern (non-linearity), funnel shape (heteroscedasticity)

**2. Normal Q-Q Plot**
- What to look for: Points on the diagonal line
- Red flag: Systematic departures, heavy tails (S-shape), skewness (bowing)

**3. Scale-Location (Spread-Location)**
- Plot $\sqrt{|e_i|}$ vs. $\hat{Y}_i$
- What to look for: Flat trend line
- Red flag: Increasing trend (variance increases with fitted values)

**4. Residuals vs. Leverage**
- What to look for: No points in the upper-right or lower-right corners
- Red flag: Points with high leverage AND large residuals

### 9.2. Leverage and Influence

**Leverage** ($h_{ii}$): How far observation $i$'s predictors are from the center of the predictor space. Comes from the diagonal of the hat matrix:

$$h_{ii} = [H]_{ii} = x_i^\top (X^\top X)^{-1} x_i$$

- Average leverage = $(p+1)/n$
- High leverage: $h_{ii} > 2(p+1)/n$
- Range: $0 \leq h_{ii} \leq 1$

**Influence**: How much removing observation $i$ changes the fitted model.

**Cook's Distance**: Combines leverage and residual magnitude:

$$D_i = \frac{e_i^2}{(p+1) \cdot \text{MSE}} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}$$

- Rule of thumb: $D_i > 4/n$ or $D_i > 1$ indicates a highly influential point
- High Cook's distance means removing that observation would substantially change the regression

**DFFITS**: Another influence measure:

$$\text{DFFITS}_i = \frac{\hat{Y}_i - \hat{Y}_{i(i)}}{\sqrt{\text{MSE}_{(i)} \cdot h_{ii}}}$$

where $\hat{Y}_{i(i)}$ is the prediction for observation $i$ when it's excluded from the fit.

**Interview tip**: "An outlier has an unusual $Y$ value. A high-leverage point has unusual $X$ values. An influential point is one whose removal would substantially change the regression. A point can be high-leverage without being influential (if it follows the trend) or an outlier without being influential (if it's in the middle of the $X$ space)."

### 9.3. Robust Standard Errors

When heteroscedasticity is present, use heteroscedasticity-consistent (HC) standard errors instead of the classical formula:

| Estimator | Formula for $\hat{V}$ | Notes |
|---|---|---|
| **Classical** | $s^2 (X^\top X)^{-1}$ | Assumes homoscedasticity |
| **HC0** (White) | $(X^\top X)^{-1} X^\top \text{diag}(e_i^2) X (X^\top X)^{-1}$ | Original robust estimator |
| **HC1** | $\frac{n}{n-p-1} \cdot \text{HC0}$ | Degrees-of-freedom correction |
| **HC3** | Uses $\frac{e_i^2}{(1-h_{ii})^2}$ instead of $e_i^2$ | Best for small samples, **recommended default** |

**Practical advice**: Just use HC3 robust standard errors by default. The coefficient estimates don't change — only the standard errors, confidence intervals, and p-values.

In Python (statsmodels): `model.fit(cov_type='HC3')`

## 10. Practical Considerations

### 10.1. Why Linear Regression in 2026?

In a world of gradient boosting and large language models, linear regression isn't going anywhere. Here's why:

**Interpretability is a requirement, not a nice-to-have.** Regulated industries — finance (SR 11-7, Basel), healthcare (FDA), insurance (actuarial standards) — need models where you can explain exactly what each feature does and why. "The random forest says so" doesn't satisfy a regulator. Linear regression does.

**It's the foundation of modern ML.** Logistic regression is linear regression + a sigmoid link function. Neural networks are stacked linear transformations with nonlinear activations between them. GLMs generalize linear regression to non-normal outcomes. If you don't understand linear regression deeply, you're building on sand.

**It's still used in production.** A/B test analysis (estimating treatment effects), causal inference in econometrics (difference-in-differences, instrumental variables), clinical trial endpoints, insurance pricing (GLMs), and real estate valuation all run on regression. Most "ML" in traditional industries is still some form of linear model.

**It's often good enough.** With thoughtful feature engineering, linear models are competitive with complex models on tabular data — especially when $n$ is small, interpretability matters, or you need confidence intervals on individual predictions. And they train in milliseconds, not hours.

**When to use it:**
- You need interpretable coefficients ("what drives the outcome?")
- The relationship is approximately linear (or can be made so with transformations)
- You have more observations than features ($n > p$)
- Stakeholders need a model they can understand and explain
- As a baseline before trying more complex models

**Don't use it when**:
- The relationship is strongly non-linear and transformations don't help
- You have more features than observations ($p > n$) — use regularization or dimensionality reduction
- Prediction accuracy is the only goal and you have enough data for complex models
- The outcome is categorical (use logistic regression) or count-based (use Poisson regression)

### 10.2. Interpreting Coefficients with Transformed Features

**Standardized features** ($z$-scored): Coefficient = change in $Y$ for a 1 standard deviation change in $x_j$. Useful for comparing relative importance of features on different scales.

**Log-transformed target**: See Section 7.4 for interpretation rules.

**One-hot encoded features**: Coefficient = difference from the reference category.

**TF-IDF features**: Coefficient represents the association between the frequency of a term (weighted by inverse document frequency) and the outcome. Hard to interpret individually; more useful as a group.

### 10.3. Common Interview Questions & Answers

**Q: What is linear regression?**
A: "A model that estimates the linear relationship between a continuous outcome and one or more predictors. OLS finds the coefficients that minimize squared prediction errors. Each coefficient tells you the expected change in $Y$ for a one-unit change in that predictor, holding others constant."

**Q: What are the assumptions? What if they're violated?**
A: See Section 3.7 summary table. Lead with: "The key assumptions are linearity, independence, normality, and constant variance — but they're not all equally important. Linearity and exogeneity violations cause bias, which is serious. Normality violations are usually fine with large samples."

**Q: When would you use Ridge vs Lasso?**
A: "Ridge when I have many correlated features and want to keep all of them (e.g., sensor data). Lasso when I suspect many features are irrelevant and want automatic selection (e.g., genomics with thousands of genes). Elastic Net when I want both — variable selection with correlated features."

**Q: What's the difference between R² and adjusted R²?**
A: "$R^2$ always increases when you add predictors, even noise. Adjusted $R^2$ penalizes for unnecessary complexity. Use adjusted $R^2$ when comparing models with different numbers of features."

**Q: How do you handle multicollinearity?**
A: "First, detect it with VIF. If VIF > 10, I'd either drop one of the correlated predictors, combine them (e.g., PCA), or use Ridge regression. It doesn't bias coefficients, but it inflates their variance, making individual coefficients unreliable."

**Q: What's the bias-variance tradeoff?**
A: "OLS has zero bias but can have high variance, especially with many features or multicollinearity. Regularization introduces some bias but reduces variance. The total error (MSE = bias² + variance) is often lower with regularization. Cross-validation helps find the right balance."

**Q: Explain $p$-value in the context of regression.**
A: "The p-value for a coefficient is the probability of seeing a coefficient as extreme as ours (or more) if the true coefficient were zero. A small p-value means the observed relationship is unlikely to be due to chance alone. It does not tell you the effect is large or practically important."

**Q: How do you check if your model is good?**
A: "I look at multiple things: (1) $R^2$ and adjusted $R^2$ for overall fit, (2) residual plots for assumption violations, (3) cross-validated metrics for out-of-sample performance, (4) individual coefficient significance and signs for interpretability, (5) VIF for multicollinearity."

## Glossary

**Adjusted $R^2$**: A modified $R^2$ that penalizes for additional predictors. Unlike $R^2$, it can decrease when adding unhelpful variables: $R^2_{\text{adj}} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$.

**AIC (Akaike Information Criterion)**: Model comparison metric balancing fit and complexity: $\text{AIC} = n\ln(\text{SSE}/n) + 2(p+1)$. Lower is better. Tends to select more complex models than BIC.

**BIC (Bayesian Information Criterion)**: Like AIC but with a stronger complexity penalty: $\text{BIC} = n\ln(\text{SSE}/n) + \ln(n)(p+1)$. Lower is better. Tends to select simpler models than AIC.

**BLUE**: Best Linear Unbiased Estimator. OLS is BLUE under the Gauss-Markov assumptions. "Best" means minimum variance among all linear unbiased estimators.

**Coefficient of Determination**: See $R^2$.

**Collinearity**: High correlation between two predictors. **Multicollinearity** extends this to linear relationships among three or more predictors.

**Cook's Distance**: A measure of influence combining leverage and residual magnitude: $D_i = \frac{e_i^2}{(p+1)\text{MSE}} \cdot \frac{h_{ii}}{(1-h_{ii})^2}$. Points with $D_i > 4/n$ warrant investigation.

**Cross-Validation**: A resampling technique for estimating out-of-sample model performance. $k$-fold CV splits data into $k$ parts, trains on $k-1$, and tests on the held-out fold, rotating through all folds.

**Design Matrix ($X$)**: The $n \times (p+1)$ matrix of predictor values, including a column of ones for the intercept.

**Dummy Variable**: A binary (0/1) indicator variable representing one level of a categorical predictor. For $k$ categories, use $k-1$ dummy variables.

**Dummy Variable Trap**: Perfect multicollinearity caused by including all $k$ dummy variables for a $k$-level categorical predictor along with an intercept. Avoided by dropping one category as the reference.

**Elastic Net**: Regularization method combining L1 and L2 penalties: $\lambda[\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2]$. Handles correlated features better than pure Lasso while still performing variable selection.

**Endogeneity**: Violation of the exogeneity assumption ($\mathbb{E}[\varepsilon|X] \neq 0$). Causes biased and inconsistent OLS estimates. Common causes: omitted variables, simultaneity, measurement error.

**Exogeneity**: The assumption that errors are uncorrelated with predictors: $\mathbb{E}[\varepsilon|X] = 0$. Required for OLS unbiasedness.

**F-test**: A test comparing the fit of two nested models. The overall F-test checks whether any predictor is useful. The partial F-test checks whether a subset of predictors improves the model.

**Gauss-Markov Theorem**: Under the classical assumptions (linearity, exogeneity, homoscedasticity, no multicollinearity), OLS is the Best Linear Unbiased Estimator (BLUE). Does not require normality.

**Hat Matrix ($H$)**: The projection matrix $H = X(X^\top X)^{-1}X^\top$ that maps observed values to fitted values: $\hat{Y} = HY$. Diagonal elements $h_{ii}$ measure leverage.

**Heteroscedasticity**: Non-constant error variance ($\text{Var}(\varepsilon_i)$ depends on $i$ or $X$). Doesn't bias OLS coefficients but invalidates standard errors. Fix with robust (HC) standard errors or WLS.

**Homoscedasticity**: Constant error variance: $\text{Var}(\varepsilon_i) = \sigma^2$ for all $i$. One of the classical regression assumptions.

**Influential Point**: An observation whose removal substantially changes the fitted model. Measured by Cook's distance or DFFITS. Can be high-leverage, an outlier, or both.

**Interaction Term**: A product of two predictors ($x_1 \times x_2$) included in the model to capture how the effect of one predictor varies with the level of another.

**Lasso (Least Absolute Shrinkage and Selection Operator)**: L1-regularized regression that minimizes $\|Y - X\beta\|^2 + \lambda\|\beta\|_1$. Sets some coefficients exactly to zero, performing variable selection.

**Leverage ($h_{ii}$)**: How extreme observation $i$'s predictor values are relative to the rest. High-leverage points disproportionately influence the fitted line. Computed from the hat matrix diagonal.

**MAE (Mean Absolute Error)**: Average of absolute residuals: $\frac{1}{n}\sum|Y_i - \hat{Y}_i|$. More robust to outliers than RMSE. Same units as $Y$.

**MSE (Mean Squared Error)**: Can refer to: (1) the average squared prediction error, or (2) the estimator's expected squared deviation from the true parameter ($\text{Bias}^2 + \text{Variance}$).

**Normal Equations**: The system $X^\top X\hat{\beta} = X^\top Y$ whose solution gives the OLS estimator $\hat{\beta} = (X^\top X)^{-1}X^\top Y$.

**Omitted Variable Bias**: Bias in OLS estimates caused by excluding a relevant variable that is correlated with both an included predictor and the outcome.

**OLS (Ordinary Least Squares)**: The standard method for estimating linear regression coefficients by minimizing the sum of squared residuals.

**Overfitting**: When a model captures noise in the training data rather than the true signal, leading to poor performance on new data. Indicated by a large gap between training and test performance.

**Prediction Interval**: An interval for a single new observation, wider than a confidence interval because it accounts for both estimation uncertainty and irreducible noise.

**$R^2$ (R-squared)**: The proportion of variance in $Y$ explained by the model: $R^2 = 1 - \text{SSE}/\text{SST}$. Ranges from 0 to 1 in standard OLS with intercept.

**Residual ($e_i$)**: The observed minus fitted value: $e_i = Y_i - \hat{Y}_i$. An estimate of the unobservable error $\varepsilon_i$.

**Ridge Regression**: L2-regularized regression that minimizes $\|Y - X\beta\|^2 + \lambda\|\beta\|_2^2$. Shrinks coefficients toward zero but never to exactly zero. Closed-form: $\hat{\beta} = (X^\top X + \lambda I)^{-1}X^\top Y$.

**RMSE (Root Mean Squared Error)**: Square root of the average squared residual: $\sqrt{\frac{1}{n}\sum(Y_i - \hat{Y}_i)^2}$. Penalizes large errors more than MAE. Same units as $Y$.

**Robust Standard Errors**: Standard error estimates that remain valid under heteroscedasticity. HC3 is the recommended default. Do not change coefficients — only adjust standard errors, CIs, and p-values.

**SSE (Sum of Squared Errors)**: $\sum(Y_i - \hat{Y}_i)^2$. The total squared deviation of observations from fitted values. Also called RSS (Residual Sum of Squares).

**SSR (Sum of Squares Regression)**: $\sum(\hat{Y}_i - \bar{Y})^2$. The total squared deviation of fitted values from the mean. Measures how much variability the model explains.

**SST (Total Sum of Squares)**: $\sum(Y_i - \bar{Y})^2$. The total variability in the outcome. Identity: SST = SSR + SSE.

**Standardization ($z$-scoring)**: Transforming a variable to have mean 0 and standard deviation 1: $z = (x - \bar{x})/s_x$. Makes coefficients comparable across features with different scales.

**t-test (for regression)**: Tests whether an individual coefficient is significantly different from zero. Test statistic: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$, distributed as $t_{n-p-1}$ under the null.

**VIF (Variance Inflation Factor)**: Measures multicollinearity for predictor $x_j$: $\text{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ is from regressing $x_j$ on all other predictors. VIF > 10 indicates serious multicollinearity.

**WLS (Weighted Least Squares)**: A variant of OLS that assigns different weights to observations to handle heteroscedasticity. Minimizes $\sum w_i(Y_i - X_i^\top\beta)^2$.
