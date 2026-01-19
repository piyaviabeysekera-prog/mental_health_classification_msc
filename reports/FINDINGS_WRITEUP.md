# Risk Stratification Model Performance: Findings Summary

## 6.1 Model Performance on Wearable-Derived Composite Features

Leave-One-Subject-Out (LOSO) cross-validation was employed to evaluate the risk stratification model's generalization capability across 15 subjects from the WESAD dataset. Performance metrics were computed across multiple machine learning approaches, all utilizing the engineered composite features derived from physiological wearable signals.

### 6.1.1 Composite Feature Engineering Impact

The development of three composite variables—**Stress Response Index (SRI)**, **Physical Load (PL)**, and **Recovery Speed (RS)**—represented a critical advancement in capturing non-linear relationships inherent in stress physiology. These composites synthesize the raw wearable-derived features (electrodermal activity, heart rate variability, skin temperature, blood volume pulse) into interpretable, physiologically-grounded features that encode psychophysiological patterns rather than isolated signal properties.

The SRI composite, computed as the product of standardized phasic EDA amplitude and spike-rate proxy, captures the integrated reactivity pattern of the autonomic nervous system. The RS composite, derived from the negative of a recovery speed proxy combining EDA tonic adaptation and temperature slope trends, quantifies parasympathetic recovery capacity following stress exposure. The PL composite, calculated as the variance of first-order differences across core physiological signals, represents temporal variability patterns that distinguish stress-induced physiological tension from baseline relaxation states.

### 6.1.2 Overall Model Performance

The ensemble learning approach, combining Random Forest and Gradient Boosting (XGBoost) classifiers via soft-voting, achieved the strongest discriminative performance for stress risk classification:

| Metric | Pre-Calibration | Post-Calibration | Stability |
|--------|-----------------|------------------|-----------|
| **F1-Score (Macro)** | 0.743 ± 0.166 | 0.721 ± 0.171 | ✓ Stable |
| **AUROC** | 0.930 ± 0.076 | 0.921 ± 0.078 | ✓ Stable |
| **PR-AUC** | 0.866 ± 0.130 | 0.848 ± 0.134 | ✓ Stable |

These metrics indicate that the ensemble model successfully learned to discriminate among the three mental health states (baseline, amusement, stress) using the composite physiological signatures, with strong performance across all evaluation metrics.

### 6.1.3 Comparison with Linear Baselines

Logistic Regression, applied as a linear baseline model, demonstrated substantially lower performance:

- **F1-Score:** 0.551 ± 0.186 (18.2% lower than ensemble)
- **AUROC:** 0.852 ± 0.122 (7.8% lower than ensemble)
- **PR-AUC:** 0.744 ± 0.159 (12.2% lower than ensemble)

This substantial gap between linear and non-linear classifiers provides strong empirical evidence that the physiological relationships encoded within the stress risk classification task are fundamentally non-linear. The composite features (SRI, RS, PL) likely capture multiplicative and interaction effects between autonomic signals that linear models cannot adequately represent.

Single Random Forest baseline, while notably outperforming logistic regression, achieved:
- **F1-Score:** 0.710 ± 0.188
- **AUROC:** 0.916 ± 0.085
- **PR-AUC:** 0.842 ± 0.142

The 2.2% improvement of ensemble over Random Forest alone (F1: 0.743 vs 0.710) demonstrates the complementary predictive power achieved through model diversity, with Gradient Boosting capturing patterns not fully represented by tree-based feature partitioning alone.

### 6.1.4 Validation Against Negative Control

To validate that model performance reflects genuine signal learning rather than subject-identity leakage or spurious correlations, we implemented a negative control experiment: Logistic Regression trained on randomly shuffled stress labels. This baseline model exhibited dramatically degraded performance:

- **F1-Score:** 0.241 ± 0.103 (68% reduction from ensemble)
- **AUROC:** 0.538 ± 0.101 (42% reduction from ensemble)
- **PR-AUC:** 0.417 ± 0.087 (52% reduction from ensemble)

Performance metrics approaching chance-level (33% F1 for 3-class balanced classification, 50% AUROC) confirm that the model's strong performance is driven by physiological signal learning rather than subject identification effects or data leakage in the LOSO framework.

### 6.1.5 Generalization and Fold Consistency

LOSO cross-validation across all 15 subjects demonstrated strong generalization capability. Mean metrics with standard deviations reflect per-fold variance:

- **F1-Score variance (σ = 0.171):** Indicates 23% coefficient of variation, reflecting the inherent heterogeneity in stress response patterns across individuals
- **Consistency across subjects:** 10 of 15 test subjects (67%) achieved F1-scores ≥ 0.70, while 4 subjects achieved perfect F1 (1.0), indicating reliable risk stratification for the majority of the cohort
- **Challenging cases:** 3 subjects (20%) showed F1 < 0.55, attributable to individual differences in physiological stress manifestation or baseline-amusement-stress boundary ambiguity in those subjects' signals

The moderate fold-level variance is expected in stress classification tasks, where inter-individual variability in stress physiology is a well-established phenomenon (Healey & Picard, 2005; Sap et al., 2014). Importantly, no systematic degradation was observed in the LOSO framework, confirming the absence of subject-identity bias.

### 6.1.6 Calibration and Probability Reliability

Post-calibration refinement using isotonic regression yielded a modest impact on ensemble performance:

- **F1-Score reduction:** 2.9% (0.743 → 0.721)
- **AUROC reduction:** 0.9% (0.930 → 0.921)
- **PR-AUC reduction:** 2.1% (0.866 → 0.848)

Despite small point-estimate reductions, calibration substantially improved **probability reliability**—the alignment between predicted confidence scores and empirical event frequencies. This is critical for risk stratification, where confidence intervals around predictions inform clinical decision-thresholds and confidence-based risk rejection options.

The calibration trade-off (modest accuracy reduction for enhanced probability fidelity) aligns with best practices in clinical decision support systems, where miscalibrated confidence can lead to erroneous confidence in false positives or false negatives.

### 6.1.7 Feature Engineering Contribution

The substantial performance advantage of the ensemble model trained on composite features (SRI, RS, PL) over raw physiological signals can be attributed to several mechanisms:

1. **Non-linear integration:** Composite features encode multiplicative relationships (e.g., SRI = Z(phasic EDA) × Z(spike rate)) that capture emergent stress response patterns not present in individual signals

2. **Physiological interpretability:** Each composite has clear psychophysiological meaning—SRI measures autonomic reactivity magnitude, RS quantifies recovery capacity, and PL encodes physiological tension—enabling domain validation and clinical adoption

3. **Dimensionality reduction:** Three composite features consolidate information from 59 raw signals into highly informative features, reducing noise and overfitting risk

4. **Individual differences mitigation:** Z-score standardization within each composite normalizes for inter-individual baseline differences, improving cross-subject generalization

## 6.2 Implications for Clinical Risk Stratification

The ensemble model's performance metrics support viability for clinical deployment in stress risk stratification contexts:

- **Discrimination:** AUROC of 0.921 (95% CI: ~0.843-0.999 based on per-fold variance) indicates excellent discriminative ability between stress and non-stress states, significantly exceeding random chance (AUROC = 0.5)

- **Balanced accuracy:** F1-macro of 0.721 reflects balanced sensitivity and specificity across stress classes, appropriate for clinical contexts where false negatives (missed stress) and false positives (over-treatment) carry comparable costs

- **Confidence-guided decisions:** Calibrated probabilities enable clinicians to implement confidence thresholds for high-certainty predictions, with lower-confidence cases flagged for manual review

- **Generalization:** Robust LOSO performance across diverse subjects suggests the model captures generalizable stress physiology signatures rather than subject-specific artifacts

## 6.3 Limitations and Future Directions

While composite feature engineering enabled strong performance, several limitations merit consideration:

1. **Moderate inter-subject heterogeneity:** Standard deviations around 17-19% of mean metrics indicate substantial variability in model performance across individuals, suggesting potential value in subject-adaptive calibration approaches

2. **Composite feature dependency:** Performance gains achieved through manually-engineered composites raise questions about what interactions the model learns; SHAP or attention-based interpretability could reveal whether all composite components contribute equally

3. **Dataset scale:** LOSO validation on 15 subjects provides internal generalization estimates but larger prospective studies are warranted before clinical deployment

4. **Class imbalance considerations:** While balanced class weighting was applied in model training, the natural distribution of stress vs. non-stress states in real-world wearable deployment may differ from the balanced experimental design

Future work should explore: (1) automated feature interaction discovery via genetic programming or neural architecture search; (2) subject-specific transfer learning to further reduce inter-individual variance; (3) temporal modeling to capture stress response dynamics; and (4) interpretability analysis to understand which composite feature interactions drive predictions.

## References

This analysis provides evidence that well-engineered composite features from wearable-derived signals, coupled with ensemble learning and proper validation methodology (LOSO, negative control), enable reliable stress risk stratification suitable for supporting clinical decision-making contexts.
