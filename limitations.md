# Section 5.8 — Limitations (expanded replacement)

> **Note for María.** This is a proposed expansion of the existing Section 5.8 in the Discussion chapter. Your current version is a single paragraph covering cross-sectional design, external validation, and pre-pandemic timing. This draft expands it into seven subsections (5.8.1 through 5.8.7) that proactively address the weaknesses a thesis examiner or investor is most likely to probe. It's written to match your existing voice — academic but flowing, em-dash driven, confident but measured, grounded in citations. Feel free to edit, cut, or reorder.
>
> Citations marked with ⚠ are suggestions of established references in the relevant fields; verify them before including (they are real papers but you should confirm access and exact phrasing). Others are your existing citations.

---

## 5.8 Limitations

The breadth of this project — spanning predictive modelling, explainability analysis, and the construction of a deployment-ready support tool — makes an honest accounting of its constraints particularly important. The limitations outlined here are not afterthoughts but constitutive features of the work that shape what its findings can and cannot support. They fall into seven categories: construct validity, external validation, selection effects, causal interpretation, subgroup robustness, implementation fidelity, and scope.

### 5.8.1 Construct Validity: What the Models Actually Predict

A fundamental limitation of this work is that the target labels for all four outcomes were themselves derived from self-report screening instruments — the PHQ-9 for depression, the MBI-SS for burnout, the STAI for anxiety, and single-item questions for suicidal ideation. The models therefore predict screening positivity, not clinically verified diagnosis. The reported AUCs describe how well the feature set reproduces what another screener would conclude, not how well it identifies students who would meet diagnostic criteria upon structured clinical interview.

This distinction matters because the correspondence between screener output and clinical diagnosis is imperfect. Meta-analytic evidence on the PHQ-9, for example, indicates that at a cut-off of ≥10 the instrument yields pooled sensitivity around 0.85 and specificity around 0.85 against semi-structured clinical interviews (Levis et al., 2019 ⚠), meaning roughly 15% of positive screens do not meet diagnostic criteria and some portion of true cases are missed. The MBI-SS, widely used for burnout in student populations, has been similarly criticised for conflating a construct with its measurement (Bianchi et al., 2015). The depression AUC of 0.947, while genuinely high by the standards of the mental health ML literature, should therefore be understood as measuring alignment with a screening proxy rather than with diagnostic ground truth. The interpretive ceiling for any such tool is the validity of the instruments used to label the training data.

None of this invalidates the work — screening-to-screening prediction is clinically useful, and it is how the overwhelming majority of mental health prediction research in non-clinical populations operates. But it bounds the strongest claim the thesis can make: MedMind is a tool for identifying students likely to screen positive on standard instruments, which is a weaker claim than identifying students with the underlying conditions those instruments attempt to measure.

### 5.8.2 External Validation and Generalisation

The models were trained, cross-validated, and tested on a single dataset (DABE 2020) collected at a single point in time before the COVID-19 pandemic. Cross-validation tests robustness to sampling variation within a dataset; it does not test generalisation across distributions. No institution-level hold-out was performed, which means the reported performance could partly reflect features idiosyncratic to the participating universities — cohort sizes, curriculum structures, regional demographics — that may not transfer to other Spanish medical schools, let alone to medical students in other countries.

The pre-pandemic timing compounds this concern. As Section 5.2 discussed, the post-COVID evidence base uniformly reports higher prevalence of all four outcomes, which means a model calibrated on 2020 data will likely underestimate absolute risk in 2024–2026 populations. The relative importance of risk factors may remain stable — social support, life events, and academic satisfaction are plausibly stable predictors across time — but threshold-dependent decisions will require recalibration on current data before any operational deployment.

The data collection framework incorporated into the institutional hub of the MedMind application is a structural response to this limitation: it provides a standardised protocol for universities to contribute anonymised screening data, potentially enabling the external validation that this thesis could not perform. That functionality is aspirational until such data is actually collected, but it acknowledges that the present work represents a single-site study whose findings require independent replication before operational claims can be made.

### 5.8.3 Selection and Sampling Constraints

DABE 2020 relied on voluntary participation by students who chose to respond to a mental health survey. This introduces two distinct selection concerns. Students experiencing acute mental health crises may have been systematically less likely to complete a lengthy questionnaire, creating a floor effect that suppresses the observed prevalence of the most severe outcomes. Conversely, students with strong mental health interest may have been overrepresented, introducing unknown confounding with the features the models rely on. The direction and magnitude of these biases cannot be estimated without reference to an independent source of ground truth.

The cross-sectional design imposes a separate constraint on interpretation. The mid-degree vulnerability window observed for depression — peaking in Years 3 and 4 — is presented throughout this thesis as a developmental pattern, but the underlying data cannot definitively distinguish between genuine within-person deterioration as students progress through training and cohort effects reflecting differences between entry years. Longitudinal data following the same students across years is the only evidence that could definitively resolve this question, and collecting such data was beyond the scope of this project.

### 5.8.4 Causal Inference and the Intervention Recommender

The SHAP analysis identifies features that carry predictive weight within the trained models. It does not identify causal mechanisms, and the distinction matters acutely when the outputs are used to recommend interventions. The institutional intervention recommender, one of the central contributions of the deployed tool, ranks interventions by combining SHAP-derived feature importance with cost-benefit estimates — a useful decision-support heuristic, but one that rests on the assumption that correlational leverage translates to intervention leverage. This assumption may not hold.

A student who reports low social support and screens positive for depression may not be helped by a peer mentorship programme if their low social support is itself a symptom of the underlying depression — social withdrawal rather than social deficit. Addressing the statistical relationship without addressing its mechanistic origin can produce interventions that appear well-targeted on paper but fail in practice. This is a known hazard when machine learning outputs are used to inform policy recommendations (Pearl, 2009 ⚠; Molnar, 2022 ⚠), and it bears particularly on this work because the recommender's persuasive apparatus — its explicit rankings, cost estimates, and return-on-investment figures — invites stakeholders to read its outputs as evidence-based prescriptions rather than as hypothesis-generating starting points.

A related concern is that the cost estimates underlying the recommender's ROI calculations are themselves heuristic. Figures such as "€3.50 per student for peer mentorship coordination" were derived from plausible assumptions rather than from validated cost-effectiveness studies, and the ROI multipliers should be understood as illustrative rather than definitive. Peer mentorship in student populations has some RCT evidence supporting its efficacy (Dyrbye et al., 2019); career counselling has considerably less; protected rest policies are supported mostly by observational data. The honest framing is that the recommender suggests where to look first, not what will work. Future versions should either ground the cost-effectiveness estimates in published evidence where available or explicitly label the outputs as illustrative starting points for discussion rather than validated recommendations.

### 5.8.5 Subgroup Performance and Algorithmic Fairness

Section 5.6 documented a substantial gender gap in recall for suicidal ideation, with female students markedly under-detected despite a larger training sample. This finding was surfaced through deliberate subgroup evaluation, but the audit was not comprehensive. Performance was not systematically disaggregated across course year, sexual orientation, socioeconomic proxies, or ethnicity — categories for which the training data is either small, imbalanced, or absent altogether. The LGBTQ+ subgroup, which features prominently in the SHAP analysis as a risk factor for suicidal ideation, is small enough that performance estimates within it would be highly uncertain. A full fairness audit addressing demographic parity, equal opportunity across subgroups, and calibration within subgroups was not conducted.

This gap matters because algorithmic bias in mental health prediction has direct consequences for equitable detection. A tool that reliably identifies depression risk in majority students while systematically missing minority students reproduces and may amplify existing inequities in service access. The gender-stratified modelling proposed in Section 5.9 for suicidal ideation is a necessary first step; a more complete subgroup audit — including disaggregated confusion matrices, threshold analysis, and calibration plots for each identifiable subgroup — should precede any real-world deployment. The absence of such an audit in this thesis is not a design choice but a scope constraint that future versions of this work must address.

### 5.8.6 Implementation Limitations of the Support Framework

The MedMind application itself carries implementation constraints that separate it from a production-ready system. The LLM-assisted chat agent (MindGuide) uses a general-purpose model (GPT-4o-mini) with a constrained system prompt, but no clinician is in the loop, responses are not reviewed, and the underlying model is known to produce confident-sounding but inaccurate outputs in edge cases. A rule-based fallback is provided for keyword-triggered crisis responses, but the safety architecture rests on prompt engineering and soft escalation protocols rather than clinical oversight. This is appropriate for a research prototype demonstrating feasibility; it would not be appropriate for deployment to students in acute distress without additional safeguards including clinician escalation pathways, conversation monitoring, and explicit disclaimers about the non-therapeutic nature of the interaction. The fact that conversations pass through a third-party API (OpenAI) also introduces a privacy consideration that would require contractual and technical mitigation in a real deployment.

The pseudonym-based persistence system enables the longitudinal tracking features that make the tool useful beyond a single session, but it provides no authentication in any rigorous sense. A forgotten username cannot be recovered, and two students selecting the same pseudonym would overwrite each other's data. For a small pilot at a single institution this is a manageable risk; at scale it would require proper authentication infrastructure, encrypted storage at rest, and GDPR-compliant data governance that is beyond the scope of this thesis.

Finally, the work contains no deployment data. No students have used the tool outside development testing. There are no engagement metrics, no adoption rates, no qualitative feedback from intended users. The evaluation of the support framework is therefore limited to its design rationale and its technical functioning, not its real-world effectiveness. A pilot study at a participating institution — collecting even basic usage data and structured user feedback — would transform the contribution of this work from a theoretical demonstration into an empirical one, and is the most valuable single next step for establishing whether the tool delivers on its intended value.

### 5.8.7 Scope

The outcomes addressed by this work — depression, burnout, anxiety, and suicidal ideation — represent the most widely studied dimensions of medical student mental health but exclude clinically relevant conditions including eating disorders, post-traumatic stress, substance use disorders, and sleep disorders. The population is Spanish medical students at 43 universities during pre-clinical and clinical training, a meaningful but partial slice of a broader healthcare trainee population. Generalisation to dental, nursing, pharmacy, or veterinary students, to post-MIR resident physicians, or to medical students outside Spain is not demonstrated and should not be assumed. The MIR preparation hub included in the application draws on Spanish-specific examination structures and support resources that would not transfer directly to other national contexts without substantial adaptation.

---

## Proposed light extension to Section 5.9 (Future Work)

The existing Future Work section can remain largely as written. The additional limitations surfaced above suggest three additions to the list of priority next steps:

**A comprehensive algorithmic fairness audit** disaggregating model performance across gender, sexual orientation, course year, and available socioeconomic indicators, reported using standard fairness metrics (demographic parity, equal opportunity, calibration within subgroups). This would extend the gender-stratified analysis documented in Section 5.6 into a full audit of the kind increasingly expected for ML systems deployed in health-adjacent contexts.

**Cost-effectiveness grounding for the intervention recommender.** The heuristic cost estimates and ROI calculations used in the current implementation should be replaced with figures drawn from published cost-effectiveness studies where available, and explicitly labelled as illustrative where not. Collaboration with health economists would substantially strengthen the credibility of the recommender's outputs.

**A prospective pilot deployment** at a single Spanish medical school, with the primary aim of collecting engagement data, qualitative user feedback, and preliminary evidence on whether the screening tool's outputs influence help-seeking behaviour. Even a modest pilot (n = 50 to 200 students over one academic semester) would produce the first empirical evidence on real-world usability and move the work from a design contribution to an evaluable intervention.

---

## References to verify / add to bibliography

The following citations are suggested to support the expanded Limitations section. Verify each against your existing bibliography and access to the sources before including:

- **Levis, B., Benedetti, A., & Thombs, B. D. (2019).** Accuracy of Patient Health Questionnaire-9 (PHQ-9) for screening to detect major depression: individual participant data meta-analysis. *BMJ, 365*, l1476. — Widely cited meta-analysis documenting the gap between PHQ-9 screening and structured clinical interview diagnosis.

- **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. — The canonical reference for the distinction between correlational and causal inference in machine learning contexts.

- **Molnar, C. (2022).** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). — Openly available online. Covers SHAP interpretation and the limits of feature-importance-based policy recommendations.

- **Dyrbye, L. N., West, C. P., Satele, D., et al. (2019).** A longitudinal study exploring learning environment culture and subsequent risk of burnout among resident physicians overall and by gender. *Mayo Clinic Proceedings, 94*(8), 1509–1523. — You likely already cite Dyrbye elsewhere; this specific paper supports peer mentorship evidence claims.

- **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. — The foundational SHAP paper, if you want to cite it directly when discussing SHAP's interpretive scope.

---

## Quick notes on defending these limitations

If you want to use this section strategically in your defence, here are brief talking points for the three hardest questions an examiner is likely to ask:

**"How do you respond to the circular validation concern — your targets are themselves screeners?"**
Acknowledge it directly. The models predict screening positivity, not clinical diagnosis. The reported AUCs describe alignment with a screening proxy, which is how almost all mental health prediction research in non-clinical populations operates. The value of the work is in the explainability layer and the pipeline to action, not in claiming diagnostic accuracy. You address this in Section 5.8.1.

**"Your intervention recommender presents SHAP leverage as if it were causal. Isn't that overreach?"**
Yes, and you address it explicitly in Section 5.8.4. The recommender is designed as a hypothesis-generating decision-support tool, not as an evidence-based prescription. The cost-effectiveness estimates are heuristic and should be grounded in RCT evidence in future versions. The explicit framing in the UI could also be softened — this is a fair point and you can commit to revising the language before any real deployment.

**"You documented a gender gap in suicidal ideation recall. Why didn't you do a full fairness audit?"**
Scope. The gender gap was surfaced through deliberate subgroup evaluation, and Section 5.6 documents it honestly. A full audit across all identifiable subgroups was not possible within the time and data constraints of the project, and the most important gap — gender — is the one you surfaced and proposed mitigation for. Section 5.9 now includes a comprehensive fairness audit as a priority next step. This is a limitation you own rather than one an examiner has to discover.
