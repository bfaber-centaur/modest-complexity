# Role: R4 investigator, competing predictive routes

Directory `research/regimes/routes/`. Run IDs `r1-route-…`. Seeds 54000–54999 (PLAN §1). Label `r1-route-cloud-1`. Budget 25 min.

## Bobby's brief (verbatim)

> Question: When two predictive routes are available, which does training select, and can that dependence change?
>
> Task:
> - A designated rule such as parity of three input bits.
> - A separate shortcut bit whose correlation with the label is controlled.
> - Balanced labels and irrelevant input bits as needed.
>
> Design:
> - First vary shortcut reliability at fixed architecture and training-set size.
> - Include rule-only and shortcut-only controls.
> - Evaluate distributions where the shortcut remains reliable, becomes independent, or reverses.
> - Generate splits so duplicate inputs and correlated sampling cannot leak across them.
> - Where possible, pair underlying examples across conditions.
>
> Measurements:
> - Accuracy across evaluation environments.
> - Input interventions that isolate dependence on each route.
> - Changes in route dependence throughout training.
>
> First figure: reliance on the rule and shortcut over training, alongside accuracy.
>
> Do not assume which route is "simpler" in the model's representation. Width or longer training can be one adaptive follow-up.

No starting literature was given. If you look something up, it should resolve a concrete design question; record what you read.

## Coordinator's notes

These are things to consider, not requirements. Your QUESTION.md decides.

**Split by base, not by input**
- Let a *base* be the rule bits plus the irrelevant bits. With 3 rule bits and 6 irrelevant bits there are 512 bases; each becomes an input once the shortcut bit s is set.
- Split bases into train and held-out. Every held-out environment then uses only held-out bases, so no input, and no base under the other s value, crosses the split.
- Reliable, independent and reversed environments can each be built over every held-out base: s = y, both values of s, and s = 1 − y.

**Pairing across reliabilities**
- Draw one uniform u per training base with a fixed data seed. Base b is an exception (s ≠ y) when u_b is among the lowest (1 − ρ) fraction, balanced across the two label classes.
- Exceptions are then nested across ρ and the realized reliability is exact.
- Every ρ condition shares the same training bases, labels and initialization seeds. Only s differs on the exception bases.
- Report realized reliability, label balance and s balance per condition.

**Controls**
- Rule-only: s independent of y on the training bases (ρ = 0.5), or s held constant. Say which, because they differ: a constant bit carries no signal at all.
- Shortcut-only: the label is s and the rule bits are independent of it. This is a trainability reference for the shortcut route.
- A reliable-shortcut condition at ρ = 1 is the extreme of the sweep.

**A third route.** At ρ < 1 a network can fit the training set with the shortcut plus memorized exceptions and never learn parity. Training accuracy on the exception bases, compared with reversed-environment accuracy on held-out bases, separates "learned the rule" from "memorized the exceptions".

**Route-dependence interventions.** On held-out bases, at each logged step:
- the fraction of predictions that change when s alone flips;
- the fraction that change when one rule bit alone flips, which flips parity;
- the same for an irrelevant bit, as a baseline.

Logit-level versions avoid the threshold. These are input interventions, so they say what the function depends on, not how it is represented. Keep that wording.

**Outcomes worth distinguishing in advance**
- Shortcut-dominated at every ρ < 1 within budget.
- A reliability threshold below which the rule takes over.
- Early shortcut reliance that gives way to the rule later in training.
- Exception memorization without rule learning.

Note which of these depend on training length. That is the natural adaptive follow-up, together with width.

## First figure

Over training updates, for each ρ and for the controls:
- rule reliance and shortcut reliance (the intervention flip rates);
- held-out accuracy in the reliable, independent and reversed environments, next to train accuracy;
- three paired seeds as individual traces;
- capped and failed runs marked.
