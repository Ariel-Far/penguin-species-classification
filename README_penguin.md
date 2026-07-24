# Penguin Species Classification — End-to-End ML Pipeline

**Course:** PIC 16A, UCLA · **Team of 3** · **Tools:** Python (pandas, scikit-learn, NumPy, matplotlib, seaborn)

A complete supervised-learning pipeline on the Palmer Penguins dataset: raw data through
cleaning, exploratory analysis, feature selection, and four classification models — ending
with a neural network implemented from scratch in NumPy.

**The constraint:** predict species using at most **two numeric features and one categorical
feature**. The question wasn't whether classification was possible — it was how much signal
survives a deliberately starved feature set, and what kind of model extracts the last of it.

---

## Pipeline

**1. Cleaning without contaminating.** Audited missingness column by column, dropped constant
and identifier columns, and deferred NaN removal until after feature selection so rows weren't
discarded over columns that would never be used. In the LDA and perceptron sections, raw data
is split *before* cleaning and encoding, so test-set information never informs a training-set
decision.

**2. Feature selection on evidence.** Ranked features by absolute correlation with species,
then inspected pairwise scatter plots and model decision regions to find the combination with
the least class overlap. Selected **Culmen Length, Culmen Depth, and Island**.

*Note on Island:* it correlates with species at 0.64 — higher than most physical measurements —
because Gentoo appear only on Biscoe and Chinstrap only on Dream. It is nearly a free label for
two of three classes. Correct to keep under the assignment's constraint, but the reason it works
is a fact about sampling, not about penguins, and it would not generalize to another archipelago.

**3. Model comparison.** Logistic regression, linear SVM, and LDA evaluated under **k-fold
cross-validation** with per-fold confusion matrices, plus repeated random train/test splits to
confirm accuracy was stable rather than lucky.

**4. Neural network — my contribution.** A multi-layer perceptron built from scratch in NumPy,
no deep-learning framework:

- Min–max feature scaling
- Hidden layer: 9 neurons, ReLU activation
- Output: softmax over 3 classes, cross-entropy loss
- Manually derived forward pass, backpropagation, and gradient descent
- Learning rate and epoch count selected by inspecting training-loss curves for convergence
  and diminishing returns

## Results

| Model | Boundary | Validation | Accuracy | Failure mode |
|---|---|---|---|---|
| Logistic Regression | Linear | 10-fold CV + random splits | ~97% | Adelie/Chinstrap overlap |
| Linear SVM | Linear | 10-fold CV | ~95% | Convergence, untuned C |
| Linear Discriminant Analysis | Linear | 20-fold CV | ~97–99% | Same overlap, fewer cases |
| **Perceptron (from scratch)** | **Non-linear** | Held-out test set | **~99%** | — |

All three linear models stalled around 97%, and the confusion matrices showed the same failure
each time: a small cluster of Adelie and Chinstrap birds overlapping in culmen space that no
straight line separates. The remaining error wasn't noise — it was the *shape of the boundary*.
The perceptron's curved decision regions recovered those cases.

Three independent algorithms failing on the same handful of birds is a diagnosis, not a
disappointment: it says the error is geometric, and predicts that a non-linear boundary should
fix it. Writing the network by hand meant the fix followed from the diagnosis rather than from
swapping models until a number improved.

## Contributions

Group project with two collaborators. Data cleaning, exploratory statistics, feature selection,
and the model-evaluation function were done jointly. **I implemented the perceptron model and
the scatter-plot analysis**; my teammates handled logistic regression, LDA, and accuracy testing.

## Files

- `Penguins_Classification_Perceptron.ipynb` — full analysis (cleaning → EDA → model comparison → perceptron)

## Run it

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
jupyter notebook Penguins_Classification_Perceptron.ipynb
```

---

*Data: Palmer Penguins (Horst, Hill & Gorman) — 344 observations, three species, Palmer Station LTER.*
