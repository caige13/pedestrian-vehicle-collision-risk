# Project Proposal

## Predicting Pedestrian–Vehicle Near-Collision Risk Using Engineered Trajectory Features

**[Your Name]**  
MS in Artificial Intelligence  
The University of Texas at Austin

---

## 1. Topic Summary

Pedestrian–vehicle collision risk prediction is an important problem in intelligent transportation systems and autonomous vehicle research. Many existing approaches rely on raw video, image-based object detection, or deep neural networks that require large labeled datasets and substantial computational resources. However, many near-collision situations can be described through geometry alone: two agents moving through shared space on converging paths.

This project investigates whether **engineered geometric trajectory features** derived from short-term motion histories can outperform a **naive proximity baseline** for classifying pedestrian–vehicle near-collision risk in urban environments. Rather than relying only on distance thresholds, this work uses interpretable vector-based features such as relative velocity, angular divergence, cosine similarity, and time-to-collision to model risk.

The goal is not to predict true real-world collisions directly, since the selected dataset does not contain explicit collision labels, but rather to predict **rule-defined near-collision risk events** based on trajectory geometry. This framing allows the project to remain methodologically sound while evaluating whether richer motion features provide better predictive power than simple proximity alone.

---

## 2. Research Questions

### Primary Research Question
Do engineered geometric trajectory features outperform a naive proximity baseline for classifying pedestrian–vehicle near-collision risk?

### Secondary Research Questions
1. Which engineered trajectory features contribute most to near-collision risk classification?
2. How do classical machine learning models compare in this setting, including logistic regression, random forest, and gradient-boosted trees?
3. How does the temporal observation window affect performance and prediction lead time?

---

## 3. Data

The primary dataset for this project will be the **Stanford Drone Dataset (SDD)**. SDD contains top-down aerial video collected over a university campus and provides tracked bounding box annotations for multiple agent types, including pedestrians, cars, bicyclists, and others. These annotations include frame number, track ID, bounding box coordinates, and class label, making it possible to reconstruct agent trajectories over time.

From the raw annotations, I will compute center-point trajectories for pedestrians and vehicles across frames. Using these trajectories, I will construct pedestrian–vehicle pairs and derive geometric features from short temporal windows.

Because SDD does not provide explicit collision or near-collision labels, I will define a **rule-based near-collision labeling scheme**. A pair will be labeled as at risk if, within a future prediction horizon, the pedestrian and vehicle satisfy both:
- a minimum distance threshold, and
- a convergence condition indicating that they are moving toward a potentially conflicting path rather than away from one another.

This label definition is central to the project and will be clearly documented as part of the methodology.

A secondary dataset such as **JAAD (Joint Attention in Autonomous Driving)** may be discussed in related work for context, but the experimental analysis will focus on **SDD only** in order to keep the project scope manageable.

---

## 4. Features

The project will use engineered geometric trajectory features computed from pedestrian–vehicle pairs over short windows of preceding frames. Candidate features include:

- **Euclidean distance** between pedestrian and vehicle
- **Relative velocity magnitude**
- **Relative velocity direction**
- **Angular divergence** between motion vectors
- **Cosine similarity** between heading directions
- **Time-to-collision (TTC)** under a constant-velocity assumption
- **Distance trend** over recent frames
- **Projected path distance** or scalar projection of one agent’s position onto the other’s motion vector

These features are intended to capture not just whether two agents are near one another, but whether they are moving in a way that suggests rising collision risk.

---

## 5. Baseline and Models

### Naive Proximity Baseline
The naive baseline will classify a pedestrian–vehicle pair as at risk whenever the Euclidean distance between them falls below a fixed threshold. This baseline ignores motion direction, speed, and path convergence.

Formally:

\[
\hat{y} =
\begin{cases}
1 & \text{if } d(p, v) < d_{\text{thresh}} \\
0 & \text{otherwise}
\end{cases}
\]

where \(d(p, v)\) is the distance between the pedestrian and vehicle, and \(d_{\text{thresh}}\) is a chosen threshold. Multiple thresholds may be tested, with the strongest threshold-only result used as the primary baseline.

### Machine Learning Models
I plan to train and compare the following models:

1. **Logistic Regression**  
   A simple interpretable baseline for understanding linear relationships in the engineered features.

2. **Random Forest**  
   A non-linear ensemble model that can capture feature interactions.

3. **Gradient Boosted Trees (XGBoost or similar)**  
   The primary model for this project, expected to perform well on structured tabular features.

A sequential deep learning model such as an LSTM may be considered only if time permits, but the main scope of the project will remain centered on classical machine learning using engineered features.

---

## 6. Approach

The experimental workflow will consist of the following stages:

### 1. Data Preprocessing
- Extract pedestrian and vehicle trajectories from SDD annotations
- Compute center points from bounding boxes
- Construct candidate pedestrian–vehicle pairs within a spatial filtering window

### 2. Label Construction
- Define rule-based near-collision risk labels using minimum future distance and convergence criteria
- Assign binary labels for supervised classification

### 3. Feature Engineering
- Compute trajectory-based geometric features over temporal windows such as 5, 10, or 15 preceding frames
- Aggregate features if needed across the observation window

### 4. Model Training
- Train the naive proximity baseline
- Train logistic regression, random forest, and gradient-boosted tree models
- Compare performance across multiple window lengths

### 5. Evaluation
Models will be evaluated using:
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

Because near-collision events may be relatively rare, I expect **precision, recall, and F1 score** to be more informative than accuracy alone.

Feature importance methods such as permutation importance or SHAP may also be used to determine which geometric features are most predictive.

---

## 7. Expected Contribution

This project aims to show whether **interpretable, engineered trajectory features** can provide a practical and computationally efficient alternative to proximity-only heuristics for near-collision risk classification.

The contribution is not a new dataset or a claim to predict actual accidents, but rather:

1. A reproducible framework for defining near-collision risk labels from trajectory geometry
2. A comparison between simple threshold-based risk detection and feature-based machine learning models
3. An analysis of which trajectory features are most useful for classifying near-collision risk

This makes the project both technically feasible and relevant to broader questions in transportation safety and motion-based risk prediction.

---

## 8. Questions for Instructors

I would appreciate guidance on the following points:

1. **Label definition:**  
   Since SDD does not contain explicit collision labels, is it acceptable to define near-collision risk using geometric criteria such as minimum future distance and motion convergence?

2. **Project scope:**  
   Is a comparison among classical machine learning models sufficient for the depth expected in the course, or would a deep learning component be expected?

3. **Dataset scope:**  
   Is it appropriate to focus only on SDD for the experiments, while discussing other datasets such as JAAD in related work rather than including them in the empirical evaluation?

---

## 9. Preliminary Hypothesis

I hypothesize that engineered geometric trajectory features will outperform a naive proximity baseline because they capture whether two agents are moving toward a conflict point, rather than merely being close together. In particular, features such as relative velocity, angular divergence, and time-to-collision should improve classification by reducing false positives from nearby agents that are not actually on converging paths.