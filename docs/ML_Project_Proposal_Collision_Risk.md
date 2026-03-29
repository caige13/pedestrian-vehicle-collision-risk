# Project Proposal

## Predicting Pedestrian–Vehicle Collision Risk Using Trajectory Vector Features

*[Your Name] · MS in Artificial Intelligence · The University of Texas at Austin*

---

### 1. Topic Summary

Pedestrian–vehicle collision prediction is a critical problem in intelligent transportation systems and autonomous vehicle research. Most existing approaches rely on image-based object detection or deep learning over raw video, requiring substantial computational resources and large labeled datasets. However, the underlying geometry of collision events—two agents moving through shared space on converging paths—can be captured through trajectory-level vector features without requiring pixel-level visual processing.

This project investigates whether geometric vector features derived from short-term trajectory data can effectively classify near-collision risk between pedestrians and vehicles in urban environments. Rather than relying on proximity alone (e.g., simple distance thresholds), this work proposes a feature engineering approach grounded in vector mathematics—including relative velocity vectors, angular divergence between motion headings, cosine similarity of trajectory directions, and time-to-collision projections—to train supervised machine learning classifiers.

### 2. Research Questions

**Primary:** Can geometric trajectory vector features outperform naive proximity-based thresholds in predicting near-collision events between pedestrians and vehicles?

**Secondary:** (1) Which vector-derived features contribute most to collision risk classification? (2) How does prediction accuracy vary across model complexity, from logistic regression baselines to ensemble methods? (3) What temporal observation window (number of preceding frames) yields the best tradeoff between prediction lead time and accuracy?

### 3. Data

**Stanford Drone Dataset (SDD)** will serve as the primary data source. SDD provides top-down aerial video of a university campus with bounding box annotations for over 20,000 tracked agents—including approximately 11,200 pedestrians, 1,300 cars, 6,400 bicyclists, and additional agent types—across eight distinct scenes at 30 frames per second. Each annotation includes a track ID, bounding box coordinates, frame number, and agent class label, enabling extraction of per-agent trajectory sequences.

From these raw annotations, I will compute center-point trajectories for each agent per frame and derive the following vector features for each pedestrian–vehicle pair within a spatial proximity window:

**Relative velocity vector** (magnitude and direction of closing speed); **Angular divergence** (angle between the two agents' heading vectors via dot product); **Cosine similarity** (alignment of trajectory directions); **Time-to-collision (TTC)** (projected time until minimum distance assuming constant velocity); and **Distance projection onto path** (scalar projection of one agent's position onto the other's trajectory vector).

Since SDD does not include pre-labeled collision events, I will define a collision risk labeling scheme based on minimum distance thresholds combined with converging trajectory criteria. This formalization of what constitutes a near-collision event will itself be a methodological contribution discussed in the paper.

The **Joint Attention in Autonomous Driving (JAAD)** dataset may be used as a secondary reference for discussion, as it provides dashcam-perspective pedestrian behavioral annotations (crossing intent, attention state) across 346 video clips, offering a complementary viewpoint to SDD's aerial perspective.

### 4. Models and Approach

The experimental pipeline consists of three stages:

**Feature Engineering:** Extract trajectory center-points from SDD bounding boxes, compute per-pair vector features over sliding temporal windows (e.g., 5, 10, 15 preceding frames), and generate the binary collision risk labels described above.

**Model Training:** Train and evaluate multiple classifiers of increasing complexity: (1) Logistic Regression as a linear baseline, (2) Random Forest for non-linear feature interactions, and (3) Gradient Boosted Trees (XGBoost) as the primary model. An optional LSTM component may be explored if time permits, to evaluate whether sequential modeling of trajectory windows improves over fixed-window feature aggregation.

**Evaluation:** Models will be compared against a naive proximity-threshold baseline (flagging all pairs within a fixed distance as at-risk). Metrics will include accuracy, precision, recall, F1 score, and AUC-ROC. Feature importance analysis (via permutation importance and SHAP values) will identify which geometric features contribute most to prediction quality.

### 5. Questions for Instructors

I would appreciate guidance on the following:

**1. Label definition:** Since SDD does not include explicit collision labels, I plan to derive them from geometric criteria (e.g., minimum pairwise distance below a threshold combined with converging headings). Is this approach acceptable, or would the instructors prefer I use a dataset with pre-existing collision annotations?

**2. Scope of model complexity:** I plan to focus on classical ML models (logistic regression, random forest, gradient boosting) with an optional LSTM extension. Is this sufficient depth for the project, or would the inclusion of a deep learning component be expected?

**3. Single vs. multiple datasets:** Is focusing exclusively on SDD appropriate, or should I aim to validate findings across a second dataset such as JAAD?
