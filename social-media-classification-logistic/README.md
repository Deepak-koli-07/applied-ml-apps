# Viral Content Prediction (Two-Stage ML)

Predict whether a social media post will go viral using a production-style ML workflow.

This project demonstrates a key industry insight:

- **Pre-post metadata alone is weak** for predicting virality.  
- **Early engagement signals make virality highly predictable.**

Hence, the system is designed as a **two-stage model**.

## Baseline Model
A simple rule-based baseline always predicts **“Not Viral”**.

- It correctly labels non-viral posts.
- It misses **all** viral posts.

This sets a clear bar: any ML model must outperform this rule to be useful.

## Stage A — Pre-Post Model
Uses only features available *before publishing*:
- platform, content_type, topic, language, region  
- time features (hour, day_of_week, is_weekend)  
- hashtag_count, sentiment_score  

Purpose: provide a weak prior for *“Should I tweak this before posting?”*  
Result: near-random performance (expected).

## Stage B — Post-Early Model
Uses early engagement signals:
- views, likes, comments, shares, engagement_rate  
+ all metadata features

Purpose: decide *“Should this post be boosted?”*  
Result: near-perfect separation of viral vs non-viral posts.

## What This Shows
Virality is driven primarily by **human reaction after posting**.  
A production system should therefore use:

1. Pre-post guidance (weak prior)  
2. Post-early decision engine (high confidence)
