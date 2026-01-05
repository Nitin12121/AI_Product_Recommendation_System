# AI_Product_Recommendation_System
## Overview
This project implements an AI-based product recommendation system using:
- User clustering (K-Means)
- User-based collaborative filtering
- Cosine similarity

## Dataset
- User ratings: 200+
- Products: 20+
- Ratings matrix generated from user-product interactions

## Methodology
1. Data preprocessing
2. User clustering using K-Means
3. Similarity computation using cosine similarity
4. Personalized recommendation generation

## Output
- Cluster visualization of users
- Top-N product recommendations for a given user
- Recommendations saved as CSV

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
```bash
pip install -r requirements.txt
python user_clustering.py
python recommendation_output.py
