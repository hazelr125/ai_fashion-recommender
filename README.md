# 👗 AI Fashion Recommender

An intelligent fashion recommendation system that segments mall customers using **K-Means Clustering** and ranks personalised product suggestions with an **A\* heuristic search** algorithm — inspired by Burberry's AI-driven retail experience.

---

## 📌 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Algorithms Used](#algorithms-used)
4. [Dataset](#dataset)
5. [Customer Segments](#customer-segments)
6. [Product Catalog](#product-catalog)
7. [Tech Stack](#tech-stack)
8. [Getting Started](#getting-started)
9. [Project Structure](#project-structure)
10. [Future Enhancements](#future-enhancements)

---

## Overview

This project builds an end-to-end AI pipeline that:

1. **Analyses** mall customer data (age, income, spending score, gender).
2. **Clusters** customers into 5 behavioural segments using K-Means.
3. **Recommends** the top 5 fashion products for each customer using A\* Search, balancing item popularity and price-tier alignment.
4. **Displays** results in an interactive Streamlit web app with live product images pulled from the Pexels API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Google Colab Notebook                    │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   EDA &      │    │  K-Means     │    │  Model Serialise  │  │
│  │  Visualise   │───▶│  Training    │───▶│  (pickle files)   │  │
│  └──────────────┘    └──────────────┘    └───────────────────┘  │
│                                                    │             │
│                                                    ▼             │
│                          ┌─────────────────────────────────┐    │
│                          │        app.py  (Streamlit)       │    │
│                          │                                  │    │
│                          │  ┌──────────┐  ┌─────────────┐  │    │
│                          │  │ K-Means  │  │  A* Search  │  │    │
│                          │  │ Predict  │─▶│  Ranker     │  │    │
│                          │  └──────────┘  └─────────────┘  │    │
│                          │         │                        │    │
│                          │         ▼                        │    │
│                          │  ┌─────────────────────────┐    │    │
│                          │  │  Pexels API  (images)   │    │    │
│                          │  └─────────────────────────┘    │    │
│                          └──────────────────┬───────────────┘    │
│                                             │                    │
│                          ┌──────────────────▼───────────────┐    │
│                          │    pyngrok public tunnel          │    │
│                          │    localhost:8501 → public URL    │    │
│                          └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Role |
|---|---|
| **Jupyter / Colab Notebook** | EDA, training, model export, app launch |
| **Mall_Customers.csv** | Source training data |
| **`StandardScaler`** | Feature normalisation before clustering |
| **K-Means model** (`kmeans_model.pkl`) | Assigns new customers to segments |
| **Segment metadata** (`.pkl` files) | Labels, descriptions, and product lists per segment |
| **`app.py` (Streamlit)** | Interactive UI — takes user inputs, runs both AI steps, shows results |
| **A\* Recommender** | Ranks products within a segment using a cost + heuristic function |
| **Pexels REST API** | Fetches live fashion images for each recommended product |
| **pyngrok** | Exposes the local Streamlit server as a publicly accessible URL from Colab |

---

## Algorithms Used

### 1. K-Means Clustering

**Purpose:** Group customers into behavioural segments based on their profile.

**How it works:**
- Features used: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`, `Gender_encoded`
- All features are normalised with `StandardScaler` to prevent income from dominating.
- The **Elbow Method** (plotting inertia for k = 2 … 10) determines the optimal number of clusters. For this dataset, **k = 5** provides a clean elbow.
- `KMeans(n_clusters=5, random_state=42, n_init=10)` is trained on the scaled data and persisted via pickle.

**Prediction at inference:**
```
User profile → StandardScaler.transform() → KMeans.predict() → Segment ID (0-4)
```

---

### 2. A\* Heuristic Search Recommender

**Purpose:** Within the assigned segment, rank products so the most relevant item surfaces first.

**Cost function:**

```
f(n) = g(n) + h(n)
```

| Term | Meaning | Formula |
|---|---|---|
| `g(n)` | Unpopularity cost — prefers popular items | `(100 − popularity) / 100` |
| `h(n)` | Price-tier mismatch heuristic — prefers items priced close to the user's income bracket | `abs(product_tier − customer_tier) × 0.3` |
| `f(n)` | Total score — **lower is better** | `g(n) + h(n)` |

**Price tier mapping:**

| Annual Income | Tier |
|---|---|
| < $30k | 1 (Budget) |
| $30k – $59k | 2 (Mid-range) |
| $60k – $89k | 3 (Premium) |
| ≥ $90k | 4 (Luxury) |

The algorithm inserts all candidate products into a **min-heap** keyed by `f(n)`, then pops the top 5 — an efficient O(n log n) ranked retrieval.

---

## Dataset

**Source:** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (`Mall_Customers.csv`)

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Customer age (years) |
| `Annual Income (k$)` | Annual income in thousands of USD |
| `Spending Score (1-100)` | Score assigned by the mall (1 = low, 100 = high) |

**Size:** 200 customers, no missing values.

---

## Customer Segments

| ID | Name | Icon | Profile | Fashion Strategy |
|---|---|---|---|---|
| 0 | Mid Income, Mid Spenders | 🛍️ | Average income & spending | Versatile mid-range everyday fashion |
| 1 | High Income, Low Spenders | 💼 | Earn well, spend cautiously | Premium quality, timeless classics |
| 2 | Low Income, Low Spenders | 💰 | Budget-conscious shoppers | Affordable basics and sale items |
| 3 | Low Income, High Spenders | ✨ | Low income but love to splurge | Trendy, aspirational pieces |
| 4 | High Income, High Spenders | 💎 | Top-tier customers | Exclusive luxury items and new arrivals |

---

## Product Catalog

Each segment carries 7 curated items with a **price tier** (1–4) and **popularity score** (0–100). Sample products:

| Segment | Sample Products |
|---|---|
| Mid Spenders | Classic Trench Coat, Slim Fit Chinos, Canvas Sneakers |
| High Income, Low Spend | Cashmere Sweater, Tailored Blazer, Leather Oxford Shoes |
| Budget Shoppers | Basic T-Shirt Pack, Denim Jeans, Casual Hoodie |
| Aspirational Splurgers | Sequin Evening Dress, Statement Handbag, Block Heel Boots |
| Luxury Customers | Heritage Trench Coat, Monogram Leather Bag, Designer Sunglasses |

---

## Tech Stack

| Library / Tool | Version | Purpose |
|---|---|---|
| Python | 3.x | Core language |
| pandas | latest | Data loading and manipulation |
| numpy | latest | Numerical operations |
| scikit-learn | latest | StandardScaler, KMeans |
| matplotlib / seaborn | latest | EDA visualisations |
| Streamlit | latest | Interactive web UI |
| pyngrok | latest | Public tunnel for Colab-hosted app |
| requests | latest | Pexels API calls |
| pickle | stdlib | Model serialisation |
| heapq | stdlib | Min-heap for A\* Search |

---

## Getting Started

### Prerequisites

```bash
pip install streamlit pyngrok scikit-learn pandas numpy matplotlib seaborn -q
```

### Running in Google Colab

1. Open `ai_minproj.ipynb` in Google Colab (GPU runtime recommended).
2. Place `Mall_Customers.csv` in the Colab working directory (or mount Google Drive).
3. Run all cells in order:
   - **Cells 0–1:** Install dependencies and import libraries.
   - **Cell 2:** Load and inspect the dataset.
   - **Cells 3–5:** Exploratory visualisations.
   - **Cell 6:** Feature engineering and scaling.
   - **Cells 7–8:** Elbow method + K-Means training.
   - **Cell 9:** Cluster visualisation.
   - **Cell 10:** Save models to `.pkl` files.
   - **Cell 11:** Write `app.py` (the Streamlit app).
   - **Cell 12:** Launch Streamlit + open a public ngrok URL.
4. Click the URL printed by Cell 12 to open the live app.

### Running Locally (outside Colab)

```bash
# After saving the .pkl files
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Using the App

1. Use the **left sidebar** to enter your profile (Age, Annual Income, Spending Score, Gender).
2. Click **🔍 Get Recommendations**.
3. View:
   - Your **customer segment** and its description.
   - **Top 5 recommended products** with live images, popularity scores, and A\* f(n) scores.
   - A **score table** explaining the ranking.
   - A **profile summary** and **decision explanation**.

---

## Project Structure

```
ai_fashion-recommender/
│
├── ai_minproj.ipynb          # Main notebook: EDA, training, app generation
├── Mall_Customers.csv        # Training dataset (must be provided)
│
│── (generated at runtime)
├── app.py                    # Streamlit web application
├── kmeans_model.pkl          # Trained K-Means model
├── scaler.pkl                # Fitted StandardScaler
├── segment_labels.pkl        # Human-readable segment names
├── segment_descriptions.pkl  # Segment strategy descriptions
└── segment_products.pkl      # Product lists per segment
```

---

## Future Enhancements

| Enhancement | Description |
|---|---|
| **Collaborative Filtering** | Incorporate user–item interaction history (purchase, clicks) to surface items liked by similar customers, moving beyond static segment catalogs. |
| **Content-Based Filtering** | Encode product attributes (colour, style, material, season) as feature vectors and recommend items similar to a user's past purchases. |
| **Hybrid Recommender** | Combine K-Means segmentation, A\*, and collaborative/content-based signals into a weighted ensemble for higher accuracy. |
| **Deep Learning Embeddings** | Use a neural network (e.g., two-tower model) to learn dense user and product embeddings, enabling more nuanced similarity matching. |
| **Real-Time User Feedback** | Allow users to rate recommendations (👍 / 👎) and update product popularity scores dynamically for continuous learning. |
| **Larger & Richer Dataset** | Replace the 200-row mall dataset with a real e-commerce dataset including transaction history, browse events, and return rates. |
| **Image-Based Fashion Matching** | Use a CNN (e.g., ResNet) to extract visual features from clothing images and recommend visually similar items. |
| **Seasonal & Trend Awareness** | Adjust recommendations based on the current season, trending styles, or real-time social media signals. |
| **User Authentication & Profiles** | Persist user profiles across sessions so recommendations improve over time for returning users. |
| **Cloud Deployment** | Deploy on Streamlit Community Cloud, Heroku, or AWS to eliminate the need for ngrok and Colab. |
| **A/B Testing Framework** | Run controlled experiments comparing different recommendation strategies to measure conversion improvements. |
| **Explainability Dashboard** | Visualise why a specific product was recommended (segment membership probability, f(n) score breakdown). |
| **Multi-Language Support** | Internationalise the UI for global retail markets. |

---

## License

This project is for educational / demonstration purposes. The Mall Customers dataset is publicly available on Kaggle.
