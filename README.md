# 🍿 StreamAI — Multi-Platform Streaming Content Recommender

> **An AI-powered web application that analyzes 22,000+ titles across Netflix, Amazon Prime, Disney+, and Hulu to deliver intelligent content recommendations and competitive market insights.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

---

## 📌 Project Overview

**StreamAI** is a full-stack data science application that solves a real-world problem: *"I finished a great show — what should I watch next?"*

Unlike single-platform recommendation systems, StreamAI works **across all major streaming services simultaneously**. It uses **Natural Language Processing (NLP)** to understand the content of each title and find the most similar alternatives — regardless of which platform they live on.

The application features two core modules:
1. **AI Recommendation Engine** — Find content similar to any movie or TV show across 4 platforms
2. **Market Intelligence Dashboard** — Analyze content strategies, genre trends, and platform growth

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🔍 **Cross-Platform Search** | Recommends content from Netflix, Amazon Prime, Disney+, and Hulu in one place |
| 🤖 **NLP-Based Matching** | Uses TF-IDF vectorization + Cosine Similarity to find semantically similar content |
| 📊 **Market Analytics** | Interactive charts showing platform market share, annual content growth, and top genres |
| ⚡ **Optimized Performance** | On-demand cosine similarity via `linear_kernel` eliminates the need to store a full similarity matrix |
| 🎨 **Modern UI** | Glassmorphism design with smooth hover animations and a cinematic dark theme |
| 🚀 **Production-Ready** | Cached data loading with `@st.cache_resource` for fast, scalable performance |

---

## 🧠 How the Recommendation Engine Works

```
User selects a title
        ↓
TF-IDF Matrix lookup (pre-computed on app start)
        ↓
linear_kernel computes cosine similarity on-demand
        ↓
Top 5 most similar titles returned across all 4 platforms
        ↓
Results displayed with platform, year, duration & synopsis
```

**Why TF-IDF + Cosine Similarity?**
- The model transforms each title's description, genre tags, and metadata into a high-dimensional vector
- Cosine similarity measures the angle between vectors — titles with similar content cluster together
- `linear_kernel` is used instead of `cosine_similarity` for significantly faster computation on sparse matrices

---

## 📊 Market Intelligence Dashboard

The analytics page provides business-grade insights:

- **KPI Cards** — Total content count, Movies vs. TV Shows breakdown, number of platforms
- **Market Share Pie Chart** — Visual comparison of each platform's content library size
- **Annual Content Race (Area Chart)** — How each platform's content output has grown since 2010
- **Genre Treemap** — The 20 most dominant content categories across all platforms

---

## 🗃️ Dataset

The application merges and cleans four official Kaggle datasets:

| Dataset | Source | Size |
|---|---|---|
| `netflix_titles.csv` | Kaggle — Netflix Movies and TV Shows | ~8,800 titles |
| `amazon_prime_titles.csv` | Kaggle — Amazon Prime Movies and TV Shows | ~9,600 titles |
| `disney_plus_titles.csv` | Kaggle — Disney+ Movies and TV Shows | ~1,400 titles |
| `hulu_titles.csv` | Kaggle — Hulu Movies and TV Shows | ~3,000 titles |

> **Total: 22,000+ titles** after merging, cleaning, and deduplication.

The cleaned dataset and pre-computed TF-IDF matrix are serialized as `.pkl` files for instant loading at runtime.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive web UI |
| **ML / NLP** | scikit-learn (TF-IDF, linear_kernel) | Content vectorization & similarity |
| **Data** | Pandas, NumPy | Data processing & manipulation |
| **Visualization** | Plotly Express | Interactive charts and graphs |
| **Serialization** | Joblib | Model & matrix persistence |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/StreamAI.git
cd StreamAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> **Note:** The `movie_data.pkl` and `tfidf_matrix.pkl` files must be present in the root directory. These are generated from the data preprocessing notebook (`movie.ipynb`).

---

## 📁 Project Structure

```
StreamAI/
│
├── app.py                  # Main Streamlit application
├── movie.ipynb             # Data cleaning & TF-IDF preprocessing notebook
├── movie_data.pkl          # Cleaned & merged dataset (serialized)
├── tfidf_matrix.pkl        # Pre-computed TF-IDF sparse matrix
├── requirements.txt        # Python dependencies
│
└── data/
    ├── netflix_titles.csv
    ├── amazon_prime_titles.csv
    ├── disney_plus_titles.csv
    └── hulu_titles.csv
```

---

## 💡 Technical Highlights

- **Handling large sparse matrices** — Storing a full 22,000×22,000 cosine similarity matrix (~4GB) is impractical. This project uses `linear_kernel` for row-by-row on-demand computation, reducing memory usage by over 99%.
- **Multi-source data merging** — Cleaning and normalizing 4 different CSV schemas into a unified format required careful column alignment, null handling, and platform tagging.
- **Streamlit performance optimization** — `@st.cache_resource` ensures data is loaded once per session rather than on every user interaction, significantly reducing response time.
- **NLP feature engineering** — Combined title descriptions, genres, and metadata into a single text feature for more robust similarity matching.

---

## 🔮 Future Improvements

- [ ] Add user rating data (IMDb/TMDB API integration) for hybrid collaborative + content-based filtering
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces for public access
- [ ] Add poster images via TMDB API
- [ ] Implement genre-based and year-based filtering in the recommendation engine
- [ ] Add a "Trending Now" section based on recent additions

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ using Python, Streamlit, and scikit-learn*
