import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics.pairwise import linear_kernel

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="Universal Streaming Guide",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS İLE MODERN ARAYÜZ (Glassmorphism) ---
st.markdown("""
<style>
    /* Arka Plan: Derin Uzay Teması */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(20, 20, 30) 90%);
        color: white;
    }
    
    /* Kart Yapısı (Glassmorphism) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.02);
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Büyük Metrik Yazıları */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #a0a0a0;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Film Kartları */
    .movie-card {
        background-color: #1a1a2e;
        border-radius: 15px;
        overflow: hidden;
        height: 100%;
        border: 1px solid #333;
        transition: 0.3s;
    }
    .movie-card:hover {
        border-color: #e50914;
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.4);
    }
    .movie-content {
        padding: 15px;
    }
    
    /* Platform Etiketleri */
    .tag {
        font-size: 0.75rem;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. VERİ YÜKLEME (OPTİMİZE EDİLDİ) ---
@st.cache_resource
def load_data():
    try:
        df = joblib.load('movie_data.pkl')
        # DİKKAT: Artık TF-IDF Matrix yüklüyoruz (Cosine Sim değil)
        tfidf_matrix = joblib.load('tfidf_matrix.pkl')
        return df, tfidf_matrix
    except FileNotFoundError:
        return None, None

df, tfidf_matrix = load_data()

# --- 4. TAVSİYE FONKSİYONU (OPTİMİZE EDİLDİ - Linear Kernel) ---
def get_recommendations(title, tfidf_matrix=tfidf_matrix):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    try:
        idx = indices[title]
        if isinstance(idx, pd.Series): idx = idx.iloc[0]
        
        # HESAPLAMA BURADA ANLIK YAPILIYOR (RAM DOSTU)
        sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
        
        # Liste formatına çevir
        sim_scores = list(enumerate(sim_scores[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return df.iloc[movie_indices][['title', 'platform', 'description', 'release_year', 'duration', 'listed_in']]
    except:
        return None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/864/864808.png", width=80)
    st.title("StreamAI")
    st.markdown("Veri Destekli Yayın Rehberi")
    
    st.write("---")
    menu = st.radio("Menü:", ["📊 Pazar Analizi", "🔍 Film Önerisi Bul"])
    
    st.write("---")
    st.info("Bu proje 22,000+ içeriği analiz ederek size en doğru öneriyi sunar.")

# --- SAYFA: ANALİZ ---
if menu == "📊 Pazar Analizi":
    st.title("📈 Streaming Savaşları Raporu")
    st.markdown("Platformların içerik stratejilerini ve büyüme hızlarını analiz ettik.")
    
    if df is not None:
        # 1. KPI KARTLARI
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Toplam İçerik", f"{len(df):,}", col1),
            ("Film Sayısı", f"{len(df[df['type']=='Movie']):,}", col2),
            ("Dizi Sayısı", f"{len(df[df['type']=='TV Show']):,}", col3),
            ("Platformlar", "4 Dev", col4)
        ]
        
        for label, value, col in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # 2. GRAFİKLER
        c1, c2 = st.columns([1, 1])
        colors = {'Netflix': '#E50914', 'Amazon Prime': '#00A8E1', 'Disney+': '#113CCF', 'Hulu': '#1CE783'}

        with c1:
            st.subheader("🍰 Pazar Payı Dağılımı")
            fig_pie = px.pie(df, names='platform', 
                             color='platform',
                             color_discrete_map=colors,
                             hole=0.5)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("📅 Yıllık İçerik Yarışı")
            # HATA VEREN KISIM BURASIYDI - DÜZELTİLDİ
            yearly = df[df['release_year'] >= 2010].groupby(['release_year', 'platform']).size().reset_index(name='count')
            
            fig_area = px.area(yearly, x='release_year', y='count', color='platform',
                               color_discrete_map=colors)
            fig_area.update_layout(
                xaxis_title="Yıl", yaxis_title="Eklenen İçerik",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
        # 3. TREEMAP
        st.subheader("🧩 En Popüler Türler")
        df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0])
        genre_counts = df['main_genre'].value_counts().head(20).reset_index()
        genre_counts.columns = ['Tür', 'Sayı']
        
        fig_tree = px.treemap(genre_counts, path=['Tür'], values='Sayı',
                              color='Sayı', color_continuous_scale='Deep')
        fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_tree, use_container_width=True)

# --- SAYFA: TAVSİYE ---
elif menu == "🔍 Film Önerisi Bul":
    st.title("🍿 Yapay Zeka Tavsiye Asistanı")
    st.markdown("Hangi platformda olduğu önemli değil. Siz filmi söyleyin, biz benzerini bulalım.")
    
    selected_movie = st.selectbox(
        "🎬 İzlediğiniz ve sevdiğiniz bir yapım seçin:",
        df['title'].values,
        index=None,
        placeholder="Yazmaya başlayın... (Örn: Inception)"
    )
    
    if st.button("Benzerlerini Getir 🚀", type="primary", use_container_width=True):
        if selected_movie:
            with st.spinner('Analiz yapılıyor...'):
                recs = get_recommendations(selected_movie)
            
            if recs is not None:
                st.markdown("### ✨ Sizin İçin Seçtiklerimiz")
                st.write("")
                
                cols = st.columns(5)
                for i, (idx, row) in enumerate(recs.iterrows()):
                    p_color = "#E50914"
                    if "Amazon" in row['platform']: p_color = "#00A8E1"
                    elif "Disney" in row['platform']: p_color = "#113CCF"
                    elif "Hulu" in row['platform']: p_color = "#1CE783"
                    
                    with cols[i]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <div style="height: 5px; background-color: {p_color}; width: 100%;"></div>
                            <div class="movie-content">
                                <span class="tag" style="background-color: {p_color}; color: white;">{row['platform']}</span>
                                <h4 style="margin: 10px 0; min-height: 50px; font-size: 1rem;">{row['title']}</h4>
                                <p style="font-size: 0.8rem; color: #aaa;">📅 {row['release_year']} | ⏳ {row['duration']}</p>
                                <p style="font-size: 0.8rem; color: #ccc; height: 80px; overflow: hidden; text-overflow: ellipsis;">{row['description'][:100]}...</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Veri bulunamadı.")