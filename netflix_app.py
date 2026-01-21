import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

USER_FILE = "users.csv"
DATA_FILE = "netflix_titles.csv"


def show_splash_screen():
    st.markdown("""
        <div style="text-align:center; padding-top: 50px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg" width="200"/>
        <h2 style="color: white; font-family: 'Helvetica Neue', sans-serif; margin-top: 30px;">Explore Titles, Discover Trends, Predict the Future.</h2>
        <p style="color: #ccc; font-size: 18px;">Welcome to Netflix Explorer – your gateway to intelligent entertainment analysis.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎮 Start Exploring", key="splash_continue"):
            st.session_state.show_splash = False
            st.rerun()


def apply_theme(theme):
    dark_theme = """
    <style>
    html, body, .stApp {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1, h2, h3 {
        color: #E50914;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
    }
    input, textarea {
        background-color: #222 !important;
        color: #ffffff !important;
    }
    </style>
    """

    light_theme = """
    <style>
    html, body, .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1, h2, h3 {
        color: #E50914;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
    }
    input, textarea {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
    }
    </style>
    """

    if theme == "Dark":
        st.markdown(dark_theme, unsafe_allow_html=True)
    else:
        st.markdown(light_theme, unsafe_allow_html=True)



def initialize_user_file():
    if not os.path.exists(USER_FILE):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USER_FILE, index=False)


def load_users():
    try:
        return pd.read_csv(USER_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["username", "password"])


def check_credentials(username, password):
    users = load_users()
    return not users[(users["username"] == username) & (users["password"] == password)].empty


def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame([{"username": username, "password": password}])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_FILE, index=False)
    return True


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


def show_dashboard(username):
    st.markdown(f"<h2>🎬 Welcome, {username}!</h2>", unsafe_allow_html=True)
    df = load_data()

    st.markdown("### 🎛️ Filters")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("🌍 Select Country", ["All"] + sorted(df["country"].dropna().unique()))
            genre = st.selectbox("🎭 Select Genre", ["All"] + sorted(df["listed_in"].dropna().unique()))
        with col2:
            dtype = st.selectbox("📽️ Select Type", ["All"] + sorted(df["type"].dropna().unique()))
            cast = st.text_input("👥 Enter Cast Name (partial or full)")

    filtered_df = df.copy()
    if country != "All":
        filtered_df = filtered_df[filtered_df["country"] == country]
    if genre != "All":
        filtered_df = filtered_df[filtered_df["listed_in"].str.contains(genre, na=False)]
    if dtype != "All":
        filtered_df = filtered_df[filtered_df["type"] == dtype]
    if cast:
        filtered_df = filtered_df[filtered_df["cast"].fillna("").str.contains(cast, case=False)]

    st.markdown("### 📄 Filtered Titles")
    st.dataframe(filtered_df)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "filtered_netflix.csv", "text/csv")

    st.markdown("### 📊 Content Type Distribution")
    type_counts = filtered_df["type"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    st.markdown("### 🌍 Top 10 Countries by Titles")
    country_counts = filtered_df["country"].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    ax2.bar(country_counts.index, country_counts.values, color="skyblue")
    plt.xticks(rotation=45)
    st.pyplot(fig2)


def show_recommendations():
    st.markdown("<h2>🎯 Netflix Recommendations</h2>", unsafe_allow_html=True)
    df = load_data().dropna(subset=["title", "description"])
    df["description"] = df["description"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["description"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    titles = df["title"].values
    title = st.selectbox("Select a Show Title", sorted(titles))
    if title:
        idx = df[df["title"] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        rec_indices = [i[0] for i in sim_scores]
        similar_titles = df["title"].iloc[rec_indices].values
        similarity_scores = [round(sim_scores[i][1], 3) for i in range(len(sim_scores))]

        fig, ax = plt.subplots()
        ax.barh(similar_titles[::-1], similarity_scores[::-1], color='skyblue')
        ax.set_xlabel("Cosine Similarity Score")
        ax.set_title("Top 5 Similar Titles")
        st.pyplot(fig)


def show_prediction():
    st.markdown("<h2>🔍 Predict Movie or TV Show</h2>", unsafe_allow_html=True)
    df = load_data()
    df = df.dropna(subset=["rating", "listed_in", "country"])
    df["is_movie"] = df["type"].apply(lambda x: 1 if x == "Movie" else 0)

    df["features"] = df["rating"] + " " + df["listed_in"] + " " + df["country"]
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(df["features"])
    y = df["is_movie"]

    model = RandomForestClassifier()
    model.fit(X, y)

    input_rating = st.text_input("🎫 Enter Rating (e.g., PG, TV-MA)")
    input_genre = st.text_input("🎭 Enter Genre (e.g., Drama, Comedy)")
    input_country = st.text_input("🌍 Enter Country (e.g., United States)")

    if st.button("🔍 Predict Type"):
        input_text = input_rating + " " + input_genre + " " + input_country
        input_vec = tfidf.transform([input_text])
        pred = model.predict(input_vec)[0]
        pred_label = "Movie" if pred == 1 else "TV Show"
        st.success(f"📽️ Predicted Type: {pred_label}")


def show_genre_prediction():
    st.markdown("<h2>🔮 Predict Genre from Description</h2>", unsafe_allow_html=True)
    df = load_data()
    df = df.dropna(subset=["description", "listed_in"])
    df = df[df["listed_in"].str.contains(",") == False]
    df = df.drop_duplicates(subset=["description", "listed_in"])

    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(df["description"])
    y = df["listed_in"]

    model = MultinomialNB()
    model.fit(X, y)

    user_input = st.text_area("📝 Enter a short description of the show/movie:")
    if st.button("🎯 Predict Genre"):
        if user_input.strip() != "":
            input_vec = tfidf.transform([user_input])
            pred = model.predict(input_vec)[0]
            st.success(f"🎭 Predicted Genre: {pred}")
        else:
            st.warning("⚠️ Please enter a valid description.")


def main():
    st.set_page_config(
        page_title="Netflix Explorer",
        page_icon="🎮",
        layout="wide"
    )

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"
    if "show_splash" not in st.session_state:
        st.session_state.show_splash = True

    if st.session_state.show_splash:
        apply_theme("Dark")
        show_splash_screen()
        return

    st.sidebar.markdown("🎨 **Theme Settings**")
    theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"], index=1)
    st.session_state.theme = theme
    apply_theme(theme)

    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg" width="50" style="margin-right: 10px;">
            <h1 style="color:#E50914; font-family:'Helvetica Neue', sans-serif;">Netflix Explorer</h1>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.logged_in:
        st.success(f"Welcome back, {st.session_state.username}!")

        tab_labels = [
            "📊 Dashboard",
            "🎯 Recommendations",
            "🔍 Type Predictor",
            "🎭 Genre Predictor"
        ]
        tabs = st.radio("Navigate", tab_labels, horizontal=True, label_visibility="collapsed")

        if tabs == "📊 Dashboard":
            show_dashboard(st.session_state.username)
        elif tabs == "🎯 Recommendations":
            show_recommendations()
        elif tabs == "🔍 Type Predictor":
            show_prediction()
        elif tabs == "🎭 Genre Predictor":
            show_genre_prediction()

        if st.sidebar.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.feedback_given = True
            st.rerun()

    elif st.session_state.feedback_given:
        st.markdown("<h2>📝 We value your feedback!</h2>", unsafe_allow_html=True)
        feedback = st.text_area("Please share your experience using Netflix Explorer:")
        if st.button("Submit Feedback"):
            st.success("✅ Thank you for your feedback!")
            st.session_state.feedback_given = False

    else:
        menu = ["Login", "Sign Up"]
        choice = st.selectbox("🔑 Select Option", menu)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if choice == "Login":
                st.markdown("### 🔐 Login to Your Account")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")

                if st.button("Login"):
                    if check_credentials(username, password):
                        st.success("✅ Logged in successfully!")
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

            else:
                st.markdown("### 👥 Create a New Account")
                username = st.text_input("Choose Username", key="signup_username")
                password = st.text_input("Choose Password", type="password", key="signup_password")

                if st.button("Register"):
                    if save_user(username, password):
                        st.success("✅ User registered! Please log in.")
                    else:
                        st.warning("⚠️ Username already exists.")

    st.markdown("""
        <hr style="margin-top: 3rem; margin-bottom: 1rem;">
        <p style='text-align: center; color: #aaa; font-size: 14px;'>
            Built with ❤️ using Streamlit
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()