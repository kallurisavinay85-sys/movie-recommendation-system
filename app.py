import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite movie using content-based filtering.")

@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id"
    )

    movies.rename(
        columns={"title_x": "title"},
        inplace=True
    )

    movies = movies[
        [
            "id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_average",
            "vote_count"
        ]
    ]

    movies.dropna(inplace=True)

    def convert(obj):
        result = []

        for i in ast.literal_eval(obj):
            result.append(i["name"])

        return result

    def convert_cast(obj):
        result = []

        for i in ast.literal_eval(obj):
            result.append(i["name"])

            if len(result) == 3:
                break

        return result

    def fetch_director(obj):
        result = []

        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                result.append(i["name"])
                break

        return result

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["genres"] = movies["genres"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["keywords"] = movies["keywords"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["cast"] = movies["cast"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["crew"] = movies["crew"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    movies["tags"] = movies["tags"].apply(
        lambda x: " ".join(x)
    )

    movies = movies[
        [
            "id",
            "title",
            "tags",
            "vote_average",
            "vote_count"
        ]
    ]

    return movies


@st.cache_resource
def create_model(movies):

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = tfidf.fit_transform(movies["tags"])

    similarity = cosine_similarity(vectors)

    return similarity


movies = load_data()

similarity = create_model(movies)

movie_index = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

st.sidebar.header("🎥 Select a Movie")

movie = st.sidebar.selectbox(
    "Choose a movie:",
    sorted(movies["title"].tolist())
)

if st.sidebar.button("Recommend Movies", type="primary"):

    index = movie_index[movie]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    st.subheader(f"🎬 Movies similar to **{movie}**")

    for number, (i, score) in enumerate(movie_list, 1):

        title = movies.iloc[i]["title"]
        rating = movies.iloc[i]["vote_average"]
        votes = movies.iloc[i]["vote_count"]

        col1, col2, col3 = st.columns([5, 2, 2])

        with col1:
            st.markdown(
                f"### {number}. {title}"
            )

        with col2:
            st.metric(
                "⭐ Rating",
                f"{rating}/10"
            )

        with col3:
            st.metric(
                "🔗 Similarity",
                f"{score:.3f}"
            )

        st.caption(
            f"👥 TMDB Votes: {votes}"
        )

        st.divider()

else:

    st.info(
        "👈 Select a movie from the sidebar and click "
        "**Recommend Movies**."
    )

st.sidebar.markdown("---")

st.sidebar.write("### ⚙️ How it works")

st.sidebar.write(
    "This system uses TF-IDF vectorization and "
    "cosine similarity to find movies with similar "
    "content."
)