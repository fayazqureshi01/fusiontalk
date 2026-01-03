"""
Streamlit Web Interface for Movie Recommendation Chatbot with User Profiling
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from collections import Counter
from datetime import datetime
import json

# ============================================================================
# USER PROFILING CLASS
# ============================================================================

class UserProfile:
    """Tracks and learns user preferences"""
    
    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.interaction_history = []
        self.liked_movies = []
        self.favorite_genres = Counter()
        self.session_start = datetime.now()
        
    def record_interaction(self, movie_title, genres, interaction_type='view'):
        interaction = {
            'timestamp': datetime.now(),
            'movie': movie_title,
            'genres': genres,
            'type': interaction_type
        }
        self.interaction_history.append(interaction)
        
        genre_list = genres.split()
        for genre in genre_list:
            weight = 2 if interaction_type == 'like' else 1
            self.favorite_genres[genre.lower()] += weight
        
        if interaction_type in ['like', 'search']:
            if movie_title not in self.liked_movies:
                self.liked_movies.append(movie_title)
    
    def get_genre_preferences(self, top_n=5):
        return [genre for genre, _ in self.favorite_genres.most_common(top_n)]
    
    def get_preference_vector(self, df, tfidf_matrix):
        if not self.liked_movies:
            return None
        
        liked_indices = []
        for movie in self.liked_movies:
            idx = df[df['title'].str.lower() == movie.lower()].index
            if len(idx) > 0:
                liked_indices.append(idx[0])
        
        if not liked_indices:
            return None
        
        liked_vectors = tfidf_matrix[liked_indices]
        preference_vector = np.asarray(liked_vectors.mean(axis=0)).flatten()
        
        return preference_vector
    
    def get_stats(self):
        return {
            'total_interactions': len(self.interaction_history),
            'liked_movies_count': len(self.liked_movies),
            'top_genres': self.get_genre_preferences(3),
            'session_duration': (datetime.now() - self.session_start).seconds
        }


# ============================================================================
# ENHANCED RECOMMENDATION ENGINE
# ============================================================================

class MovieRecommendationEngine:
    
    def __init__(self, df):
        self.df = df
        self.tfidf_matrix = None
        self.cosine_sim = None
        
    def build_tfidf_matrix(self):
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = vectorizer.fit_transform(self.df['combined'])
        
    def compute_similarity(self):
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def get_recommendations_by_title(self, title, top_n=10):
        idx = self.df[self.df['title'].str.lower() == title.lower()].index
        if len(idx) == 0:
            return None
        idx = idx[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        for i, score in sim_scores:
            recommendations.append({
                'title': self.df.iloc[i]['title'],
                'genres': self.df.iloc[i]['genres'],
                'score': round(score, 3)
            })
        return recommendations
    
    def get_recommendations_by_genres(self, genres, top_n=10):
        mask = self.df['genres'].str.lower().str.contains('|'.join(genres), case=False, na=False)
        filtered_movies = self.df[mask]
        if len(filtered_movies) == 0:
            return None
        
        results = []
        for _, row in filtered_movies.head(top_n).iterrows():
            results.append({
                'title': row['title'],
                'genres': row['genres'],
                'overview': row['overview'][:100] + '...'
            })
        return results
    
    def get_personalized_recommendations(self, user_profile, top_n=10, exclude_seen=True):
        preference_vector = user_profile.get_preference_vector(self.df, self.tfidf_matrix)
        
        if preference_vector is None:
            return None
        
        similarities = cosine_similarity([preference_vector], self.tfidf_matrix)[0]
        similar_indices = similarities.argsort()[::-1]
        
        seen_movies_lower = [m.lower() for m in user_profile.liked_movies]
        
        recommendations = []
        for idx in similar_indices:
            movie_title = self.df.iloc[idx]['title']
            
            if exclude_seen and movie_title.lower() in seen_movies_lower:
                continue
            
            recommendations.append({
                'title': movie_title,
                'genres': self.df.iloc[idx]['genres'],
                'score': round(similarities[idx], 3),
                'reason': 'Based on your preferences'
            })
            
            if len(recommendations) >= top_n:
                break
        
        return recommendations
    
    def get_hybrid_recommendations(self, movie_title, user_profile, top_n=10):
        content_recs = self.get_recommendations_by_title(movie_title, top_n=20)
        
        if not content_recs:
            return None
        
        favorite_genres = user_profile.get_genre_preferences(3)
        
        for rec in content_recs:
            genre_boost = 0
            movie_genres = rec['genres'].lower().split()
            
            for fav_genre in favorite_genres:
                if fav_genre in movie_genres:
                    genre_boost += 0.1
            
            rec['original_score'] = rec['score']
            rec['score'] = min(1.0, rec['score'] + genre_boost)
            rec['personalized'] = genre_boost > 0
        
        content_recs.sort(key=lambda x: x['score'], reverse=True)
        
        return content_recs[:top_n]


# ============================================================================
# NLP PROCESSOR
# ============================================================================

class NLPProcessor:
    
    def __init__(self, df):
        self.df = df
        self.movie_titles = df['title'].str.lower().tolist()
        self.genre_keywords = {
            'action', 'adventure', 'animation', 'comedy', 'crime',
            'documentary', 'drama', 'family', 'fantasy', 'history',
            'horror', 'music', 'mystery', 'romance', 'science fiction',
            'sci-fi', 'scifi', 'thriller', 'war', 'western'
        }
        
    def extract_movie_title(self, query):
        query_lower = query.lower()
        for title in self.movie_titles:
            if title in query_lower:
                return self.df[self.df['title'].str.lower() == title]['title'].iloc[0]
        
        words = query_lower.split()
        for i in range(len(words)):
            for j in range(i+1, len(words)+1):
                phrase = ' '.join(words[i:j])
                matches = get_close_matches(phrase, self.movie_titles, n=1, cutoff=0.8)
                if matches:
                    return self.df[self.df['title'].str.lower() == matches[0]]['title'].iloc[0]
        return None
    
    def extract_genres(self, query):
        query_lower = query.lower()
        detected_genres = []
        for genre in self.genre_keywords:
            if genre in query_lower:
                if genre in ['sci-fi', 'scifi']:
                    detected_genres.append('science fiction')
                else:
                    detected_genres.append(genre)
        return list(set(detected_genres))
    
    def detect_intent(self, query):
        query_lower = query.lower()
        
        personalized_keywords = ['for me', 'personalized', 'based on my', 'what should i watch', 
                                 'recommend something', 'suggest something', 'my preferences']
        for keyword in personalized_keywords:
            if keyword in query_lower:
                return 'personalized'
        
        feedback_keywords = ['i liked', 'i loved', 'i enjoyed', 'that was great', 'i like']
        for keyword in feedback_keywords:
            if keyword in query_lower:
                return 'feedback_positive'
        
        similarity_keywords = ['like', 'similar', 'same as', 'such as', 'recommend movies like']
        for keyword in similarity_keywords:
            if keyword in query_lower:
                return 'recommend_similar'
        
        if any(genre in query_lower for genre in self.genre_keywords):
            return 'recommend_genre'
        
        return 'unknown'


# ============================================================================
# ENHANCED CHATBOT
# ============================================================================

class MovieChatbot:
    
    def __init__(self, df, rec_engine, nlp_processor, user_profile=None):
        self.df = df
        self.rec_engine = rec_engine
        self.nlp = nlp_processor
        self.user_profile = user_profile or UserProfile()
        
    def process_query(self, query):
        movie_title = self.nlp.extract_movie_title(query)
        genres = self.nlp.extract_genres(query)
        intent = self.nlp.detect_intent(query)
        
        if intent == 'feedback_positive' and movie_title:
            self.user_profile.record_interaction(
                movie_title, 
                self._get_movie_genres(movie_title),
                'like'
            )
            return {
                'type': 'feedback_received',
                'message': f"Great! I've noted that you liked '{movie_title}'. This will help me give better recommendations!",
                'movie': movie_title
            }
        
        if intent == 'personalized':
            if len(self.user_profile.liked_movies) == 0:
                return {
                    'type': 'no_history',
                    'message': "I don't have enough information about your preferences yet. Try:\n" +
                              "  • Tell me movies you like: 'I liked Inception'\n" +
                              "  • Ask for specific movies: 'Recommend movies like The Matrix'\n" +
                              "  • Search by genre: 'Show me sci-fi movies'"
                }
            
            recommendations = self.rec_engine.get_personalized_recommendations(
                self.user_profile, top_n=10
            )
            
            if recommendations:
                return {
                    'type': 'personalized',
                    'recommendations': recommendations,
                    'profile_stats': self.user_profile.get_stats()
                }
        
        if intent == 'recommend_similar' and movie_title:
            self.user_profile.record_interaction(
                movie_title,
                self._get_movie_genres(movie_title),
                'search'
            )
            
            if len(self.user_profile.liked_movies) > 2:
                recommendations = self.rec_engine.get_hybrid_recommendations(
                    movie_title, self.user_profile
                )
            else:
                recommendations = self.rec_engine.get_recommendations_by_title(movie_title)
            
            if recommendations:
                return {
                    'type': 'movie_based',
                    'query_movie': movie_title,
                    'recommendations': recommendations,
                    'personalized': len(self.user_profile.liked_movies) > 2
                }
            else:
                return {
                    'type': 'error',
                    'message': f"Sorry, I couldn't find '{movie_title}' in the database."
                }
        
        elif intent == 'recommend_genre' and genres:
            recommendations = self.rec_engine.get_recommendations_by_genres(genres)
            
            if recommendations:
                return {
                    'type': 'genre_based',
                    'genres': genres,
                    'recommendations': recommendations
                }
            else:
                return {
                    'type': 'error',
                    'message': f"Sorry, I couldn't find movies for genres: {', '.join(genres)}"
                }
        
        else:
            return {
                'type': 'clarification',
                'message': "I can help you find movies! Try:\n" +
                          "  • 'Recommend movies like Inception'\n" +
                          "  • 'I liked The Matrix' (I'll learn your preferences!)\n" +
                          "  • 'Show me action thriller movies'\n" +
                          "  • 'Recommend something for me' (personalized)"
            }
    
    def _get_movie_genres(self, movie_title):
        movie = self.df[self.df['title'].str.lower() == movie_title.lower()]
        if len(movie) > 0:
            return movie.iloc[0]['genres']
        return ""


# ============================================================================
# INITIALIZE SYSTEM
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = UserProfile()

@st.cache_resource
def initialize_system():
    try:
        df = pd.read_csv('movies_cleaned.csv')
        rec_engine = MovieRecommendationEngine(df)
        rec_engine.build_tfidf_matrix()
        rec_engine.compute_similarity()
        nlp_processor = NLPProcessor(df)
        return rec_engine, nlp_processor, df
    except FileNotFoundError:
        return None, None, None


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Movie Recommendation Chatbot",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Movie Recommendation Chatbot")
    st.markdown("### Ask me for movie recommendations - I learn your preferences!")
    
    # Initialize system
    if not st.session_state.system_initialized:
        with st.spinner("Loading movie database and building recommendation engine..."):
            rec_engine, nlp_processor, df = initialize_system()
            
            if rec_engine is None:
                st.error("❌ Error: Could not load 'movies_cleaned.csv'. Please ensure the file exists.")
                st.stop()
            
            st.session_state.rec_engine = rec_engine
            st.session_state.nlp_processor = nlp_processor
            st.session_state.df = df
            st.session_state.chatbot = MovieChatbot(
                df, rec_engine, nlp_processor, st.session_state.user_profile
            )
            st.session_state.system_initialized = True
            st.success(f"✅ Loaded {len(df)} movies successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("👤 Your Profile")
        stats = st.session_state.user_profile.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Movies Liked", stats['liked_movies_count'])
        with col2:
            st.metric("Interactions", stats['total_interactions'])
        
        if stats['top_genres']:
            st.write("**Favorite Genres:**")
            for genre in stats['top_genres']:
                st.write(f"• {genre.title()}")
        
        if st.session_state.user_profile.liked_movies:
            st.write("**Movies You Liked:**")
            for movie in st.session_state.user_profile.liked_movies[:5]:
                st.write(f"• {movie}")
            if len(st.session_state.user_profile.liked_movies) > 5:
                st.caption(f"... and {len(st.session_state.user_profile.liked_movies) - 5} more")
        
        st.divider()
        st.header("💡 Try These")
        
        examples = [
            "Recommend something for me",
            "I liked Inception",
            "Movies like The Matrix",
            "Show me action thriller movies",
            "Give me sci-fi movies"
        ]
        
        for example in examples:
            if st.button(example, key=example, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                response = st.session_state.chatbot.process_query(example)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset Profile", use_container_width=True):
                st.session_state.user_profile = UserProfile()
                st.session_state.chatbot.user_profile = st.session_state.user_profile
                st.success("Profile reset!")
                st.rerun()
        
        st.caption(f"Total movies: {len(st.session_state.df)}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                response = message["content"]
                
                if response['type'] == 'movie_based':
                    personalized_note = "🎯 **Personalized based on your preferences!**" if response.get('personalized') else ""
                    st.write(f"**Movies similar to '{response['query_movie']}':**")
                    if personalized_note:
                        st.info(personalized_note)
                    
                    for i, rec in enumerate(response['recommendations'], 1):
                        personalized_icon = "⭐ " if rec.get('personalized') else ""
                        with st.expander(f"{personalized_icon}{i}. {rec['title']} (Similarity: {rec['score']})"):
                            st.write(f"**Genres:** {rec['genres']}")
                            if rec.get('personalized'):
                                st.success("Matches your favorite genres!")
                
                elif response['type'] == 'personalized':
                    stats = response['profile_stats']
                    st.success(f"🎯 **Personalized Recommendations** (Based on {stats['liked_movies_count']} movies you liked)")
                    st.write(f"**Your favorite genres:** {', '.join(stats['top_genres'])}")
                    
                    for i, rec in enumerate(response['recommendations'], 1):
                        with st.expander(f"{i}. {rec['title']} (Match: {rec['score']})"):
                            st.write(f"**Genres:** {rec['genres']}")
                            st.info(rec.get('reason', 'Recommended for you'))
                
                elif response['type'] == 'genre_based':
                    st.write(f"**Top {', '.join(response['genres']).title()} Movies:**")
                    for i, rec in enumerate(response['recommendations'], 1):
                        with st.expander(f"{i}. {rec['title']}"):
                            st.write(f"**Genres:** {rec['genres']}")
                            st.write(f"**Overview:** {rec['overview']}")
                
                elif response['type'] == 'feedback_received':
                    st.success(response['message'])
                    st.balloons()
                
                elif response['type'] == 'no_history':
                    st.info(response['message'])
                
                elif response['type'] == 'error':
                    st.error(response['message'])
                
                elif response['type'] == 'clarification':
                    st.info(response['message'])
    
    # Chat input
    if prompt := st.chat_input("Ask for movie recommendations..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = st.session_state.chatbot.process_query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()