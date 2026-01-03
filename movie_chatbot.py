"""
Movie Recommendation System with Chat Interface + User Profiling
Academic Project - Content-Based Filtering with NLP and Preference Learning
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import get_close_matches
import json
from datetime import datetime
from collections import Counter

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

class MovieDataLoader:
    """Handles loading and initial processing of movie dataset"""
    
    def __init__(self, filepath='movies_cleaned.csv'):
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Load the movie dataset from CSV"""
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✓ Loaded {len(self.df)} movies successfully")
            return self.df
        except FileNotFoundError:
            print(f"Error: File '{self.filepath}' not found")
            return None
    
    def get_dataframe(self):
        """Return the loaded dataframe"""
        return self.df


# ============================================================================
# PART 2: USER PROFILING AND PREFERENCE LEARNING
# ============================================================================

class UserProfile:
    """
    Tracks and learns user preferences over time
    Implements implicit feedback learning (no explicit ratings needed)
    """
    
    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.interaction_history = []
        self.liked_movies = []  # Movies user showed interest in
        self.favorite_genres = Counter()  # Genre preferences
        self.preferred_features = None  # Learned feature vector
        self.session_start = datetime.now()
        
    def record_interaction(self, movie_title, genres, interaction_type='view'):
        """
        Record user interaction with a movie
        
        Args:
            movie_title: Title of the movie
            genres: Genres of the movie
            interaction_type: 'view', 'like', 'search'
        """
        interaction = {
            'timestamp': datetime.now(),
            'movie': movie_title,
            'genres': genres,
            'type': interaction_type
        }
        self.interaction_history.append(interaction)
        
        # Update genre preferences
        genre_list = genres.split()
        for genre in genre_list:
            weight = 2 if interaction_type == 'like' else 1
            self.favorite_genres[genre.lower()] += weight
        
        # Add to liked movies if positive interaction
        if interaction_type in ['like', 'search']:
            self.liked_movies.append(movie_title)
    
    def get_genre_preferences(self, top_n=5):
        """Get top N favorite genres"""
        return [genre for genre, _ in self.favorite_genres.most_common(top_n)]
    
    def get_preference_vector(self, df, tfidf_matrix):
        """
        Build a preference vector based on liked movies
        This creates a "profile" of what the user likes
        
        Returns:
            Averaged TF-IDF vector representing user preferences
        """
        if not self.liked_movies:
            return None
        
        # Get indices of liked movies
        liked_indices = []
        for movie in self.liked_movies:
            idx = df[df['title'].str.lower() == movie.lower()].index
            if len(idx) > 0:
                liked_indices.append(idx[0])
        
        if not liked_indices:
            return None
        
        # Average the TF-IDF vectors of liked movies
        liked_vectors = tfidf_matrix[liked_indices]
        preference_vector = np.asarray(liked_vectors.mean(axis=0)).flatten()
        
        return preference_vector
    
    def save_profile(self, filepath='user_profile.json'):
        """Save user profile to file"""
        profile_data = {
            'user_id': self.user_id,
            'liked_movies': self.liked_movies,
            'favorite_genres': dict(self.favorite_genres),
            'interaction_count': len(self.interaction_history),
            'session_start': self.session_start.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"✓ Profile saved to {filepath}")
    
    def load_profile(self, filepath='user_profile.json'):
        """Load user profile from file"""
        try:
            with open(filepath, 'r') as f:
                profile_data = json.load(f)
            
            self.user_id = profile_data['user_id']
            self.liked_movies = profile_data['liked_movies']
            self.favorite_genres = Counter(profile_data['favorite_genres'])
            
            print(f"✓ Profile loaded: {len(self.liked_movies)} liked movies")
            return True
        except FileNotFoundError:
            print("No existing profile found. Starting fresh.")
            return False
    
    def get_stats(self):
        """Get user profile statistics"""
        return {
            'total_interactions': len(self.interaction_history),
            'liked_movies_count': len(self.liked_movies),
            'top_genres': self.get_genre_preferences(3),
            'session_duration': (datetime.now() - self.session_start).seconds
        }


# ============================================================================
# PART 3: ENHANCED RECOMMENDATION ENGINE
# ============================================================================

class MovieRecommendationEngine:
    """
    Enhanced recommendation engine with user preference integration
    Combines content-based filtering with personalization
    """
    
    def __init__(self, df):
        self.df = df
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.vectorizer = None
        
    def build_tfidf_matrix(self):
        """Build TF-IDF matrix from the 'combined' column"""
        print("\n🔧 Building TF-IDF Matrix...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined'])
        print(f"✓ TF-IDF Matrix shape: {self.tfidf_matrix.shape}")
        
    def compute_similarity(self):
        """Compute cosine similarity between all movies"""
        print("🔧 Computing Cosine Similarity Matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print(f"✓ Similarity Matrix shape: {self.cosine_sim.shape}")
        
    def get_recommendations_by_title(self, title, top_n=10):
        """Get movie recommendations based on a movie title"""
        idx = self.df[self.df['title'].str.lower() == title.lower()].index
        
        if len(idx) == 0:
            return None
        
        idx = idx[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        recommendations = []
        for idx, score in zip(movie_indices, scores):
            recommendations.append({
                'title': self.df.iloc[idx]['title'],
                'genres': self.df.iloc[idx]['genres'],
                'score': round(score, 3)
            })
        
        return recommendations
    
    def get_recommendations_by_genres(self, genres, top_n=10):
        """Get movie recommendations based on genre keywords"""
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
        """
        Get personalized recommendations based on user profile
        Uses learned preferences to find similar movies
        
        Args:
            user_profile: UserProfile object with interaction history
            top_n: Number of recommendations
            exclude_seen: Whether to exclude already seen movies
            
        Returns:
            List of personalized movie recommendations
        """
        # Get user preference vector
        preference_vector = user_profile.get_preference_vector(self.df, self.tfidf_matrix)
        
        if preference_vector is None:
            return None
        
        # Calculate similarity between user profile and all movies
        similarities = cosine_similarity([preference_vector], self.tfidf_matrix)[0]
        
        # Get indices sorted by similarity
        similar_indices = similarities.argsort()[::-1]
        
        # Filter out already seen movies if requested
        seen_movies_lower = [m.lower() for m in user_profile.liked_movies]
        
        recommendations = []
        for idx in similar_indices:
            movie_title = self.df.iloc[idx]['title']
            
            # Skip if already seen
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
        """
        Hybrid recommendations combining content similarity and user preferences
        
        Args:
            movie_title: Reference movie
            user_profile: User profile for personalization
            top_n: Number of recommendations
            
        Returns:
            Hybrid recommendations with weighted scores
        """
        # Get content-based recommendations
        content_recs = self.get_recommendations_by_title(movie_title, top_n=20)
        
        if not content_recs:
            return None
        
        # Get user preferences
        favorite_genres = user_profile.get_genre_preferences(3)
        
        # Re-rank based on user preferences
        for rec in content_recs:
            genre_boost = 0
            movie_genres = rec['genres'].lower().split()
            
            # Boost score if movie matches user's favorite genres
            for fav_genre in favorite_genres:
                if fav_genre in movie_genres:
                    genre_boost += 0.1
            
            # Combine content similarity with genre preference
            rec['original_score'] = rec['score']
            rec['score'] = min(1.0, rec['score'] + genre_boost)
            rec['personalized'] = genre_boost > 0
        
        # Re-sort by new score
        content_recs.sort(key=lambda x: x['score'], reverse=True)
        
        return content_recs[:top_n]


# ============================================================================
# PART 4: NLP MODULE
# ============================================================================

class NLPProcessor:
    """Handles natural language understanding for user queries"""
    
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
        """Extract movie title from user query using fuzzy matching"""
        query_lower = query.lower()
        
        # Try direct match first
        for title in self.movie_titles:
            if title in query_lower:
                return self.df[self.df['title'].str.lower() == title]['title'].iloc[0]
        
        # Try fuzzy matching
        words = query_lower.split()
        for i in range(len(words)):
            for j in range(i+1, len(words)+1):
                phrase = ' '.join(words[i:j])
                matches = get_close_matches(phrase, self.movie_titles, n=1, cutoff=0.8)
                if matches:
                    return self.df[self.df['title'].str.lower() == matches[0]]['title'].iloc[0]
        
        return None
    
    def extract_genres(self, query):
        """Extract genre keywords from user query"""
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
        """
        Detect user intent from query
        
        Returns:
            'recommend_similar': Similar movies
            'recommend_genre': Genre-based
            'personalized': User wants personalized recommendations
            'feedback_positive': User liked something
            'unknown': Cannot determine
        """
        query_lower = query.lower()
        
        # Check for personalized recommendation request
        personalized_keywords = ['for me', 'personalized', 'based on my', 'what should i watch', 
                                 'recommend something', 'suggest something', 'my preferences']
        for keyword in personalized_keywords:
            if keyword in query_lower:
                return 'personalized'
        
        # Check for positive feedback
        feedback_keywords = ['i liked', 'i loved', 'i enjoyed', 'that was great', 'i like']
        for keyword in feedback_keywords:
            if keyword in query_lower:
                return 'feedback_positive'
        
        # Check for similarity search
        similarity_keywords = ['like', 'similar', 'same as', 'such as', 'recommend movies like']
        for keyword in similarity_keywords:
            if keyword in query_lower:
                return 'recommend_similar'
        
        # Check for genre intent
        if any(genre in query_lower for genre in self.genre_keywords):
            return 'recommend_genre'
        
        return 'unknown'


# ============================================================================
# PART 5: ENHANCED CHATBOT WITH USER PROFILING
# ============================================================================

class MovieChatbot:
    """
    Enhanced chatbot with user profiling and preference learning
    """
    
    def __init__(self, df, rec_engine, nlp_processor, user_profile=None):
        self.df = df
        self.rec_engine = rec_engine
        self.nlp = nlp_processor
        self.user_profile = user_profile or UserProfile()
        
    def process_query(self, query):
        """Process user query with personalization"""
        # Extract information
        movie_title = self.nlp.extract_movie_title(query)
        genres = self.nlp.extract_genres(query)
        intent = self.nlp.detect_intent(query)
        
        # Handle positive feedback
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
        
        # Handle personalized recommendation request
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
        
        # Handle movie-based recommendations (with hybrid approach if profile exists)
        if intent == 'recommend_similar' and movie_title:
            # Record interaction
            self.user_profile.record_interaction(
                movie_title,
                self._get_movie_genres(movie_title),
                'search'
            )
            
            # Use hybrid recommendations if user has history
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
        
        # Handle genre-based recommendations
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
        """Helper to get genres for a movie"""
        movie = self.df[self.df['title'].str.lower() == movie_title.lower()]
        if len(movie) > 0:
            return movie.iloc[0]['genres']
        return ""
    
    def format_response(self, response):
        """Format response for display"""
        if response['type'] == 'movie_based':
            personalized_note = "\n(🎯 Personalized based on your preferences!)\n" if response.get('personalized') else ""
            output = f"\n🎬 Movies similar to '{response['query_movie']}':{personalized_note}\n"
            for i, rec in enumerate(response['recommendations'], 1):
                output += f"{i}. {rec['title']}\n"
                output += f"   Genres: {rec['genres']}\n"
                output += f"   Similarity: {rec['score']}\n"
                if rec.get('personalized'):
                    output += f"   ⭐ Matches your favorite genres!\n"
                output += "\n"
            return output
        
        elif response['type'] == 'personalized':
            stats = response['profile_stats']
            output = f"\n🎯 Personalized Recommendations (Based on {stats['liked_movies_count']} movies you liked):\n"
            output += f"Your favorite genres: {', '.join(stats['top_genres'])}\n\n"
            
            for i, rec in enumerate(response['recommendations'], 1):
                output += f"{i}. {rec['title']}\n"
                output += f"   Genres: {rec['genres']}\n"
                output += f"   Match Score: {rec['score']}\n\n"
            return output
        
        elif response['type'] == 'genre_based':
            output = f"\n🎬 Top {', '.join(response['genres']).title()} Movies:\n\n"
            for i, rec in enumerate(response['recommendations'], 1):
                output += f"{i}. {rec['title']}\n"
                output += f"   Genres: {rec['genres']}\n"
                output += f"   {rec['overview']}\n\n"
            return output
        
        elif response['type'] == 'feedback_received':
            return f"\n✅ {response['message']}\n"
        
        elif response['type'] == 'no_history':
            return f"\n💡 {response['message']}\n"
        
        elif response['type'] == 'error':
            return f"\n❌ {response['message']}\n"
        
        elif response['type'] == 'clarification':
            return f"\n💡 {response['message']}\n"
        
        return "\n❌ Something went wrong.\n"
    
    def show_profile_stats(self):
        """Display user profile statistics"""
        stats = self.user_profile.get_stats()
        
        print("\n" + "="*60)
        print("📊 YOUR PROFILE STATISTICS")
        print("="*60)
        print(f"Total interactions: {stats['total_interactions']}")
        print(f"Movies you liked: {stats['liked_movies_count']}")
        print(f"Your top genres: {', '.join(stats['top_genres']) if stats['top_genres'] else 'None yet'}")
        
        if self.user_profile.liked_movies:
            print(f"\nYour liked movies:")
            for movie in self.user_profile.liked_movies[:5]:
                print(f"  • {movie}")
            if len(self.user_profile.liked_movies) > 5:
                print(f"  ... and {len(self.user_profile.liked_movies) - 5} more")
        
        print("="*60 + "\n")


# ============================================================================
# PART 6: ENHANCED COMMAND-LINE INTERFACE
# ============================================================================

def run_cli_chatbot():
    """Run the enhanced chatbot with user profiling"""
    print("=" * 60)
    print("🎬 MOVIE RECOMMENDATION CHATBOT (WITH USER PROFILING)")
    print("=" * 60)
    
    # Initialize system
    print("\n📂 Loading movie database...")
    loader = MovieDataLoader('movies_cleaned.csv')
    df = loader.load_data()
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Build recommendation engine
    rec_engine = MovieRecommendationEngine(df)
    rec_engine.build_tfidf_matrix()
    rec_engine.compute_similarity()
    
    # Initialize user profile
    user_profile = UserProfile()
    user_profile.load_profile()  # Load existing profile if available
    
    # Initialize NLP and chatbot
    nlp_processor = NLPProcessor(df)
    chatbot = MovieChatbot(df, rec_engine, nlp_processor, user_profile)
    
    print("\n✅ System ready! Start chatting...\n")
    print("Special commands:")
    print("  • 'profile' - View your profile statistics")
    print("  • 'save' - Save your profile")
    print("  • 'quit' or 'exit' - End conversation\n")
    
    # Chat loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nSaving your profile...")
            user_profile.save_profile()
            print("Bot: Thanks for using the Movie Recommendation System! Goodbye! 👋\n")
            break
        
        if user_input.lower() == 'profile':
            chatbot.show_profile_stats()
            continue
        
        if user_input.lower() == 'save':
            user_profile.save_profile()
            continue
        
        if not user_input:
            continue
        
        # Process query and get response
        response = chatbot.process_query(user_input)
        formatted_response = chatbot.format_response(response)
        
        print(f"Bot: {formatted_response}")


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_cli_chatbot()