from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import json
import random
from datetime import datetime

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

class SongRecommender:
    def __init__(self):
        self.songs_df = None
        self.user_interactions = {}
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.user_item_matrix = None
        self.load_sample_data()
        self.build_models()
    
    def load_sample_data(self):
        """Load sample song data for demonstration"""
        # Sample song data with rich metadata
        songs_data = [
            {"id": 1, "title": "Blinding Lights", "artist": "The Weeknd", "genre": "Pop", "year": 2019, "duration": 200, "energy": 0.8, "danceability": 0.5, "valence": 0.3, "image": "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb36", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 2, "title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop", "year": 2017, "duration": 233, "energy": 0.6, "danceability": 0.8, "valence": 0.9, "image": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 3, "title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "year": 1975, "duration": 355, "energy": 0.6, "danceability": 0.3, "valence": 0.4, "image": "https://i.scdn.co/image/ab67616d0000b273ce4f1737bc8a646c8c4bd25a", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 4, "title": "Hotel California", "artist": "Eagles", "genre": "Rock", "year": 1976, "duration": 391, "energy": 0.5, "danceability": 0.2, "valence": 0.3, "image": "https://i.scdn.co/image/ab67616d0000b2734637341b9f507521afa9a778", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 5, "title": "Billie Jean", "artist": "Michael Jackson", "genre": "Pop", "year": 1982, "duration": 294, "energy": 0.7, "danceability": 0.9, "valence": 0.6, "image": "https://i.scdn.co/image/ab67616d0000b273de09e02aa7febf3d8b8e5774", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 6, "title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Rock", "year": 1991, "duration": 301, "energy": 0.9, "danceability": 0.5, "valence": 0.4, "image": "https://i.scdn.co/image/ab67616d0000b273e175a19e530c898d167d39bf", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 7, "title": "Dancing Queen", "artist": "ABBA", "genre": "Pop", "year": 1976, "duration": 230, "energy": 0.8, "danceability": 0.9, "valence": 0.9, "image": "https://i.scdn.co/image/ab67616d0000b2739e2f95ae77cf436017ada9cb", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 8, "title": "Stairway to Heaven", "artist": "Led Zeppelin", "genre": "Rock", "year": 1971, "duration": 482, "energy": 0.4, "danceability": 0.2, "valence": 0.3, "image": "https://i.scdn.co/image/ab67616d0000b273c8a11e48c91a982d086afc69", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 9, "title": "Someone Like You", "artist": "Adele", "genre": "Pop", "year": 2011, "duration": 285, "energy": 0.3, "danceability": 0.5, "valence": 0.2, "image": "https://i.scdn.co/image/ab67616d0000b27368f3e0e28d477c28d6777c37", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 10, "title": "Sweet Child O' Mine", "artist": "Guns N' Roses", "genre": "Rock", "year": 1987, "duration": 356, "energy": 0.8, "danceability": 0.4, "valence": 0.6, "image": "https://i.scdn.co/image/ab67616d0000b273e44963b8bb127552ac761873", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 11, "title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "genre": "Pop", "year": 2014, "duration": 270, "energy": 0.9, "danceability": 0.9, "valence": 0.9, "image": "https://i.scdn.co/image/ab67616d0000b273e419ccba0baa8bd3f3d7abf2", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 12, "title": "Imagine", "artist": "John Lennon", "genre": "Pop", "year": 1971, "duration": 183, "energy": 0.2, "danceability": 0.4, "valence": 0.7, "image": "https://i.scdn.co/image/ab67616d0000b273e3e3b64cea45265469d4cafa", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 13, "title": "Thriller", "artist": "Michael Jackson", "genre": "Pop", "year": 1982, "duration": 357, "energy": 0.7, "danceability": 0.7, "valence": 0.5, "image": "https://i.scdn.co/image/ab67616d0000b273de09e02aa7febf3d8b8e5774", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 14, "title": "Back in Black", "artist": "AC/DC", "genre": "Rock", "year": 1980, "duration": 255, "energy": 0.9, "danceability": 0.4, "valence": 0.6, "image": "https://i.scdn.co/image/ab67616d0000b27003ce2b70e3b8c526fa0d8ba5", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 15, "title": "Hey Jude", "artist": "The Beatles", "genre": "Rock", "year": 1968, "duration": 431, "energy": 0.4, "danceability": 0.3, "valence": 0.7, "image": "https://i.scdn.co/image/ab67616d0000b273dc30583ba3d90eaa9d74e3c3", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 16, "title": "Rolling in the Deep", "artist": "Adele", "genre": "Pop", "year": 2010, "duration": 228, "energy": 0.6, "danceability": 0.6, "valence": 0.4, "image": "https://i.scdn.co/image/ab67616d0000b27368f3e0e28d477c28d6777c37", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 17, "title": "Don't Stop Believin'", "artist": "Journey", "genre": "Rock", "year": 1981, "duration": 251, "energy": 0.7, "danceability": 0.5, "valence": 0.8, "image": "https://i.scdn.co/image/ab67616d0000b273c0b84c48e41c3dc9b862fb2b", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 18, "title": "Bad Guy", "artist": "Billie Eilish", "genre": "Pop", "year": 2019, "duration": 194, "energy": 0.4, "danceability": 0.7, "valence": 0.1, "image": "https://i.scdn.co/image/ab67616d0000b273a8d2c0f4e8c0b5b6f2e9a8c2", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 19, "title": "Wonderwall", "artist": "Oasis", "genre": "Rock", "year": 1995, "duration": 258, "energy": 0.5, "danceability": 0.4, "valence": 0.6, "image": "https://i.scdn.co/image/ab67616d0000b273ada101c2e9c97eed2b1b0f4c", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"},
            {"id": 20, "title": "Perfect", "artist": "Ed Sheeran", "genre": "Pop", "year": 2017, "duration": 263, "energy": 0.3, "danceability": 0.6, "valence": 0.8, "image": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96", "preview_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"}
        ]
        
        self.songs_df = pd.DataFrame(songs_data)
        
        # Generate sample user interactions
        users = [f"user_{i}" for i in range(1, 21)]
        interactions = []
        
        for user in users:
            # Each user rates 8-15 random songs
            rated_songs = random.sample(range(1, 21), random.randint(8, 15))
            for song_id in rated_songs:
                rating = random.choice([3, 4, 5]) + random.random()  # 3-6 rating scale
                interactions.append({
                    'user_id': user,
                    'song_id': song_id,
                    'rating': round(rating, 1),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        self.interactions_df = pd.DataFrame(interactions)
    
    def build_models(self):
        """Build recommendation models"""
        # Content-based: TF-IDF on combined features
        self.songs_df['features'] = (
            self.songs_df['artist'] + ' ' + 
            self.songs_df['genre'] + ' ' + 
            self.songs_df['year'].astype(str)
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.songs_df['features'])
        
        # Collaborative filtering: User-Item matrix
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='song_id', 
            values='rating', 
            fill_value=0
        )
        
        # SVD for matrix factorization
        self.svd_model = TruncatedSVD(n_components=10, random_state=42)
        self.user_features = self.svd_model.fit_transform(self.user_item_matrix)
    
    def content_based_recommend(self, song_id, n_recommendations=5):
        """Content-based recommendations"""
        print(f"Content-based recommendation for song_id: {song_id}")  # Debug
        
        if song_id not in self.songs_df['id'].values:
            print(f"Song ID {song_id} not found in database")  # Debug
            return []
        
        song_idx = self.songs_df[self.songs_df['id'] == song_id].index[0]
        print(f"Song index: {song_idx}")  # Debug
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(self.tfidf_matrix[song_idx:song_idx+1], self.tfidf_matrix).flatten()
        
        # Get similar songs
        similar_indices = cosine_sim.argsort()[::-1][1:n_recommendations+1]
        print(f"Similar indices: {similar_indices}")  # Debug
        
        recommendations = []
        for idx in similar_indices:
            song = self.songs_df.iloc[idx]
            recommendations.append({
                'id': int(song['id']),
                'title': str(song['title']),
                'artist': str(song['artist']),
                'genre': str(song['genre']),
                'year': int(song['year']),
                'image': str(song['image']),
                'similarity': float(round(cosine_sim[idx], 3)),
                'method': 'Content-Based'
            })
        
        print(f"Generated {len(recommendations)} content-based recommendations")  # Debug
        return recommendations
    
    def collaborative_recommend(self, user_id, n_recommendations=5):
        """Collaborative filtering recommendations"""
        if user_id not in self.user_item_matrix.index:
            # For new users, recommend popular songs
            popular_songs = self.interactions_df.groupby('song_id')['rating'].agg(['mean', 'count'])
            popular_songs = popular_songs[popular_songs['count'] >= 3].sort_values('mean', ascending=False)
            
            recommendations = []
            for song_id in popular_songs.head(n_recommendations).index:
                song = self.songs_df[self.songs_df['id'] == song_id].iloc[0]
                recommendations.append({
                    'id': int(song['id']),
                    'title': str(song['title']),
                    'artist': str(song['artist']),
                    'genre': str(song['genre']),
                    'year': int(song['year']),
                    'image': str(song['image']),
                    'predicted_rating': float(round(popular_songs.loc[song_id, 'mean'], 2)),
                    'method': 'Popular (New User)'
                })
            return recommendations
        
        # Get user ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_songs = user_ratings[user_ratings == 0].index
        
        if len(unrated_songs) == 0:
            return []
        
        # Calculate user similarity
        user_idx = list(self.user_item_matrix.index).index(user_id)
        user_sim = cosine_similarity([self.user_features[user_idx]], self.user_features).flatten()
        
        # Predict ratings for unrated songs
        predictions = []
        for song_id in unrated_songs:
            if song_id in self.user_item_matrix.columns:
                # Get ratings from similar users
                song_ratings = self.user_item_matrix[song_id]
                rated_users = song_ratings[song_ratings > 0].index
                
                if len(rated_users) > 0:
                    similar_user_ratings = []
                    similarities = []
                    
                    for other_user in rated_users:
                        other_idx = list(self.user_item_matrix.index).index(other_user)
                        if user_sim[other_idx] > 0.1:  # Only consider somewhat similar users
                            similar_user_ratings.append(song_ratings[other_user])
                            similarities.append(user_sim[other_idx])
                    
                    if similar_user_ratings:
                        # Weighted average prediction
                        predicted_rating = np.average(similar_user_ratings, weights=similarities)
                        predictions.append((song_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for song_id, pred_rating in predictions[:n_recommendations]:
            song = self.songs_df[self.songs_df['id'] == song_id].iloc[0]
            recommendations.append({
                'id': int(song['id']),
                'title': str(song['title']),
                'artist': str(song['artist']),
                'genre': str(song['genre']),
                'year': int(song['year']),
                'image': str(song['image']),
                'predicted_rating': float(round(pred_rating, 2)),
                'method': 'Collaborative Filtering'
            })
        
        return recommendations
    
    def hybrid_recommend(self, user_id=None, song_id=None, n_recommendations=5):
        """Hybrid recommendations combining both methods"""
        recommendations = []
        
        if user_id:
            collab_recs = self.collaborative_recommend(user_id, n_recommendations//2 + 1)
            recommendations.extend(collab_recs)
        
        if song_id:
            content_recs = self.content_based_recommend(song_id, n_recommendations//2 + 1)
            recommendations.extend(content_recs)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_recs = []
        for rec in recommendations:
            if rec['id'] not in seen_ids:
                unique_recs.append(rec)
                seen_ids.add(rec['id'])
                if len(unique_recs) >= n_recommendations:
                    break
        
        return unique_recs[:n_recommendations]
    
    def add_user_rating(self, user_id, song_id, rating):
        """Add user rating (for learning from feedback)"""
        new_interaction = pd.DataFrame([{
            'user_id': user_id,
            'song_id': song_id,
            'rating': rating,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        
        self.interactions_df = pd.concat([self.interactions_df, new_interaction], ignore_index=True)
        
        # Rebuild user-item matrix
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='song_id', 
            values='rating', 
            fill_value=0
        )
        
        # Rebuild SVD model
        self.user_features = self.svd_model.fit_transform(self.user_item_matrix)
    
    def get_stats(self):
        """Get system statistics"""
        return {
            'total_songs': int(len(self.songs_df)),
            'total_users': int(len(self.user_item_matrix)),
            'total_ratings': int(len(self.interactions_df)),
            'avg_rating': float(round(self.interactions_df['rating'].mean(), 2)),
            'genres': list(self.songs_df['genre'].unique()),
            'top_artists': {str(k): int(v) for k, v in self.songs_df['artist'].value_counts().head(5).to_dict().items()}
        }

# Initialize the recommender system
recommender = SongRecommender()

@app.route('/')
def index():
    """Main recommendation interface"""
    stats = recommender.get_stats()
    popular_songs = recommender.songs_df.head(8).to_dict('records')
    return render_template('index.html', stats=stats, popular_songs=popular_songs)

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    stats = recommender.get_stats()
    
    # Additional analytics with proper type conversion
    genre_dist = {str(k): int(v) for k, v in recommender.songs_df['genre'].value_counts().to_dict().items()}
    year_dist = {str(k): int(v) for k, v in recommender.songs_df.groupby('year').size().to_dict().items()}
    rating_dist = {str(k): int(v) for k, v in recommender.interactions_df['rating'].round().value_counts().sort_index().to_dict().items()}
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         genre_dist=genre_dist,
                         year_dist=year_dist,
                         rating_dist=rating_dist)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for recommendations"""
    data = request.json
    method = data.get('method', 'hybrid')
    user_id = data.get('user_id')
    song_id = data.get('song_id')
    n_recs = data.get('n_recommendations', 5)
    
    try:
        if method == 'content':
            if not song_id:
                return jsonify({'error': 'song_id required for content-based recommendations'}), 400
            recommendations = recommender.content_based_recommend(int(song_id), n_recs)
        elif method == 'collaborative':
            if not user_id:
                return jsonify({'error': 'user_id required for collaborative filtering'}), 400
            recommendations = recommender.collaborative_recommend(user_id, n_recs)
        else:  # hybrid
            recommendations = recommender.hybrid_recommend(user_id, int(song_id) if song_id else None, n_recs)
        
        # Ensure all values are JSON serializable
        serializable_recs = []
        for rec in recommendations:
            serializable_rec = {}
            for key, value in rec.items():
                if isinstance(value, (np.integer, np.int64)):
                    serializable_rec[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    serializable_rec[key] = float(value)
                else:
                    serializable_rec[key] = value
            serializable_recs.append(serializable_rec)
        
        return jsonify({
            'recommendations': serializable_recs,
            'method': method,
            'count': len(serializable_recs)
        })
    
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/api/rate', methods=['POST'])
def api_rate():
    """API endpoint for rating songs"""
    data = request.json
    user_id = data.get('user_id')
    song_id = data.get('song_id')
    rating = data.get('rating')
    
    if not all([user_id, song_id, rating]):
        return jsonify({'error': 'user_id, song_id, and rating are required'}), 400
    
    try:
        recommender.add_user_rating(user_id, int(song_id), float(rating))
        return jsonify({'message': 'Rating added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/songs')
def api_songs():
    """API endpoint to get all songs"""
    songs = recommender.songs_df.to_dict('records')
    # Convert numpy types to native Python types
    serializable_songs = []
    for song in songs:
        serializable_song = {}
        for key, value in song.items():
            if isinstance(value, (np.integer, np.int64)):
                serializable_song[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                serializable_song[key] = float(value)
            else:
                serializable_song[key] = value
        serializable_songs.append(serializable_song)
    
    return jsonify({'songs': serializable_songs})

@app.route('/api/stats')
def api_stats():
    """API endpoint for system statistics"""
    return jsonify(recommender.get_stats())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)