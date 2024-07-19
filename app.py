from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.cluster import KMeans
import folium
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Placestestf.csv', encoding='latin-1')

# Ensure dataset has necessary columns
required_columns = ['City', 'Place_Name', 'latitude', 'longitude', 'Ratings', 'votes', 'Categories', 'Place_desc']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Drop rows with missing coordinates
df = df.dropna(subset=['latitude', 'longitude'])

# Handle missing values in Ratings and votes
df['Ratings'] = df['Ratings'].fillna(0)
df['votes'] = df['votes'].fillna(0)

# Function to calculate weight
def calculate_weight(row):
    if row['Ratings'] > 0 and row['votes'] > 0:
        return (row['Ratings'] + (row['votes'] / 1000)) / 2
    elif row['Ratings'] > 0:
        return row['Ratings']
    elif row['votes'] > 0:
        return row['votes'] / 1000
    else:
        return 0

df['Weight'] = df.apply(calculate_weight, axis=1)

# Min-max normalization
min_weight = df['Weight'].min()
max_weight = df['Weight'].max()

df['Score'] = 2.5 + (df['Weight'] - min_weight) * (3.5 - 2.5) / (max_weight - min_weight)

# Function to match categories
def match_categories(categories_string, selected_categories):
    if pd.isna(categories_string):
        return False
    categories_list = categories_string.split(',')
    for category in categories_list:
        if any(cat.strip().lower() in category.strip().lower() for cat in selected_categories):
            return True
    return False

# Function to recommend places
def recommend_places(city, categories, num_days, places_per_day):
    city_places = df[df['City'] == city].sort_values(by=['Score'], ascending=False)
    primary_places = city_places[city_places['Categories'].apply(lambda x: match_categories(str(x), categories))]
    secondary_places = city_places[~city_places['Categories'].apply(lambda x: match_categories(str(x), categories))]
    combined_places = pd.concat([primary_places, secondary_places]).drop_duplicates(subset=['Place_Name'])

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    combined_places['Cluster'] = kmeans.fit_predict(combined_places[['latitude', 'longitude']])

    clusters = combined_places.groupby('Cluster')
    itinerary = []
    selected_indices = set()
    maps = []
    for day in range(num_days):
        daily_itinerary = []
        day_map = folium.Map(location=[combined_places['latitude'].mean(), combined_places['longitude'].mean()], zoom_start=12)
        for cluster_id, cluster_data in clusters:
            cluster_places = cluster_data[~cluster_data['Place_Name'].isin(selected_indices)].head(places_per_day)
            for place in cluster_places.to_dict('records'):
                folium.Marker(
                    location=[place['latitude'], place['longitude']],
                    popup=place['Place_Name'],
                    icon=folium.Icon(color='red')
                ).add_to(day_map)
            daily_itinerary.extend(cluster_places.to_dict('records'))
            selected_indices.update(cluster_places['Place_Name'])
            if len(daily_itinerary) >= places_per_day:
                break
        if daily_itinerary:  # Only append non-empty days
            itinerary.append(daily_itinerary[:places_per_day])
            maps.append(day_map)
        combined_places = combined_places[~combined_places['Place_Name'].isin(selected_indices)]
        clusters = combined_places.groupby('Cluster')

    return itinerary, len(itinerary), maps

@app.route('/', methods=['GET', 'POST'])
def index():
    available_cities = df['City'].unique()
    if request.method == 'POST':
        city = request.form['city']
        selected_categories = request.form.getlist('categories')
        additional_categories = request.form.getlist('additional_categories')
        places_per_day = int(request.form['places_per_day'])
        num_days = int(request.form['num_days'])

        if additional_categories:
            selected_categories.extend(additional_categories)

        itinerary, available_days, maps = recommend_places(city, selected_categories, num_days, places_per_day)

        return render_template('index.html', available_cities=available_cities, itinerary=itinerary, available_days=available_days, selected_categories=selected_categories, maps=maps)

    return render_template('index.html', available_cities=available_cities)

@app.route('/get_categories', methods=['POST'])
def get_categories():
    data = request.get_json()
    city = data['city']
    available_categories = df[df['City'] == city]['Categories'].dropna().unique()
    available_categories = [cat.strip() for sublist in available_categories for cat in sublist.split(',')]
    available_categories = list(set(available_categories))
    return jsonify(available_categories)

@app.route('/calculate_max_days', methods=['POST'])
def calculate_max_days():
    data = request.get_json()
    city = data['city']
    selected_categories = data['selected_categories']
    places_per_day = data['places_per_day']
    
    total_places = df[(df['City'] == city) & (df['Categories'].apply(lambda x: match_categories(str(x), selected_categories)))].shape[0]
    max_days = -(-total_places // places_per_day)  # Ceiling division
    return jsonify(max_days=max_days)

@app.route('/map/<int:day>')
def map(day):
    global maps
    return render_template('map.html', map_html=maps[day-1].get_root().render())

if __name__ == '__main__':
    app.run(debug=True)