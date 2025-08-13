import gradio as gr
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic

# --- CONFIGURATION ---
PARQUET_FILE = "flattened_by_seaons_with_userinfo.parquet"
MODEL_FILE = "lgbm_dish_recommender_by_seasons.pkl"
USER_LOCATIONS_FILE = "Users.csv"
RESTAURANT_FILE = "restaurants_with_cuisine.csv"
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
N_DISHES = 15

dish_to_cuisine = {
    'dish1': 'Indian', 'dish2': 'Indian', 'dish3': 'Indian', 'dish4': 'Indian', 'dish5': 'Indian',
    'dish6': 'Japanese', 'dish7': 'Japanese', 'dish8': 'Japanese', 'dish9': 'Japanese', 'dish10': 'Japanese',
    'dish11': 'Chinese', 'dish12': 'Chinese', 'dish13': 'Chinese', 'dish14': 'Chinese', 'dish15': 'Chinese'
}
dish_to_name = {
    'dish1': "Aamras Poori", 'dish2': "Boondi Raita", 'dish3': "Pakora", 'dish4': "Sarson da Saag & Makki di Roti", 'dish5': "Pav Bhaji",
    'dish6': "Sushi Rolls", 'dish7': "Matcha Ice Cream", 'dish8': "Udon Noodles", 'dish9': "Tonkotsu Ramen", 'dish10': "Gyoza",
    'dish11': "Chilli Paneer", 'dish12': "Gobi Manchurian", 'dish13': "Chilli Mushroom", 'dish14': "American Chopsuey", 'dish15': "Schezwan Fried Rice"
}
season_months = {
    "Winter": ['Jan', 'Feb', 'Mar', 'Nov', 'Dec'],
    "Summer": ['Apr', 'May', 'Jun', 'Jul'],
    "Monsoon": ['Aug', 'Sep', 'Oct'],
}
month_to_season = {}
for season, mlist in season_months.items():
    for m in mlist:
        month_to_season[m] = season

def bitstring_to_array(s):
    return np.array([int(ch) for ch in str(s).zfill(N_DISHES)])

def get_max_len():
    max_hist = max(len(v) for v in season_months.values()) - 1
    return max_hist * N_DISHES * 2 + len(MONTHS)
MAX_FEAT_LEN = get_max_len()

def pad_features(feats, max_len):
    if len(feats) < max_len:
        feats.extend([0] * (max_len - len(feats)))
    return feats

def build_user_features(user_row, month_to_predict):
    t = MONTHS.index(month_to_predict)
    season = month_to_season[month_to_predict]
    season_month_list = [m for m in season_months[season] if m != month_to_predict]
    feats = []
    for m in season_month_list:
        feats.extend(bitstring_to_array(user_row[m]))
        feats.extend(bitstring_to_array(user_row[f"{m}_craving"]))
    month_onehot = [0]*len(MONTHS)
    month_onehot[t] = 1
    feats.extend(month_onehot)
    feats = pad_features(feats, MAX_FEAT_LEN)
    return np.array(feats).reshape(1, -1)

def recommend(user_id, month_to_predict, top_k):
    user_rows = df[df['UserID'].astype(str) == str(user_id)]
    if user_rows.empty:
        return "UserID not found.", ""
    user_row = user_rows.iloc[0]
    user_loc_row = user_locs[user_locs['UserID'].astype(str) == str(user_id)]
    if user_loc_row.empty:
        return "User location not found.", ""
    user_lat = float(user_loc_row.iloc[0]['Latitude'])
    user_lon = float(user_loc_row.iloc[0]['Longitude'])
    try:
        features = build_user_features(user_row, month_to_predict)
    except Exception as e:
        return f"Error: {e}", ""
    model = model_package['model']
    thresholds = model_package['thresholds']
    dish_names = [f'dish{i+1}' for i in range(N_DISHES)]
    proba = np.array([p[:,1] for p in model.predict_proba(features)]).T[0]
    predicted = (proba > thresholds).astype(int)
    if top_k:
        top_indices = np.argsort(proba)[-top_k:][::-1]
        recommended = [dish_names[i] for i in top_indices]
    else:
        recommended = [dish for dish, p in zip(dish_names, predicted) if p]
    recs_out = []
    for dish in recommended:
        recs_out.append(f"{dish_to_name[dish]} (Cuisine: {dish_to_cuisine[dish]})")
    # Restaurant recommendations
    rest_out = ""
    cuisines = set(dish_to_cuisine[dish] for dish in recommended)
    for cuisine in cuisines:
        subset = restaurants[restaurants['Cuisine'].str.lower() == cuisine.lower()].copy()
        subset['DistanceKM'] = subset.apply(
            lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).km, axis=1
        )
        top_nearby = subset.sort_values('DistanceKM').head(5)
        rest_out += f"\n---\nNearest {cuisine} restaurants:\n"
        for _, row in top_nearby.iterrows():
            rest_out += f"{row['RestaurantName']} — {row['DistanceKM']:.2f} km\n"
    return "\n".join(recs_out), rest_out.strip()

# Load all data at top-level
df = pd.read_parquet(PARQUET_FILE)
model_package = joblib.load(MODEL_FILE)
user_locs = pd.read_csv(USER_LOCATIONS_FILE)
restaurants = pd.read_csv(RESTAURANT_FILE)

months_dropdown = gr.Dropdown(choices=MONTHS, label="Month to Predict")
inputs = [
    gr.Textbox(label="UserID"),
    months_dropdown,
    gr.Number(label="Top K Dishes to Recommend (optional)", value=None, precision=0)
]
outputs = [
    gr.Textbox(label="Recommended Dishes"),
    gr.Textbox(label="Nearby Restaurants"),
]

with gr.Blocks(
    css="""
.gradio-container {
    background: linear-gradient(135deg, #9d50bb 0%, #6e48aa 45%, #f9d423 100%);
    min-height: 100vh;
}
h1, h2, label, .label, .title, .block-title { color: #fff !important; }
input, select { border-radius: 6px !important; }
"""
) as demo:
    gr.Markdown("<h1 style='text-align:center;'>Personalized Restaurant & Dish Recommender</h1>", elem_id="title1")
    gr.Markdown("<p style='text-align:center;'>Enter UserID and month to get season-aware recommendations, including the nearest restaurants for each predicted cuisine.</p>")
    gr.Interface(
        fn=recommend,
        inputs=inputs,
        outputs=outputs,
        allow_flagging="never"
    )

demo.launch()
