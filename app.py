import streamlit as st 
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(page_title="Phoenix Fertility Engine", layout="centered")

if st.sidebar.button("⬅️ Back to Home"):
    st.components.v1.html(
        """
        <script>
            window.parent.history.back();
        </script>
        """,
        height=0,
    )

# --- Theme and Language Toggles ---
theme = st.sidebar.radio("🌗 Choose Theme", ["Dark", "Light"])
language = st.sidebar.selectbox("🌐 Language", ["English", "தமிழ்"])

# --- Translations ---
translations = {
    "English": {
        "title": "Phoenix Fertility Engine",
        "objective": "Objective",
        "problem": "Problem Statement",
        "algorithm": "Algorithm Used",
        "start": "Start Prediction",
        "recommend": "Fertilizer Recommendations",
        "sensor": "Sensor Metrics Overview",
        "adjust": "Adjust Chemical Values",
        "graph": "Interactive Metric Graph",
        "manual": "Manual Toxicity Check",
        "predict": "Predict Toxicity",
        "average": "Average of Inputs",
        "safe": "Fertilizer is GOOD for plants.",
        "bad": "Fertilizer is BAD for plants.",
        "model": "Model Used",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "home_desc_1": "Design a machine learning system to predict fertilizer safety based on soil and chemical parameters.",
        "home_desc_2": "Farmers struggle to find safe fertilizer mixes. This system helps predict toxicity and improve crop yield.",
        "home_desc_3": "Random Forest Classifier — ensemble of decision trees with majority voting.",
        "lbl_pH": "🌊 pH Level",
        "lbl_N": "🌬️ Nitrogen",
        "lbl_P": "🔥 Phosphorus",
        "lbl_K": "🪨 Potassium",
        "lbl_OM": "🌿 Organic Matter",
        "lbl_SM": "💧 Soil Moisture",
        "lbl_PMR": "⚔️ Pest Mortality Rate",
        "lbl_PHI": "🌟 Plant Health Index",
        "lbl_remaining": "Remaining"
    },
    "தமிழ்": {
        "title": "பீனிக்ஸ் உரம் இயந்திரம்",
        "objective": "நோக்கம்",
        "problem": "சிக்கல் விளக்கம்",
        "algorithm": "பயன்படுத்தப்படும் அல்காரிதம்",
        "start": "முன்னறிதலை தொடங்கு",
        "recommend": "உர பரிந்துரைகள்",
        "sensor": "சென்சார் அளவீட்டு மேடைகள்",
        "adjust": "வேதியியல் மதிப்புகளை மாற்றவும்",
        "graph": "மெட்ரிக் வரைபடம்",
        "manual": "கைமுறை நச்சுத்தன்மை கணிப்பு",
        "predict": "நச்சுத்தன்மையை கணிக்கவும்",
        "average": "உள்ளீடுகளின் சராசரி",
        "safe": "உரம் செடிகளுக்கு நல்லது.",
        "bad": "உரம் செடிகளுக்கு தீங்கு விளைவிக்கிறது.",
        "model": "பயன்படுத்தப்பட்ட மாதிரி",
        "accuracy": "துல்லியம்",
        "precision": "நிகர்த்தன்மை",
        "home_desc_1": "மண் மற்றும் வேதியியல் அளவுருக்களின் அடிப்படையில் உரத்தின் பாதுகாப்பை கணிக்க ஒரு இயந்திர கற்றல் அமைப்பை வடிவமைத்தல்.",
        "home_desc_2": "விவசாயிகள் பாதுகாப்பான உர கலவைகளைக் கண்டறிய சிரமப்படுகிறார்கள். இந்த அமைப்பு நச்சுத்தன்மையை கணிக்கவும் பயிர் விளைச்சலை மேம்படுத்தவும் உதவுகிறது.",
        "home_desc_3": "ரேண்டம் பாரஸ்ட் வகைப்படுத்தி — பெரும்பான்மை வாக்குப்பதிவு கொண்ட முடிவு மரங்களின் தொகுப்பு.",
        "lbl_pH": "🌊 pH அளவு",
        "lbl_N": "🌬️ நைட்ரஜன்",
        "lbl_P": "🔥 பாஸ்பரஸ்",
        "lbl_K": "🪨 பொட்டாசியம்",
        "lbl_OM": "🌿 கரிமப் பொருட்கள்",
        "lbl_SM": "💧 மண் ஈரம்",
        "lbl_PMR": "⚔️ பூச்சி இறப்பு விகிதம்",
        "lbl_PHI": "🌟 தாவர ஆரோக்கிய குறியீடு",
        "lbl_remaining": "மீதமுள்ளவை"
    }
}
t = translations[language]

# --- Styling ---
bg_color = "#1e1e1e" if theme == "Dark" else "#f5f5f5"
text_color = "#ffffff" if theme == "Dark" else "#000000"
accent = "#00ff88" if theme == "Dark" else "#ff6600"

st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .circle {{ width: 80px; height: 80px; border-radius: 50%; margin: 20px auto; }}
    .green {{ background-color: #00ff88; box-shadow: 0 0 25px #00ff88; }}
    .red {{ background-color: #ff4444; box-shadow: 0 0 25px #ff4444; }}
    .fade-in {{ animation: fadeIn 1s ease-in; }}
    @keyframes fadeIn {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
    .mythic-panel {{ background-color: rgba(255,255,255,0.05); border: 1px solid {accent}; border-radius: 10px; padding: 15px; margin-bottom: 20px; }}
    .phoenix-logo {{ animation: pulse 2s infinite; margin: auto; display: block; }}
    @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); filter: drop-shadow(0 0 10px {accent}); }} 100% {{ transform: scale(1); }} }}
    </style>
""", unsafe_allow_html=True)

# --- Navigation ---
st.sidebar.markdown("### 🔀 Navigate")
nav_options = [
    ("🏠 Home", "home"), 
    ("📡 Sensor Metrics", "sensor"), 
    ("🎛️ Adjust Values", "adjust"), 
    ("📊 Show Graph", "graph"), 
    ("🧪 Manual Toxicity Check", "manual")
]
for label, page in nav_options:
    if st.sidebar.button(label):
        st.session_state.page = page
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- Load Data & Train Model ---
try:
    data = pd.read_csv(r"fertilizer_ph_data.csv") 
    le = LabelEncoder()
    data['Toxicity'] = le.fit_transform(data['Toxicity'])
    X = data.drop('Toxicity', axis=1)
    y = data['Toxicity']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = round(accuracy_score(y, y_pred) * 100, 2)
    precision = round(precision_score(y, y_pred, average='macro') * 100, 2)
    default = X.iloc[0]
except FileNotFoundError:
    st.error("Error: 'fertilizer_ph_data.csv' not found. Please check the file path.")
    st.stop()

fertilizer_image = "https://www.gardendesign.com/pictures/images/900x705Max/site_3/applying-fertilizer-blue-trowel-fertilizing-tomato-plant-shutterstock-com_15275.jpg"

# --- Helper Functions ---
def recommend_fertilizer(pH, N, P, K, OM, SM, PMR, PHI):
    recs = []
    if language == "English":
        if pH < 5.5: recs.append("🧪 Add lime to reduce acidity.")
        elif pH > 7.5: recs.append("🧪 Add sulfur or compost to lower alkalinity.")
        if N < 1.5: recs.append("🌬️ Use urea or ammonium sulfate.")
        if P < 1.0: recs.append("🔥 Apply single super phosphate.")
        if K < 1.5: recs.append("🪨 Use muriate of potash or composted banana peels.")
        if OM < 3.0: recs.append("🌿 Add organic manure or vermicompost.")
        if SM < 40: recs.append("💧 Improve irrigation or add mulch.")
        if PMR < 75: recs.append("⚔️ Use neem-based biopesticides.")
        if PHI < 80: recs.append("🌟 Apply balanced NPK and monitor stress.")
    else:  # Tamil
        if pH < 5.5: recs.append("🧪 அமிலத்தன்மையை குறைக்க சுண்ணாம்பு சேர்க்கவும்.")
        elif pH > 7.5: recs.append("🧪 காரத்தன்மையை குறைக்க சல்பர் அல்லது கம்போஸ்ட் சேர்க்கவும்.")
        if N < 1.5: recs.append("🌬️ யூரியா அல்லது அமோனியம் சல்பேட் பயன்படுத்தவும்.")
        if P < 1.0: recs.append("🔥 சிங்கிள் சூப்பர் பாஸ்பேட் பயன்படுத்தவும்.")
        if K < 1.5: recs.append("🪨 முரியேட் ஆஃப் பொட்டாஷ் அல்லது வாழைப்பழ தோல் கம்போஸ்ட் பயன்படுத்தவும்.")
        if OM < 3.0: recs.append("🌿 இயற்கை உரம் அல்லது வெர்மி கம்போஸ்ட் சேர்க்கவும்.")
        if SM < 40: recs.append("💧 நீர்ப்பாசனத்தை மேம்படுத்தவும் அல்லது மல்ச் பயன்படுத்தவும்.")
        if PMR < 75: recs.append("⚔️ வேப்பை அடிப்படையிலான உயிர் பூச்சிக்கொல்லிகளை பயன்படுத்தவும்.")
        if PHI < 80: recs.append("🌟 சமநிலை NPK உரம் பயன்படுத்தி செடி அழுத்தத்தை கண்காணிக்கவும்.")
    return recs

def show_prediction_block(values):
    sample = [values]
    prediction = model.predict(sample)
    result = le.inverse_transform(prediction)[0]
    average = round(sum(values) / len(values), 2)
    st.markdown(f"**📊 {t['average']}:** {average}")
    if result == "Safe":
        st.markdown('<div class="circle green"></div>', unsafe_allow_html=True)
        st.success(f"✅ {t['safe']}")
    else:
        st.markdown('<div class="circle red"></div>', unsafe_allow_html=True)
        st.error(f"❌ {t['bad']}")
    st.markdown(f"**{t['model']}:** Random Forest Classifier")
    st.markdown(f"**{t['accuracy']}:** {accuracy}%  |  **{t['precision']}:** {precision}%")
    recs = recommend_fertilizer(*values)
    st.markdown(f"### 🌿 {t['recommend']}:")
    for r in recs:
        st.markdown(f"- {r}")

# --- Page Logic ---
if st.session_state.page == "home":
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Phoenix-Fantasy.svg/800px-Phoenix-Fantasy.svg.png", width=120)
    st.markdown(f"<h1 style='text-align:center;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.image(fertilizer_image, caption="Applying fertilizer to tomato plant 🌱", use_container_width=True)
    st.markdown(f"### 🧭 {t['objective']}")
    st.markdown(t['home_desc_1']) 
    st.markdown(f"### 🧪 {t['problem']}")
    st.markdown(t['home_desc_2']) 
    st.markdown(f"### 🧠 {t['algorithm']}")
    st.markdown(t['home_desc_3']) 
    if st.button(f"🚀 {t['start']}"):
        st.session_state.page = "manual"

elif st.session_state.page == "sensor":
    st.header(f"📡 {t['sensor']}")
    st.image(fertilizer_image, use_container_width=True)
    chem_data = [("lbl_pH", default["pH"], 9.0), ("lbl_N", default["Nitrogen"], 5.0), ("lbl_P", default["Phosphorus"], 5.0), ("lbl_K", default["Potassium"], 5.0), ("lbl_OM", default["OrganicMatter"], 10.0), ("lbl_SM", default["SoilMoisture"], 100.0), ("lbl_PMR", default["PestMortalityRate"], 100.0), ("lbl_PHI", default["PlantHealthIndex"], 100.0)]
    for label_key, value, max_val in chem_data:
        translated_label = t[label_key]
        fig = go.Figure(go.Pie(labels=[translated_label, t['lbl_remaining']], values=[value, max_val - value], hole=0.5, marker=dict(colors=[accent, '#2e2e2e']), hoverinfo='label+percent', textinfo='value'))
        fig.update_layout(title=translated_label, template="plotly_dark" if theme == "Dark" else "plotly_white", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "adjust":
    st.header(f"🎛️ {t['adjust']}")
    st.image(fertilizer_image, use_container_width=True)
    st.session_state.pH = st.slider(t['lbl_pH'], 3.0, 9.0, float(default["pH"]))
    st.session_state.N = st.slider(t['lbl_N'], 0.0, 5.0, float(default["Nitrogen"]))
    st.session_state.P = st.slider(t['lbl_P'], 0.0, 5.0, float(default["Phosphorus"]))
    st.session_state.K = st.slider(t['lbl_K'], 0.0, 5.0, float(default["Potassium"]))
    st.session_state.OM = st.slider(t['lbl_OM'], 0.0, 10.0, float(default["OrganicMatter"]))
    st.session_state.SM = st.slider(t['lbl_SM'], 0, 100, int(default["SoilMoisture"]))
    st.session_state.PMR = st.slider(t['lbl_PMR'], 0, 100, int(default["PestMortalityRate"]))
    st.session_state.PHI = st.slider(t['lbl_PHI'], 0, 100, int(default["PlantHealthIndex"]))
    if st.button(f"🔍 {t['predict']}"):
        show_prediction_block([st.session_state.pH, st.session_state.N, st.session_state.P, st.session_state.K, st.session_state.OM, st.session_state.SM, st.session_state.PMR, st.session_state.PHI])

elif st.session_state.page == "graph":
    st.header(f"📊 {t['graph']}")
    st.image(fertilizer_image, use_container_width=True)
    x_labels = [t['lbl_pH'], t['lbl_N'], t['lbl_P'], t['lbl_K'], t['lbl_OM'], t['lbl_SM'], t['lbl_PMR'], t['lbl_PHI']]
    y_vals = [st.session_state.get("pH", default["pH"]), st.session_state.get("N", default["Nitrogen"]), st.session_state.get("P", default["Phosphorus"]), st.session_state.get("K", default["Potassium"]), st.session_state.get("OM", default["OrganicMatter"]), st.session_state.get("SM", default["SoilMoisture"]), st.session_state.get("PMR", default["PestMortalityRate"]), st.session_state.get("PHI", default["PlantHealthIndex"])]
    fig = go.Figure(data=[go.Bar(x=x_labels, y=y_vals, marker_color=[accent] * 8)])
    fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", yaxis=dict(range=[0, 100]), height=450)
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "manual":
    st.header(f"🧪 {t['manual']}")
    st.image(fertilizer_image, use_container_width=True)
    m_pH = st.number_input(t['lbl_pH'], 3.0, 9.0, value=6.5)
    m_N = st.number_input(t['lbl_N'], 0.0, 5.0, value=2.5)
    m_P = st.number_input(t['lbl_P'], 0.0, 5.0, value=2.0)
    m_K = st.number_input(t['lbl_K'], 0.0, 5.0, value=2.5)
    m_OM = st.number_input(t['lbl_OM'], 0.0, 10.0, value=5.0)
    m_SM = st.number_input(t['lbl_SM'], 0, 100, value=60)
    m_PMR = st.number_input(t['lbl_PMR'], 0, 100, value=80)
    m_PHI = st.number_input(t['lbl_PHI'], 0, 100, value=85)
    if st.button(f"🔍 {t['predict']}"):
        show_prediction_block([m_pH, m_N, m_P, m_K, m_OM, m_SM, m_PMR, m_PHI])