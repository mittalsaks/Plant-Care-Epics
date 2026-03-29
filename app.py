import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ──────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantCare AI – Crop Disease Checker",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

:root {
    --soil:   #2b2b2b;
    --bark:   #5c3d2e;
    --leaf:   #2d6a4f;
    --lime:   #52b788;
    --sun:    #f4a261;
    --cream:  #faf3e8;
    --red:    #c1121f;
    --white:  #ffffff;
    --shadow: rgba(59,42,26,0.12);
}

html, body  {
    font-family: 'Nunito', sans-serif;
    background-color: var(--cream);
    color: var(--soil);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 800px; }

.hero {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #52b788 100%);
    border-radius: 22px;
    padding: 2.2rem 2rem 1.8rem;
    color: white;
    text-align: center;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 36px var(--shadow);
}
.hero .emoji { font-size: 3rem; display: block; margin-bottom: 0.5rem; }
.hero h1     { font-size: 2.1rem; font-weight: 900; margin: 0 0 0.3rem; letter-spacing: -0.5px; }
.hero p      { font-size: 1rem; opacity: 0.88; margin: 0; }

.result-card {
    background: var(--white);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 6px 28px var(--shadow);
    margin-bottom: 1.2rem;
    border-left: 7px solid var(--leaf);
}
.result-card.danger  { border-left-color: var(--red); }
.result-card.healthy { border-left-color: var(--lime); }
.result-card.critical{ border-left-color: #8b0000; }

.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.badge-danger  { background: #fde8ea; color: var(--red); }
.badge-healthy { background: #d8f3dc; color: var(--leaf); }
.badge-critical{ background: #fde8ea; color: #8b0000; }

.crop-name    { font-size: 1.65rem; font-weight: 900; margin: 0; }
.disease-name { font-size: 1.1rem;  font-weight: 700; color: var(--bark); margin: 5px 0 0; }
.confidence   { font-size: 0.88rem; color: #999; margin-top: 6px; }

.conf-bar-bg {
    background: #eee;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.conf-bar-fill { height: 100%; border-radius: 999px; }

.tip-section {
    background: var(--white);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 14px var(--shadow);
}
.tip-section h3 {
    font-size: 1rem;
    font-weight: 800;
    margin: 0 0 0.9rem;
}
.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 0.65rem;
    font-size: 0.94rem;
    line-height: 1.6;
}
.tip-dot {
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem;
    flex-shrink: 0;
    margin-top: 1px;
    font-weight: 900;
}
.dot-red    { background:#fde8ea; color: var(--red); }
.dot-green  { background:#d8f3dc; color: var(--leaf); }
.dot-yellow { background:#fef9e7; color: #b7790a; }
.dot-blue   { background:#e8f4f8; color: #1a6985; }

.alert-box {
    background: #fff3cd;
    border: 1.5px solid #ffc107;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.93rem;
    line-height: 1.6;
}

.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #bbb;
    margin-top: 2rem;
    padding-bottom: 1.5rem;
}

            .result-card {
    color: #2b2b2b !important;
}

.tip-section {
    color: #333 !important;
}

.tip-section h3 {
    color: #1b4332 !important;
}

.tip-item {
    color: #444 !important;
}

.alert-box {
    color: #5c3d2e !important;
}
            
p, span, label, div {
    color: inherit;
}
            
.stFileUploader label {
    color: #3b2a1a !important;
    font-weight: 600;
}
            

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Disease knowledge base  (farmer-friendly language)
# ──────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": "Apple", "disease": "Apple Scab",
        "what": "A fungal infection that creates dark, rough spots on leaves and fruits, making them unsellable.",
        "signs": ["Dark olive-green or brown spots on leaves", "Rough, scabby patches on the fruit skin", "Leaves may turn yellow and fall early"],
        "causes": ["Wet and rainy weather in spring", "Water staying on leaves for too long", "Infected fallen leaves from last season left in the field"],
        "cure": ["Spray a copper-based fungicide every 7–10 days during wet weather", "Use mancozeb or captan spray as soon as spots appear", "Remove and burn all fallen leaves to stop the disease from spreading next year"],
        "prevention": ["Choose apple varieties that resist this disease", "Prune branches so air flows freely through the tree", "Never water from above — water at the base of the tree only"],
        "severity": "medium",
    },
    "Apple___Black_rot": {
        "crop": "Apple", "disease": "Black Rot",
        "what": "A fungal disease that rots the fruit and creates purple spots on leaves. Fruits shrivel and turn completely black.",
        "signs": ["Purple spots on leaves with brown centres", "Fruits develop brown rings that spread and turn black", "Shrivelled, mummy-like rotted fruits hanging on the tree"],
        "causes": ["Dead or wounded branches left on the tree", "Warm and humid weather", "Insects causing wounds on the fruit where fungus enters"],
        "cure": ["Cut and remove all dead or diseased branches immediately", "Spray captan or thiophanate-methyl fungicide", "Remove all rotted fruits from the tree and ground and burn them"],
        "prevention": ["Prune the tree every year to remove dead wood", "Keep the orchard clean of fallen fruits and leaves", "Protect fruits from insect damage"],
        "severity": "high",
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple", "disease": "Cedar Apple Rust",
        "what": "A fungal disease spread from nearby cedar trees that creates bright orange-yellow spots on leaves.",
        "signs": ["Bright orange or yellow spots on the top side of leaves", "Small tube-like growths under the leaf", "Fruits may develop pale spots"],
        "causes": ["Fungus travels from nearby juniper or cedar trees via wind", "Wet and windy spring weather helps spread it"],
        "cure": ["Spray myclobutanil or mancozeb fungicide in early spring", "Remove any cedar or juniper trees growing near your apple orchard if possible"],
        "prevention": ["Plant rust-resistant apple varieties", "Begin fungicide spray before the rainy season starts"],
        "severity": "medium",
    },
    "Apple___healthy": {
        "crop": "Apple", "disease": "Healthy Plant",
        "what": "Your apple plant looks healthy! No signs of disease detected. Keep up the good farming practices.",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Keep watering regularly at the base of the tree", "Prune dead branches every winter", "Add compost to the soil once a year"],
        "severity": "healthy",
    },
    "Blueberry___healthy": {
        "crop": "Blueberry", "disease": "Healthy Plant",
        "what": "Your blueberry plant is perfectly healthy! Keep doing what you are doing.",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Water regularly but avoid waterlogging", "Add mulch around the base to keep moisture in", "Prune old branches after harvest"],
        "severity": "healthy",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry", "disease": "Powdery Mildew",
        "what": "A fungal disease that covers leaves and young fruits with a white powdery coating, weakening the plant and reducing fruit production.",
        "signs": ["White powdery patches on leaves — especially young ones", "Leaves curl and become twisted", "Fruits may crack or not grow properly"],
        "causes": ["Dry weather with warm days and cool nights", "Too many plants crowded together with poor airflow", "High humidity without rain"],
        "cure": ["Spray wettable sulphur or potassium bicarbonate on affected areas", "Remove and destroy badly infected leaves", "Spray neem oil mixed with water (5 ml per litre) as an organic option"],
        "prevention": ["Space plants well so wind can pass through", "Avoid watering from above", "Prune the tree to open up the centre"],
        "severity": "medium",
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry", "disease": "Healthy Plant",
        "what": "Your cherry plant is in good health. No disease found.",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Water at the base, not on leaves", "Spray neem oil once a month as a precaution", "Keep fallen leaves cleaned up around the tree"],
        "severity": "healthy",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (Maize)", "disease": "Gray Leaf Spot",
        "what": "A fungal disease creating long gray or tan strips on corn leaves, reducing grain fill and yield.",
        "signs": ["Long, narrow gray or tan coloured patches on leaves", "Patches have straight edges and run along the leaf veins", "Severely infected leaves dry out completely"],
        "causes": ["Wet and humid weather for many continuous days", "Corn planted very close together without space", "Growing corn in the same field every year"],
        "cure": ["Spray mancozeb or azoxystrobin fungicide on leaves", "Start spraying early when first spots appear", "Do not wait until infection spreads to many leaves"],
        "prevention": ["Rotate crops — do not grow corn in the same field every year", "Plant disease-resistant corn varieties", "Avoid planting corn too densely"],
        "severity": "medium",
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)", "disease": "Common Rust",
        "what": "A fungal disease that creates small reddish-brown powdery bumps on corn leaves. Reduces yield in severe cases.",
        "signs": ["Small, oval, reddish-brown raised bumps on both sides of leaves", "Bumps break open and release rust-coloured powder", "Severe cases cause leaves to yellow and dry out"],
        "causes": ["Cool nights and warm days with high humidity", "Wind carrying fungal spores from other fields"],
        "cure": ["Spray mancozeb, propiconazole, or azoxystrobin fungicide", "Spray during early morning hours for best results", "Repeat spray every 10–14 days if weather stays wet"],
        "prevention": ["Plant rust-resistant hybrid corn varieties", "Plant early in the season to avoid peak rust periods"],
        "severity": "medium",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)", "disease": "Northern Leaf Blight",
        "what": "A serious fungal disease creating large cigar-shaped gray-green spots on corn leaves. Can cause heavy yield loss.",
        "signs": ["Large, long gray-green or tan spots shaped like a cigar on leaves", "Spots can grow 10–15 cm long", "Whole leaves may die in severe cases"],
        "causes": ["Moderate temperatures and wet or humid weather for long periods", "Growing the same crop every year in the same field", "Dense planting that traps moisture between plants"],
        "cure": ["Spray propiconazole or azoxystrobin-based fungicide", "Apply as soon as the first large spots appear", "Remove severely infected plants to stop the spread"],
        "prevention": ["Use certified disease-resistant corn seeds", "Rotate crops every season — alternate with soybean or wheat", "Till the soil after harvest to destroy infected crop remains"],
        "severity": "high",
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)", "disease": "Healthy Plant",
        "what": "Your corn crop looks healthy and growing well! No disease signs found.",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Water at the base, not by sprinkling on leaves", "Add nitrogen-rich fertiliser at the right growth stage", "Keep weeds away from the crop"],
        "severity": "healthy",
    },
    "Grape___Black_rot": {
        "crop": "Grape", "disease": "Black Rot",
        "what": "A very damaging fungal disease that turns grape berries completely black and shrivelled, making them worthless.",
        "signs": ["Tan or brown spots with dark edges on leaves", "Small black dots visible inside the leaf spots", "Berries turn brown, then completely black and hard like a raisin"],
        "causes": ["Warm and wet weather during the growing season", "Infected mummified berries left on the vine from last year"],
        "cure": ["Spray mancozeb or captan fungicide every 7–10 days", "Remove all mummified berries from the vine and the ground", "Prune infected shoot tips and destroy them by burning"],
        "prevention": ["Remove all dead plant material before the new season starts", "Train vines to keep leaves dry with good airflow", "Start preventive fungicide sprays before the flowering stage"],
        "severity": "high",
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape", "disease": "Black Measles (Vine Disease)",
        "what": "A serious wood disease of grapevines that blocks water flow inside the vine. Trees suddenly wilt in hot weather and slowly die over years.",
        "signs": ["Leaves suddenly wilt and dry out completely on hot days", "Tiger-stripe pattern — yellow and brown strips on leaves", "Berries develop small dark spots and may crack open"],
        "causes": ["Fungus living inside the old wood of the vine", "Pruning wounds left exposed without protection", "Old vines (10+ years) are most vulnerable"],
        "cure": ["No complete cure exists — manage by removing infected wood", "Cut back to healthy white wood and paint wounds with copper paste or tree wound sealant", "Severely affected vines may need to be removed completely"],
        "prevention": ["Always paint pruning cuts with wound sealant right after cutting", "Prune during dry weather — never during rain", "Avoid making very large pruning wounds on old vines"],
        "severity": "high",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape", "disease": "Leaf Blight",
        "what": "A fungal disease creating dark spots on leaves that can cause early leaf drop and weaken the vine before harvest.",
        "signs": ["Dark brown irregular spots on leaves", "Spots may have a yellowish ring around them", "Badly infected leaves fall off the vine early"],
        "causes": ["Warm, humid weather during the growing season", "Poor airflow in the vineyard", "Rain splashing spores from soil up onto leaves"],
        "cure": ["Spray mancozeb or copper-based fungicide", "Remove and destroy fallen infected leaves", "Improve airflow by thinning out dense vine growth"],
        "prevention": ["Avoid overhead irrigation — use drip or furrow irrigation", "Keep the vineyard floor clean of fallen leaves", "Spray preventively before the rainy season begins"],
        "severity": "medium",
    },
    "Grape___healthy": {
        "crop": "Grape", "disease": "Healthy Plant",
        "what": "Your grapevine is looking healthy! No disease signs detected.",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Prune every year to maintain good vine shape", "Check vines weekly during the monsoon for early signs of disease", "Apply compost around the base every spring"],
        "severity": "healthy",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange / Citrus", "disease": "Citrus Greening Disease",
        "what": "One of the most deadly citrus diseases in the world. Trees produce small, bitter, green fruits and slowly die. There is NO cure — act fast to protect your other trees.",
        "signs": ["Leaves turn yellow on one side of the branch while other side stays green", "Small, lopsided fruits that stay green even when ripe and taste very bitter", "Stunted and weak new growth", "Tree slowly declines and stops producing over years"],
        "causes": ["Spread by tiny jumping insects called psyllids that suck sap and carry the disease", "Infected planting material brought from outside your farm"],
        "cure": ["There is NO cure — infected trees MUST be removed and burned far away to protect other trees", "Control the psyllid insect very aggressively on all remaining healthy trees using approved insecticide"],
        "prevention": ["Spray insecticide regularly every 3 weeks to control psyllid insects", "Buy certified disease-free seedlings only from a government nursery", "Inspect your orchard every week — catch it early", "Report to your local Krishi Vigyan Kendra (KVK) immediately if you suspect this disease"],
        "severity": "critical",
    },
    "Peach___Bacterial_spot": {
        "crop": "Peach", "disease": "Bacterial Spot",
        "what": "A bacterial infection creating dark spots and holes in leaves and fruits, reducing fruit quality and market price.",
        "signs": ["Small water-soaked spots on leaves that turn brown or purple", "Spots fall out, creating a 'shot-hole' or bullet-hole look on leaves", "Sunken dark spots on fruits", "Leaves may drop early, weakening the tree"],
        "causes": ["Warm and wet weather in spring and early summer", "Wind and rain spreading bacteria from plant to plant", "Pruning wounds left open"],
        "cure": ["Spray copper-based bactericide (copper hydroxide or copper sulphate) at the first sign", "Remove badly infected leaves and fruits and destroy them", "Avoid wetting leaves when watering"],
        "prevention": ["Plant resistant peach varieties if available in your area", "Spray copper before the rainy season as a protective measure", "Avoid working in the orchard when plants are wet"],
        "severity": "medium",
    },
    "Peach___healthy": {
        "crop": "Peach", "disease": "Healthy Plant",
        "what": "Your peach tree looks great and healthy!",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Thin fruits when young so remaining fruits grow bigger and sweeter", "Water at the base only — not on leaves or fruits", "Apply balanced fertiliser after harvest to restore tree strength"],
        "severity": "healthy",
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper", "disease": "Bacterial Spot",
        "what": "A bacterial disease causing dark spots on leaves and fruits, making peppers look bad and reducing the price you can get at the market.",
        "signs": ["Small, dark, water-soaked spots on leaves", "Spots turn brown with yellow rings around them", "Dark, raised or sunken spots on the pepper fruits themselves", "Severe infection causes leaves to drop, exposing fruits to sun damage"],
        "causes": ["Rain and wind spreading bacteria from plant to plant", "Warm and wet weather conditions", "Infected seeds or seedlings brought from outside your farm"],
        "cure": ["Spray copper-based bactericide every 7–10 days", "Remove and destroy infected plants if they are severely affected", "Avoid touching healthy plants after touching sick ones — wash hands in between"],
        "prevention": ["Buy certified disease-free seeds from a trusted source", "Do not grow peppers in the same spot every year — rotate with other crops", "Avoid watering from above — use drip irrigation at the base"],
        "severity": "medium",
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper", "disease": "Healthy Plant",
        "what": "Your pepper plant is healthy and looking good!",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Stake plants when they are young so they don't bend and break", "Apply mulch around the base to keep soil moist and cool", "Feed with potassium-rich fertiliser during fruiting for bigger, better peppers"],
        "severity": "healthy",
    },
    "Potato___Early_blight": {
        "crop": "Potato", "disease": "Early Blight",
        "what": "A common fungal disease that creates dark ring-like spots on older leaves, reducing the plant's ability to produce good tubers underground.",
        "signs": ["Dark brown spots with rings inside — like a target or dartboard — on older lower leaves", "Yellow area surrounding the spots", "Infected leaves dry and fall off, starting from the bottom of the plant upward"],
        "causes": ["Warm temperatures (24–29°C) with high humidity", "Plants weakened from lack of water or nutrients", "Fungal spores in infected soil and old crop waste left in the field"],
        "cure": ["Spray mancozeb or chlorothalonil fungicide every 7–10 days", "Remove infected leaves and burn them — do not leave on the ground", "Make sure the crop gets enough fertiliser — strong healthy plants fight disease better"],
        "prevention": ["Use certified disease-free seed potatoes", "Rotate crops — do not grow potato in the same field every year", "Water in the morning so leaves dry during the day", "Avoid overly dense planting — give each plant space"],
        "severity": "medium",
    },
    "Potato___Late_blight": {
        "crop": "Potato", "disease": "Late Blight",
        "what": "A devastating disease that can destroy an ENTIRE potato field within just 3–5 days. This is extremely serious. Act immediately — do not wait even one day.",
        "signs": ["Large, dark, water-soaked spots on leaves that appear suddenly", "White fuzzy mould ring visible at the edge of spots (usually early morning)", "Stems turn dark brown and collapse", "Tubers underground develop brown rot inside"],
        "causes": ["Cool and wet weather (10–20°C with rain, fog, or heavy dew)", "Infected seed potatoes planted from last year", "Spores spread extremely fast through wind and rain — can spread across entire village in days"],
        "cure": ["⚠️ SPRAY WITHIN 24 HOURS — use metalaxyl + mancozeb (sold as Ridomil Gold) or cymoxanil immediately", "Remove ALL infected plant parts from the field and BURN them far away — do not compost or leave in field", "If more than 30% of the crop is infected, consider harvesting tubers early to save what remains"],
        "prevention": ["Plant only certified blight-free seed potatoes from a government supplier", "Spray protective fungicide (mancozeb) before the rainy season starts every year", "Use drip irrigation at the base — never sprinkler on leaves", "Ensure good drainage so water does not stand in the field"],
        "severity": "critical",
    },
    "Potato___healthy": {
        "crop": "Potato", "disease": "Healthy Plant",
        "what": "Your potato crop is growing well with no disease signs. Keep monitoring regularly.",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Earth up soil around stems as plants grow to protect tubers from sunlight", "Keep soil evenly moist — potato tubers crack if soil dries and wets in cycles", "Harvest when the vines naturally die back on their own"],
        "severity": "healthy",
    },
    "Raspberry___healthy": {
        "crop": "Raspberry", "disease": "Healthy Plant",
        "what": "Your raspberry plants are healthy and growing well!",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Remove old fruiting canes after harvest each year", "Keep the bed weed-free around plants", "Support canes with stakes or wires to prevent bending and breakage"],
        "severity": "healthy",
    },
    "Soybean___healthy": {
        "crop": "Soybean", "disease": "Healthy Plant",
        "what": "Your soybean crop looks perfectly healthy!",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Treat seeds with rhizobium bacteria before sowing for better nitrogen fixation", "Rotate with non-legume crops like wheat every year", "Control weeds in the early growth stages when plants are still small"],
        "severity": "healthy",
    },
    "Squash___Powdery_mildew": {
        "crop": "Squash / Pumpkin / Gourd", "disease": "Powdery Mildew",
        "what": "A fungal disease that covers leaves with a white powder coating, weakening the plant and reducing fruit production and size.",
        "signs": ["White powdery coating on the top surface of leaves", "Leaves turn yellow under the white patches", "Badly affected leaves dry out and curl up"],
        "causes": ["Dry weather with high humidity — common when nights get cooler", "Plants crowded together without space for airflow"],
        "cure": ["Spray wettable sulphur or potassium bicarbonate solution on all leaf surfaces", "Neem oil spray (5 ml per litre of water) works well for organic farmers — spray every 5 days", "Remove and burn badly affected leaves"],
        "prevention": ["Space plants well — at least 1 metre between plants", "Water at the base of the plant in the morning — never in the evening", "Plant mildew-resistant varieties if available in your area"],
        "severity": "medium",
    },
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry", "disease": "Leaf Scorch",
        "what": "A fungal disease creating small dark spots that make leaves look burned or scorched. Weakens the plant and reduces fruit yield.",
        "signs": ["Small, dark purple spots on upper leaf surface", "Spots expand and their centres turn brownish-gray", "Severely affected leaves turn reddish-brown and dry up completely"],
        "causes": ["Wet weather and high humidity during the season", "Infected older leaves spreading spores to new leaves", "Poor airflow between crowded plants"],
        "cure": ["Spray captan or myclobutanil fungicide every 7–10 days", "Remove older infected leaves from the plant at the base", "Improve airflow by thinning out crowded plants"],
        "prevention": ["Plant in raised beds for better water drainage", "Avoid watering from above — use drip irrigation at the base", "Replace old plants that are more than 3 years old with fresh young seedlings"],
        "severity": "medium",
    },
    "Strawberry___healthy": {
        "crop": "Strawberry", "disease": "Healthy Plant",
        "what": "Your strawberry plants look fresh and healthy!",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Use straw mulch under plants to keep fruits clean and soil moist", "Remove extra runners regularly to keep energy focused on fruiting", "Feed with balanced fertiliser after each harvest"],
        "severity": "healthy",
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato", "disease": "Bacterial Spot",
        "what": "A bacterial disease causing small dark spots on leaves and fruits, making tomatoes unsellable at market.",
        "signs": ["Small, dark brown, water-soaked spots on leaves surrounded by yellow rings", "Dark, raised spots on green fruits", "Severely infected leaves fall off, exposing fruits to sunburn and cracking"],
        "causes": ["Warm, wet, rainy weather conditions", "Wind and rain splashing bacteria from plant to plant rapidly", "Farmers working in wet fields spread bacteria on hands and tools"],
        "cure": ["Spray copper hydroxide or copper sulphate bactericide every 7 days", "Remove and burn badly infected plants from the field", "Stop all overhead watering immediately — switch to watering at the base only"],
        "prevention": ["Buy certified disease-free seeds or seedlings", "Stake plants to keep leaves and fruits off the ground", "Do not work in the field when leaves are still wet with dew or rain", "Rotate tomato with other crops every 2–3 years"],
        "severity": "medium",
    },
    "Tomato___Early_blight": {
        "crop": "Tomato", "disease": "Early Blight",
        "what": "A fungal disease that creates dark ring-like spots on older tomato leaves, reducing the plant's strength and fruit yield.",
        "signs": ["Dark brown spots with concentric rings (like a dartboard target) starting on older lower leaves", "Yellow area around each spot", "Lower leaves die and fall off first", "Dark sunken spots may also appear on fruits near the stem"],
        "causes": ["Warm and humid weather conditions", "Plants lacking nutrition — especially nitrogen fertiliser", "Soil splashing onto leaves during heavy rain or overhead watering"],
        "cure": ["Spray mancozeb or chlorothalonil fungicide every 7–10 days", "Apply nitrogen-rich fertiliser to strengthen plants — weak plants get more disease", "Remove infected lower leaves and destroy them — do not leave on ground"],
        "prevention": ["Mulch the soil around plants to stop rain from splashing spores onto leaves", "Water at the base only — never spray water onto leaves", "Feed plants regularly with balanced fertiliser throughout the season"],
        "severity": "medium",
    },
    "Tomato___Late_blight": {
        "crop": "Tomato", "disease": "Late Blight",
        "what": "A highly destructive disease that can wipe out an ENTIRE tomato field within one week. Take urgent action TODAY.",
        "signs": ["Large, dark, water-soaked patches on leaves — appearing suddenly and spreading fast", "White fuzzy mould growth at the edge of spots on the underside of leaves (seen in early morning)", "Stems turn dark brown and collapse", "Fruits develop large dark brown patches and rot quickly"],
        "causes": ["Cool, wet weather (15–20°C) with rain or heavy dew — spreads extremely fast", "Disease in the air from nearby infected fields can reach your farm"],
        "cure": ["⚠️ URGENT — spray metalaxyl + mancozeb (Ridomil Gold) or cymoxanil TODAY — within 24 hours", "Remove ALL infected plants from the field immediately and burn them far away from your crops", "Do NOT leave infected material in the field, compost it, or throw near water sources"],
        "prevention": ["Spray protective fungicide (mancozeb) at the start of cool/rainy season every year", "Use drip irrigation — never overhead sprinklers on tomatoes", "Choose late blight resistant tomato varieties — ask your seed dealer", "Ensure good drainage so water does not pool in the field"],
        "severity": "critical",
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato", "disease": "Leaf Mold",
        "what": "A fungal disease common in humid and enclosed growing areas, covering the undersides of leaves with mold and causing them to dry out.",
        "signs": ["Pale yellow-green patches on the upper surface of leaves", "Olive-green to grayish-brown mold growth on the UNDERSIDE of leaves", "Infected leaves dry out and fall off, reducing fruit production"],
        "causes": ["Very high humidity — especially above 85%", "Poor ventilation in enclosed or polyhouse growing", "Cool nights followed by warm days"],
        "cure": ["Improve ventilation immediately — open vents or increase spacing between plants", "Spray chlorothalonil or mancozeb fungicide", "Remove and destroy infected leaves from the plants"],
        "prevention": ["Grow tomatoes in well-ventilated open areas when possible", "Avoid watering in the evening — water in the morning only", "Use mold-resistant tomato varieties for enclosed farming"],
        "severity": "medium",
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato", "disease": "Septoria Leaf Spot",
        "what": "A fungal disease that creates many small circular spots on tomato leaves, starting from the bottom of the plant and moving upward if not controlled.",
        "signs": ["Many small, round spots with dark edges and lighter centres on leaves", "Tiny dark specks visible inside each spot", "Lower leaves affected first — disease moves upward if not treated", "Heavily infected leaves turn yellow and fall off"],
        "causes": ["Wet and humid weather during the growing season", "Water splashing from soil up onto leaves during rain or watering", "Growing tomatoes in the same location year after year"],
        "cure": ["Spray mancozeb, chlorothalonil, or copper-based fungicide every 7–10 days", "Remove and destroy all infected leaves starting from the bottom of the plant", "Keep the base of plants clear of fallen leaves"],
        "prevention": ["Mulch the soil around plants to reduce soil splash onto leaves", "Rotate crops — don't grow tomatoes in the same spot every year", "Water only at the base — never from above"],
        "severity": "medium",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato", "disease": "Spider Mite Attack",
        "what": "Tiny spider-like creatures (too small to see clearly) that suck sap from leaves, causing them to look bronzed, dusty, and weak. They are not a fungus — they are pests.",
        "signs": ["Fine yellow or white speckling all over leaves — looks like someone threw dust or sand on them", "Leaves turn bronze or brown, then dry out completely", "Thin spider-like webs visible under leaves when you look closely", "Problem gets much worse in hot, dry weather"],
        "causes": ["Hot and dry weather (above 30°C) causes rapid multiplication", "Excess nitrogen fertiliser makes plants soft and very attractive to mites", "Spraying pesticides that kill the natural enemies of mites makes the problem worse"],
        "cure": ["Spray abamectin, bifenazate, or spiromesifen miticide — these are specific to mites, not normal pesticides", "Wash the undersides of leaves with a strong water spray to physically knock mites off", "Neem oil spray (organic option) every 5 days"],
        "prevention": ["Water plants regularly — mites multiply fast in dry conditions", "Avoid overusing nitrogen fertiliser", "Try to preserve natural predators by not overspraying pesticides unnecessarily"],
        "severity": "medium",
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato", "disease": "Target Spot",
        "what": "A fungal disease creating round dark spots with rings on leaves and sometimes on fruits, similar in appearance to early blight.",
        "signs": ["Round brown spots with darker rings on leaves", "Spots can also appear on stems and fruits", "Heavily infected leaves die early, reducing the plant's ability to ripen fruits"],
        "causes": ["Warm, humid, wet conditions during the growing season", "Dense planting with poor airflow between plants"],
        "cure": ["Spray azoxystrobin or chlorothalonil fungicide", "Remove infected leaves from the base of the plant upward", "Improve spacing between plants for better air movement"],
        "prevention": ["Space tomato plants at least 50–60 cm apart", "Stake plants to keep them upright and leaves off the ground", "Water at the base only — never from above"],
        "severity": "medium",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato", "disease": "Yellow Leaf Curl Virus",
        "what": "A serious virus spread by tiny white flying insects (whiteflies). Once infected, plants stop growing and produce almost no fruit. Act quickly to protect healthy plants.",
        "signs": ["Leaves curl strongly upward and turn pale yellow-green", "Leaves look very small, crinkled, and distorted", "Plant growth completely stops", "Very few or absolutely no fruits form", "Clouds of tiny white insects fly up when you shake the plant"],
        "causes": ["Spread by whitefly insects that carry the virus from sick plants to healthy ones", "There is NO cure once a plant is infected with the virus"],
        "cure": ["Remove and destroy infected plants IMMEDIATELY — before whiteflies spread the virus to more plants", "Spray imidacloprid or thiamethoxam insecticide to kill whiteflies on all remaining healthy plants"],
        "prevention": ["Place yellow sticky traps throughout the field to catch whiteflies early", "Cover seedlings with insect-proof nets until they are well established", "Spray neem oil every week as a preventive measure against whiteflies", "Buy virus-resistant tomato varieties — ask your seed dealer specifically"],
        "severity": "critical",
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato", "disease": "Mosaic Virus",
        "what": "A virus that creates a patchy mosaic pattern on leaves and reduces fruit quality and size. This virus spreads very easily through touch.",
        "signs": ["Light and dark green mosaic or mottled patchwork pattern on leaves", "Leaves look distorted, curled, or wrinkled", "Fruits may show yellow patches or be smaller than normal"],
        "causes": ["Spreads very easily by touching infected plants then healthy ones with bare hands", "Contaminated tools like knives, scissors, or stakes", "Infected seeds planted at the start of the season"],
        "cure": ["No cure for virus-infected plants — remove and burn infected plants away from the field", "Wash hands thoroughly with soap and disinfect tools with bleach water before touching other plants"],
        "prevention": ["Always wash hands with soap before working with tomato plants", "Disinfect pruning tools between every plant with bleach water", "Do not smoke near tomato plants — tobacco carries a related virus", "Buy only certified virus-free seeds"],
        "severity": "high",
    },
    "Tomato___healthy": {
        "crop": "Tomato", "disease": "Healthy Plant",
        "what": "Your tomato plant looks healthy and growing well. Excellent work!",
        "signs": [], "causes": [], "cure": [],
        "prevention": ["Stake plants when they are young before they start falling over", "Feed with potassium-rich fertiliser during the fruiting stage for better quality fruits", "Water consistently at the base — avoid wetting leaves or fruits"],
        "severity": "healthy",
    },
}

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "plantvillage_phase3_epoch25_FINAL.h5"
IMAGE_SIZE = (224, 224)
LAST_CONV  = "out_relu"

CLASS_NAMES = list(DISEASE_INFO.keys())   # order must match training

SEVERITY_META = {
    "healthy":  {"label": "✅ HEALTHY CROP",       "badge": "badge-healthy", "card": "healthy",  "bar": "#52b788"},
    "medium":   {"label": "⚠️ DISEASE DETECTED",   "badge": "badge-danger",  "card": "danger",   "bar": "#e9c46a"},
    "high":     {"label": "🚨 SERIOUS DISEASE",    "badge": "badge-danger",  "card": "danger",   "bar": "#f4a261"},
    "critical": {"label": "🆘 CRITICAL — ACT NOW", "badge": "badge-critical","card": "critical", "bar": "#c1121f"},
}

# ──────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def preprocess(pil_img):
    img = pil_img.resize(IMAGE_SIZE).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

import tensorflow as tf
import numpy as np

def make_gradcam(img_array, model, last_conv_layer_name=None):
    try:
        import numpy as np
        import tensorflow as tf

        # Ensure batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        # Auto-detect last conv layer
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break

        # Create grad model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Forward pass
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            predictions = tf.convert_to_tensor(predictions)

            pred_index = tf.argmax(predictions[0]).numpy()   # ✅ FIX

            class_channel = predictions[:, pred_index]       # ✅ FIX

        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)

        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        # Weighted sum
        heatmap = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(heatmap, axis=-1)

        # ReLU
        heatmap = tf.maximum(heatmap, 0)

        # Normalize safely
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            return None

        heatmap /= max_val

        return heatmap.numpy()

    except Exception as e:
        st.error(f"GradCAM Error: {e}")
        return None

def overlay_gradcam(pil_img, heatmap, alpha=0.4):
    import numpy as np
    import cv2

    img = np.array(pil_img.resize(IMAGE_SIZE))

    # ✅ Handle None case
    if heatmap is None:
        return img

    # ✅ Force numpy conversion
    if not isinstance(heatmap, np.ndarray):
        heatmap = np.array(heatmap)

    # ✅ Remove extra dimensions safely
    heatmap = np.squeeze(heatmap)

    # ✅ Prevent invalid shapes
    if heatmap.ndim != 2:
        heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1])

    # ✅ Normalize safely
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    # ✅ Resize
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)

    # ✅ Convert to heatmap image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# ──────────────────────────────────────────────────────────────
# HTML helpers
# ──────────────────────────────────────────────────────────────
def tips_html(items, dot_class, symbol):
    rows = "".join(
        f'<div class="tip-item">'
        f'<div class="tip-dot {dot_class}">{symbol}</div>'
        f'<div>{t}</div></div>'
        for t in items
    )
    return rows

# ══════════════════════════════════════════════════════════════
# UI START
# ══════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
  <span class="emoji">🌾</span>
  <h1>PlantCare AI</h1>
  <p>Take a photo of your crop leaf — we will tell you what is wrong and how to fix it</p>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader(
    "📷  Upload a clear photo of one leaf (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding:2rem 0; color:#aaa; font-size:0.95rem;">
        👆 Upload a clear photo of a single leaf to get your result.<br>
        <small>Tip: Good lighting and a steady hand gives the best results.</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="footer">PlantCare AI• AI Crop Disease Checker • Always confirm with your local agriculture officer before buying medicines.</div>', unsafe_allow_html=True)
    st.stop()

# ── Load & predict ─────────────────────────────────────────
pil_image = Image.open(uploaded).convert("RGB")

with st.spinner("🔍 Analysing your leaf — please wait..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"❌ Model file **{MODEL_PATH}** not found in this folder.\n\n"
            f"Make sure the model file is in the same folder as `app.py`.\n\n`{e}`"
        )
        st.stop()

    img_arr    = preprocess(pil_image)
    preds      = model.predict(img_arr, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100

    pred_class = CLASS_NAMES[pred_idx]
    info       = DISEASE_INFO.get(pred_class)

    if info is None:
        st.error("Detected class not found in knowledge base. Please check CLASS_NAMES order.")
        st.stop()

    severity = info["severity"]
    meta     = SEVERITY_META[severity]

    heatmap     = make_gradcam(img_arr, model)
    if heatmap is not None:
        overlay_img = overlay_gradcam(pil_image, heatmap)
    else:
        overlay_img = np.array(pil_image.resize(IMAGE_SIZE))

# ── Images side by side ────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 Your Leaf Photo**")
    st.image(pil_image, use_container_width=True)
with col2:
    st.markdown("**🔬 Disease Location on Leaf**")
    st.image(overlay_img, use_container_width=True)
    st.caption("🔴 Red/orange areas = where the disease is concentrated")

st.markdown("<br>", unsafe_allow_html=True)

# ── Result card ────────────────────────────────────────────
bar_color = meta["bar"]
bar_w     = round(confidence)

st.markdown(f"""
<div class="result-card {meta['card']}">
    <span class="badge {meta['badge']}">{meta['label']}</span>
    <div class="crop-name">🌿 {info['crop']}</div>
    <div class="disease-name">{info['disease']}</div>
    <div class="confidence">The AI is <b>{confidence:.1f}%</b> confident in this result</div>
    <div class="conf-bar-bg" style="margin-top:8px;">
        <div class="conf-bar-fill" style="width:{bar_w}%; background:{bar_color};"></div>
    </div>
    <p style="margin-top:1rem; font-size:0.97rem; line-height:1.7; color:#444;">{info['what']}</p>
</div>
""", unsafe_allow_html=True)

# ── Healthy branch ─────────────────────────────────────────
if severity == "healthy":
    if info["prevention"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>💡 Tips to Keep Your Crop Healthy</h3>
            {tips_html(info['prevention'], 'dot-green', '✓')}
        </div>
        """, unsafe_allow_html=True)


st.write("Heatmap type:", type(heatmap))
if heatmap is not None:
    st.write("Heatmap shape:", np.shape(heatmap))
# ── Disease branch ─────────────────────────────────────────
else:
    if info["signs"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>🔎 Signs You Will See on Your Crop</h3>
            {tips_html(info['signs'], 'dot-red', '!')}
        </div>
        """, unsafe_allow_html=True)

    if info["causes"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>🌧️ Why This is Happening</h3>
            {tips_html(info['causes'], 'dot-yellow', '?')}
        </div>
        """, unsafe_allow_html=True)

    if info["cure"]:
        st.markdown(f"""
        <div class="tip-section" style="border-left:4px solid #1a6985;">
            <h3>💊 What To Do Right Now — Step by Step</h3>
            {tips_html(info['cure'], 'dot-blue', '→')}
        </div>
        """, unsafe_allow_html=True)

    if info["prevention"]:
        st.markdown(f"""
        <div class="tip-section">
            <h3>🛡️ How to Prevent This Next Season</h3>
            {tips_html(info['prevention'], 'dot-green', '✓')}
        </div>
        """, unsafe_allow_html=True)

    # Advisory note for serious cases
    if severity in ("high", "critical"):
        st.markdown("""
        <div class="alert-box">
            <b>⚠️ Important Advisory:</b> This is a serious crop disease.
            Please contact your local Agriculture Officer, Krishi Vigyan Kendra (KVK),
            or call the Kisan Call Centre at <b>1800-180-1551 (free call)</b>
            for guidance on the right medicines available in your area.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PlantCare AI 🌾 • Powered by AI (MobileNetV2) trained on PlantVillage Dataset •
    38 crop diseases supported • Always confirm with your local agriculture expert before purchasing medicines.
</div>
""", unsafe_allow_html=True)