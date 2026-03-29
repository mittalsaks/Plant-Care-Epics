"""UI strings (en/hi) and disease text lookup via English keys."""

try:
    from .disease_hi_generated import DISEASE_HI as _DISEASE_HI
except ImportError:
    _DISEASE_HI = {}

UI_EN = {
    "choose_language": "Choose language",
    "lang_english": "English",
    "lang_hindi": "हिंदी",
    "nav_dashboard": "🏠 Dashboard",
    "nav_severity": "📊 Severity Analysis",
    "nav_voice": "🎙 Voice Assistant",
    "severity": "Severity analysis",
    "severity_subtitle": "See disease patterns and risk levels — simple charts for farmers.",
    "severity_body": "This section is under construction. Soon you will see easy charts and tips here.",
    "voice_subtitle": "Ask a question by voice. Get short, practical farming advice.",
    "voice_hint": "Tap the button, wait for listening, then speak clearly.",
    "voice_start": "Start speaking",
    "voice_listening": "Listening… speak now.",
    "voice_you_said": "You said",
    "voice_ai_prefix": "Advice",
    "voice_error_understand": "Could not catch that. Try again in a quiet place.",
    "hero_title": "PlantCare AI",
    "hero_subtitle": "Take a clear photo of one leaf. We tell you what is wrong and what to do next — in simple words.",
    "section_leaf_photo": "Your leaf photo",
    "section_add_photo": "Add your leaf photo",
    "section_how_to_send": "How do you want to send the photo?",
    "btn_upload": "📁 Upload from phone or computer",
    "btn_camera": "📸 Take a photo with camera",
    "upload_image": "Choose an image file",
    "upload_prompt": "Please add a leaf photo above to start the check.",
    "preview_label": "Preview",
    "analysing": "Checking your leaf with the AI model…",
    "skeleton_hint": "Almost ready…",
    "caption_your_photo": "📷 Your leaf",
    "caption_heatmap": "🔎 Where the problem shows on the leaf",
    "caption_heatmap_detail": "Red and orange areas are where the model looked most closely.",
    "gradcam_none": "We could not draw a heat map. Showing your original photo.",
    "gradcam_error": "Heat map error",
    "overlay_error": "Could not blend heat map with your photo",
    "invalid_heatmap_shape": "Heat map shape was not valid — showing your photo only.",
    "confidence_line": 'The AI is <strong>{conf}%</strong> sure about this result.',
    "what_it_means": "In simple words",
    "severity_label_healthy": "Healthy crop",
    "severity_label_medium": "Disease found — act soon",
    "severity_label_high": "Serious disease — take action",
    "severity_label_critical": "Very serious — act today",
    "section_signs": "Signs you may see",
    "section_causes": "Why this happens",
    "section_cure": "What to do now",
    "section_prevention": "How to stop it next time",
    "section_tips": "Tips to keep your crop healthy",
    "advisory_title": "Important",
    "advisory_body": (
        "This can harm your crop badly. Please talk to your local agriculture officer, "
        "Krishi Vigyan Kendra (KVK), or call the Kisan Call Centre at "
        "<strong>1800-180-1551</strong> (toll-free) before you buy or spray medicine."
    ),
    "footer_text": (
        "PlantCare AI · Trained on PlantVillage · 38 disease types · "
        "Always confirm sprays with a local expert."
    ),
    "prediction_done": "Report is ready.",
    "model_load_error": "Could not load the AI model file. Put the `.h5` file in the project folder next to `app.py`.",
    "class_missing_error": "This result is not in our crop guide. Please check the model file.",
    "camera_internal_title": "📸 Leaf photo",
    "camera_internal_help": "Take a picture of the leaf",
    "upload_or_capture": "Leaf photo",
    "tips": "Tips to Keep Your Crop Healthy",
    "signs": "Signs You Will See on Your Crop",
    "causes": "Why This is Happening",
    "cure": "What To Do Right Now",
    "prevention": "How to Prevent This",
    "upload_label": "📤 Upload a leaf image",
    "analyzing": "🔍 Analysing your leaf — please wait...",
    "healthy": "Healthy Plant",
    "disease_detected": "Disease Detected",
    "app_title": "PlantCare AI",
}

UI_HI = {
    "choose_language": "भाषा चुनें",
    "lang_english": "English",
    "lang_hindi": "हिंदी",
    "nav_dashboard": "🏠 डैशबोर्ड",
    "nav_severity": "📊 गंभीरता विश्लेषण",
    "nav_voice": "🎙 आवाज़ सहायक",
    "severity": "गंभीरता विश्लेषण",
    "severity_subtitle": "रोग के पैटर्न और जोखिम स्तर देखें — किसानों के लिए सरल चार्ट।",
    "severity_body": "यह हिस्सा तैयार हो रहा है। जल्द ही यहाँ आसान चार्ट और सलाह दिखेंगी।",
    "voice_subtitle": "आवाज़ से सवाल पूछें। छोटा और व्यावहारिक कृषि जवाब पाएँ।",
    "voice_hint": "बटन दबाएँ, सुनने का इंतज़ार करें, फिर साफ़ बोलें।",
    "voice_start": "बोलना शुरू करें",
    "voice_listening": "सुन रहे हैं… अब बोलें।",
    "voice_you_said": "आपने कहा",
    "voice_ai_prefix": "सलाह",
    "voice_error_understand": "समझ नहीं आया। शांत जगह पर दोबारा कोशिश करें।",
    "hero_title": "प्लांटकेयर AI",
    "hero_subtitle": "एक पत्ती की साफ़ फोटो लें। हम आसान शब्दों में बताते हैं क्या समस्या है और अब क्या करें।",
    "section_leaf_photo": "आपकी पत्ती की फोटो",
    "section_add_photo": "अपनी पत्ती की फोटो जोड़ें",
    "section_how_to_send": "फोटो कैसे भेजना है?",
    "btn_upload": "📁 फोन या कंप्यूटर से अपलोड",
    "btn_camera": "📸 कैमरे से फोटो लें",
    "upload_image": "इमेज फ़ाइल चुनें",
    "upload_prompt": "शुरू करने के लिए ऊपर एक पत्ती की फोटो जोड़ें।",
    "preview_label": "पूर्वावलोकन",
    "analysing": "AI मॉडल से पत्ती की जाँच हो रही है…",
    "skeleton_hint": "थोड़ा इंतज़ार…",
    "caption_your_photo": "📷 आपकी पत्ती",
    "caption_heatmap": "🔎 पत्ती पर समस्या कहाँ दिख रही है",
    "caption_heatmap_detail": "लाल और नारंगी हिस्से वे जगह हैं जहाँ मॉडल ने सबसे ध्यान दिया।",
    "gradcam_none": "हीट मैप नहीं बन सका। आपकी मूल फोटो दिखा रहे हैं।",
    "gradcam_error": "हीट मैप त्रुटि",
    "overlay_error": "फोटो के साथ हीट मैप नहीं जोड़ सके",
    "invalid_heatmap_shape": "हीट मैप सही नहीं था — केवल आपकी फोटो दिखा रहे हैं।",
    "confidence_line": "AI इस नतीजे के बारे में <strong>{conf}%</strong> आश्वस्त है।",
    "what_it_means": "सरल भाषा में",
    "severity_label_healthy": "स्वस्थ फसल",
    "severity_label_medium": "रोग मिला — जल्दी कदम उठाएँ",
    "severity_label_high": "गंभीर रोग — कार्रवाई ज़रूरी",
    "severity_label_critical": "बहुत गंभीर — आज ही कार्रवाई",
    "section_signs": "जो लक्षण दिख सकते हैं",
    "section_causes": "यह क्यों होता है",
    "section_cure": "अभी क्या करें",
    "section_prevention": "आगे कैसे बचाव करें",
    "section_tips": "फसल स्वस्थ रखने के टिप्स",
    "advisory_title": "ज़रूरी सूचना",
    "advisory_body": (
        "यह फसल को बहुत नुकसान पहुँचा सकता है। दवा खरीदने या छिड़काव से पहले स्थानीय कृषि अधिकारी, "
        "कृषि विज्ञान केंद्र (KVK) से बात करें, या किसान कॉल सेंटर <strong>1800-180-1551</strong> "
        "(मुफ़्त) पर फोन करें।"
    ),
    "footer_text": (
        "प्लांटकेयर AI · PlantVillage पर प्रशिक्षित · 38 रोग प्रकार · "
        "छिड़काव से पहले स्थानीय विशेषज्ञ से पुष्टि करें।"
    ),
    "prediction_done": "रिपोर्ट तैयार है।",
    "model_load_error": "AI मॉडल फ़ाइल नहीं खुल सकी। `.h5` फ़ाइल `app.py` वाले फ़ोल्डर में रखें।",
    "class_missing_error": "यह नतीजा हमारे फसल गाइड में नहीं है। मॉडल फ़ाइल जाँचें।",
    "camera_internal_title": "📸 पत्ती की फोटो",
    "camera_internal_help": "पत्ती की फोटो लें",
    "upload_or_capture": "पत्ती की फोटो",
    "tips": "फसल को स्वस्थ रखने के टिप्स",
    "signs": "पौधे में दिखने वाले लक्षण",
    "causes": "यह क्यों हो रहा है",
    "cure": "अभी क्या करें",
    "prevention": "आगे कैसे बचाव करें",
    "upload_label": "📤 पत्ती की फोटो अपलोड करें",
    "analyzing": "🔍 आपकी पत्ती का विश्लेषण किया जा रहा है...",
    "healthy": "स्वस्थ पौधा",
    "disease_detected": "रोग पाया गया",
    "app_title": "प्लांटकेयर AI",
}


def t(key: str, lang: str = "en") -> str:
    """Translate UI keys and disease English strings. Unknown keys return English text."""
    if lang == "en":
        return UI_EN.get(key, key)
    return UI_HI.get(key, UI_EN.get(key, _DISEASE_HI.get(key, key)))
