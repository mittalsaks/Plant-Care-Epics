TEXTS = {
    "en": {
        "upload": "📷 Upload a leaf image",
        "analyzing": "🔍 Analysing your leaf...",
        "severity": "📊 Severity Analysis",
        "weather": "🌦 Weather Insights",
        "voice": "🎙 Voice Assistant"
    },
    "hi": {
        "upload": "📷 पत्ती की फोटो अपलोड करें",
        "analyzing": "🔍 विश्लेषण हो रहा है...",
        "severity": "📊 रोग की गंभीरता",
        "weather": "🌦 मौसम जानकारी",
        "voice": "🎙 वॉइस सहायक"
    }
}

def t(key, lang):
    return TEXTS[lang].get(key, key)