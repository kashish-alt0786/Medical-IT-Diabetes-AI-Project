import sys
import os
from pathlib import Path

# --- MUST BE FIRST, before any `from ui.` imports ---
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import joblib

# Now these will work
from ui.sidebar import show_sidebar
from ui.input_form import show_input_form

from explainability import create_shap_plot
from results import show_results
from predictor import predict_risk

from config import (
    FEATURE_NAMES,
    DEFAULT_INSULIN,
    DEFAULT_SKIN_THICKNESS
)

from preprocessing import (
    calculate_bmi,
    map_blood_pressure,
    map_family_history,
    estimate_glucose
)

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# --- Language Dictionary ---
LANGUAGES = {
    "English": "en",
    "हिन्दी": "hi",
    "한국어": "ko",
}

TEXT = {
    "en": {
        "title": "🩺 Diabetes Risk Predictor",
        "subtitle": "Explainable AI for preventive health screening",
        "disclaimer": "⚠ **Disclaimer:** This tool predicts statistical risk only. It is NOT a medical diagnosis. Always consult a healthcare professional.",
        "model_perf": "📊 Model Performance",
        "recall": "Recall", "recall_desc": "Optimized for Medical Screening",
        "auc": "AUC-ROC", "auc_desc": "Clinical threshold > 0.7",
        "accuracy": "Accuracy", "accuracy_desc": "Tuned via GridSearchCV",
        "project_links": "Project Links:", "tech_stack": "Tech Stack:",
        "health_info": "📝 Health Information",
        "age": "Age", "age_help": "Your current age",
        "blood_sugar_header": "🩸 Blood Sugar Level",
        "have_test": "Do you have a blood sugar test result?",
        "no_test": "No, I don't know", "yes_test": "Yes, I have a test result",
        "type_fbs": "Type your Fasting Blood Sugar number",
        "fbs_help": "Check your lab report for 'Fasting Blood Sugar' or 'FBS'",
        "no_test_title": "**No test? Answer these 3 questions:**",
        "thirsty": "I feel very thirsty all the time",
        "tired": "I feel tired even after sleeping 8 hours",
        "pee": "I go to the bathroom to pee very often",
        "est_85": "Estimated blood sugar: 85 (Normal range)",
        "est_105": "Estimated blood sugar: 105 (Slightly high)",
        "est_120": "Estimated blood sugar: 120 (High)",
        "est_140": "Estimated blood sugar: 140 (Very high)",
        "cheat_sheet": "📋 What do these numbers mean? Click for examples",
        "cheat_table": """
        | Your Number | What It Means | Real Life Example |
        | --- | --- | --- |
        | **70-99** | Normal | Most healthy people when they wake up |
        | **100-125** | Pre-diabetes | Like a warning sign. Change diet now |
        | **126+** | Diabetes | Doctor will ask for 2nd test to confirm |

        **How to get this number:**
        1. **Lab Test:** Book "Fasting Blood Sugar" test. Don't eat 8 hours before.
        2. **Home Meter:** Test first thing in morning before eating/drinking water is OK.
        3. **No Test:** Use the 3 questions above. It's just an estimate.
        """,
        "body_measure": "📏 Body Measurements",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "bmi_calc": "Calculated BMI:",
        "normal": "Normal",
        "overweight": "Overweight",
        "obese": "Obese",
        "health_bg": "❤ Health Background",
        "bp_status": "Blood pressure status",
        "bp_low": "Low", "bp_normal": "Normal", "bp_high": "High Blood Pressure", "bp_not_sure": "Not Sure",
        "pregnancies": "Number of Pregnancies", "preg_help": "Enter 0 if male or not applicable",
        "family": "Do any parents, siblings, or children have diabetes?",
        "family_no": "No", "family_1": "Yes, 1 family member", "family_2": "Yes, 2 or more family members", "family_not_sure": "Not Sure",
        "family_help": "This helps assess genetic risk",
        "analyze_btn": "🔍 Analyze My Risk",
        "result_header": "📋 Risk Assessment Result",
        "low_risk": "Low Risk:", "low_desc": "Your statistical risk is low. Maintaining a healthy lifestyle is recommended.",
        "mod_risk": "Moderate Risk:", "mod_desc": "Your statistical risk is moderate. Consider lifestyle monitoring and regular health checkups.",
        "high_risk": "High Risk:", "high_desc": "Your statistical risk is elevated. Consulting a healthcare professional for further testing is strongly advised.",
        "how_calc": "🔬 How This Result Was Calculated",
        "chart_caption": "The chart shows which factors increased or decreased your risk score:",
        "chart_xlabel": "Impact on Model Output",
        "chart_title": "Feature Impact on Risk Prediction",
        "red_bars": "Red bars increase risk. Green bars decrease risk.",
        "health_tips": "💡 Personalized Health Tips: What to Eat & Avoid",
        "tips_desc": "**Based on general diabetes prevention guidelines from WHO & ADA:**",
        "low_tips_title": "**Keep doing what you're doing! Focus on maintaining:**",
        "low_tips": """
        **✅ EAT MORE:**
        - Whole grains: Brown rice, oats, whole wheat roti
        - Vegetables: Spinach, broccoli, carrots, bitter gourd (karela)
        - Fruits: Apple, orange, guava, berries - eat whole, not juice
        - Protein: Lentils (dal), chickpeas, fish, eggs, paneer
        - Healthy fats: Nuts, seeds, olive oil

        **🏃 LIFESTYLE:** 30 min walking 5 days/week + 7-8 hours sleep
        """,
        "mod_tips_title": "**Small changes make a big difference. Start here:**",
        "mod_tips": """
        **✅ EAT MORE:**
        - High fiber: Oats, daliya, rajma, vegetables with every meal
        - Protein: Grilled chicken/fish, tofu, sprouts - helps control sugar spikes
        - Good snacks: Handful of almonds, cucumber, roasted chana

        **❌ REDUCE/AVOID:**
        - Sugary drinks: Soda, packaged juice, sweet tea/coffee
        - White carbs: White bread, white rice, maida - switch to brown/whole grain
        - Sweets: Mithai, cakes, cookies - limit to special occasions
        - Fried food: Samosa, pakora, chips - try air-fried or baked

        **🏃 LIFESTYLE:** 45 min brisk walk daily + reduce sitting time. Check blood sugar every 6 months.
        """,
        "high_tips_title": "**Important: Please consult a doctor. These tips support medical care:**",
        "high_tips": """
        **✅ PRIORITIZE THESE FOODS:**
        - Non-starchy vegetables: 50% of your plate - spinach, cauliflower, bhindi, lauki
        - Lean protein: 25% of plate - grilled fish, chicken breast, dal, paneer
        - Complex carbs: 25% of plate - quinoa, brown rice, millets (bajra, jowar)
        - Best fruits: Jamun, guava, apple, pear - 1 serving/day

        **❌ STRICTLY LIMIT:**
        - Sugar: Table sugar, honey, jaggery, sweets, desserts
        - Refined carbs: White rice, white bread, pasta, potatoes
        - Packaged food: Biscuits, namkeen, instant noodles - high hidden sugar/salt
        - Fruit juice: Even 100% juice spikes sugar - eat whole fruit instead
        - Alcohol: Can cause dangerous sugar drops

        **🏃 LIFESTYLE:** Doctor-supervised exercise plan. Monitor blood sugar as advised. Never skip meals.
        """,
        "note": "**Note:** These are general guidelines, not personalized medical advice. Portion size and specific needs vary. Consult a dietitian or doctor for a custom meal plan.",
        "footer_disc": "Disclaimer: For educational and informational purposes only. Not medical advice. Model trained on Pima Indian Diabetes Dataset.",
        "footer_built": "Built with Python, Streamlit, XGBoost, SHAP.",
        "limitations": "Model Limitations: Trained on Pima Indian female dataset. May be less accurate for males or other ethnicities. Intended for initial screening only."
    },
    "hi": {
        "title": "🩺 मधुमेह जोखिम भविष्यवक्ता",
        "subtitle": "निवारक स्वास्थ्य जांच के लिए व्याख्यात्मक एआई",
        "disclaimer": "⚠ **अस्वीकरण:** यह उपकरण केवल सांख्यिकीय जोखिम की भविष्यवाणी करता है। यह चिकित्सा निदान नहीं है। हमेशा स्वास्थ्य पेशेवर से सलाह लें।",
        "model_perf": "📊 मॉडल प्रदर्शन",
        "recall": "रिकॉल", "recall_desc": "चिकित्सा स्क्रीनिंग के लिए अनुकूलित",
        "auc": "AUC-ROC", "auc_desc": "नैदानिक सीमा > 0.7",
        "accuracy": "सटीकता", "accuracy_desc": "GridSearchCV द्वारा ट्यून किया गया",
        "project_links": "प्रोजेक्ट लिंक:", "tech_stack": "तकनीकी स्टैक:",
        "health_info": "📝 स्वास्थ्य जानकारी",
        "age": "आयु", "age_help": "आपकी वर्तमान आयु",
        "blood_sugar_header": "🩸 रक्त शर्करा स्तर",
        "have_test": "क्या आपके पास रक्त शर्करा परीक्षण परिणाम है?",
        "no_test": "नहीं, मुझे नहीं पता", "yes_test": "हाँ, मेरे पास परीक्षण परिणाम है",
        "type_fbs": "अपना खाली पेट रक्त शर्करा संख्या टाइप करें",
        "fbs_help": "अपनी लैब रिपोर्ट में 'Fasting Blood Sugar' या 'FBS' देखें",
        "no_test_title": "**कोई परीक्षण नहीं? इन 3 प्रश्नों के उत्तर दें:**",
        "thirsty": "मुझे हर समय बहुत प्यास लगती है",
        "tired": "8 घंटे सोने के बाद भी थकान महसूस होती है",
        "pee": "मैं बहुत बार पेशाब करने जाता हूं",
        "est_85": "अनुमानित रक्त शर्करा: 85 (सामान्य सीमा)",
        "est_105": "अनुमानित रक्त शर्करा: 105 (थोड़ा अधिक)",
        "est_120": "अनुमानित रक्त शर्करा: 120 (उच्च)",
        "est_140": "अनुमानित रक्त शर्करा: 140 (बहुत अधिक)",
        "cheat_sheet": "📋 इन संख्याओं का क्या मतलब है? उदाहरण के लिए क्लिक करें",
        "cheat_table": """
        | आपकी संख्या | इसका मतलब | वास्तविक जीवन उदाहरण |
        | --- | --- | --- |
        | **70-99** | सामान्य | ज्यादातर स्वस्थ लोग जब वे जागते हैं |
        | **100-125** | प्री-डायबिटीज | चेतावनी संकेत की तरह। अभी आहार बदलें |
        | **126+** | डायबिटीज | डॉक्टर पुष्टि के लिए दूसरा परीक्षण पूछेंगे |

        **यह संख्या कैसे प्राप्त करें:**
        1. **लैब टेस्ट:** "Fasting Blood Sugar" टेस्ट बुक करें। 8 घंटे पहले न खाएं।
        2. **होम मीटर:** सुबह खाने से पहले सबसे पहले परीक्षण करें/पानी पीना ठीक है।
        3. **कोई टेस्ट नहीं:** ऊपर दिए गए 3 प्रश्नों का उपयोग करें। यह सिर्फ एक अनुमान है।
        """,
        "body_measure": "📏 शरीर माप",
        "height": "ऊंचाई (cm)", "weight": "वजन (kg)",
        "bmi_calc": "गणना किया गया BMI:", "normal": "सामान्य", "overweight": "अधिक वजन", "obese": "मोटापा",
        "health_bg": "❤ स्वास्थ्य पृष्ठभूमि",
        "bp_status": "रक्तचाप स्थिति",
        "bp_low": "कम", "bp_normal": "सामान्य", "bp_high": "उच्च रक्तचाप", "bp_not_sure": "पता नहीं",
        "pregnancies": "गर्भधारण की संख्या", "preg_help": "यदि पुरुष हैं या लागू नहीं है तो 0 दर्ज करें",
        "family": "क्या माता-पिता, भाई-बहन या बच्चों को मधुमेह है?",
        "family_no": "नहीं", "family_1": "हाँ, 1 सदस्य", "family_2": "हाँ, 2 या अधिक सदस्य", "family_not_sure": "पता नहीं",
        "family_help": "यह आनुवंशिक जोखिम का आकलन करने में मदद करता है",
        "analyze_btn": "🔍 मेरे जोखिम का विश्लेषण करें",
        "result_header": "📋 जोखिम मूल्यांकन परिणाम",
        "low_risk": "कम जोखिम:", "low_desc": "आपका सांख्यिकीय जोखिम कम है। स्वस्थ जीवनशैली बनाए रखने की सलाह दी जाती है।",
        "mod_risk": "मध्यम जोखिम:", "mod_desc": "आपका सांख्यिकीय जोखिम मध्यम है। जीवनशैली की निगरानी और नियमित स्वास्थ्य जांच पर विचार करें।",
        "high_risk": "उच्च जोखिम:", "high_desc": "आपका सांख्यिकीय जोखिम अधिक है। आगे की जांच के लिए स्वास्थ्य पेशेवर से सलाह लेने की दृढ़ता से सलाह दी जाती है।",
        "how_calc": "🔬 यह परिणाम कैसे गणना की गई",
        "chart_caption": "चार्ट दिखाता है कि किन कारकों ने आपके जोखिम स्कोर को बढ़ाया या घटाया:",
        "chart_xlabel": "मॉडल आउटपुट पर प्रभाव",
        "chart_title": "जोखिम भविष्यवाणी पर फीचर प्रभाव",
        "red_bars": "लाल बार जोखिम बढ़ाते हैं। हरे बार जोखिम घटाते हैं।",
        "health_tips": "💡 व्यक्तिगत स्वास्थ्य सुझाव: क्या खाएं और क्या बचें",
        "tips_desc": "**WHO और ADA दिशानिर्देशों पर आधारित:**",
        "low_tips_title": "**जो कर रहे हैं वही करते रहें! बनाए रखने पर ध्यान दें:**",
        "low_tips": """
        **✅ अधिक खाएं:**
        - साबुत अनाज: ब्राउन राइस, ओट्स, साबुत गेहूं की रोटी
        - सब्जियां: पालक, ब्रोकली, गाजर, करेला
        - फल: सेब, संतरा, अमरूद, बेरी - जूस नहीं, साबुत खाएं
        - प्रोटीन: दाल, छोले, मछली, अंडे, पनीर
        - स्वस्थ वसा: नट्स, बीज, जैतून का तेल

        **🏃 जीवनशैली:** सप्ताह में 5 दिन 30 मिनट चलना + 7-8 घंटे नींद
        """,
        "mod_tips_title": "**छोटे बदलाव बड़ा फर्क लाते हैं। यहाँ से शुरू करें:**",
        "mod_tips": """
        **✅ अधिक खाएं:**
        - उच्च फाइबर: ओट्स, दलिया, राजमा, हर भोजन के साथ सब्जियां
        - प्रोटीन: ग्रिल्ड चिकन/मछली, टोफू, अंकुरित - शुगर स्पाइक को नियंत्रित करने में मदद करता है
        - अच्छे स्नैक्स: मुट्ठी भर बादाम, खीरा, भुना चना

        **❌ कम करें/बचें:**
        - मीठे पेय: सोडा, पैक जूस, मीठी चाय/कॉफी
        - सफेद कार्ब्स: सफेद ब्रेड, सफेद चावल, मैदा - ब्राउन/साबुत अनाज पर स्विच करें
        - मिठाई: मिठाई, केक, कुकीज़ - विशेष अवसरों तक सीमित करें
        - तला हुआ खाना: समोसा, पकोड़ा, चिप्स - एयर-फ्राइड या बेक्ड ट्राई करें

        **🏃 जीवनशैली:** रोज 45 मिनट तेज चलना + बैठने का समय कम करें। हर 6 महीने में ब्लड शुगर जांचें।
        """,
        "high_tips_title": "**महत्वपूर्ण: कृपया डॉक्टर से सलाह लें। ये सुझाव चिकित्सा देखभाल का समर्थन करते हैं:**",
        "high_tips": """
        **✅ इन खाद्य पदार्थों को प्राथमिकता दें:**
        - गैर-स्टार्ची सब्जियां: आपकी प्लेट का 50% - पालक, फूलगोभी, भिंडी, लौकी
        - लीन प्रोटीन: प्लेट का 25% - ग्रिल्ड मछली, चिकन ब्रेस्ट, दाल, पनीर
        - जटिल कार्ब्स: प्लेट का 25% - क्विनोआ, ब्राउन राइस, मिलेट्स (बाजरा, ज्वार)
        - सबसे अच्छे फल: जामुन, अमरूद, सेब, नाशपाती - दिन में 1 सर्विंग

        **❌ सख्ती से सीमित करें:**
        - चीनी: टेबल शुगर, शहद, गुड़, मिठाई, डेज़र्ट
        - रिफाइंड कार्ब्स: सफेद चावल, सफेद ब्रेड, पास्ता, आलू
        - पैकेज्ड फूड: बिस्किट, नमकीन, इंस्टेंट नूडल्स - छुपी चीनी/नमक अधिक
        - फलों का रस: 100% जूस भी शुगर बढ़ाता है - साबुत फल खाएं
        - शराब: खतरनाक शुगर ड्रॉप का कारण बन सकता है

        **🏃 जीवनशैली:** डॉक्टर-पर्यवेक्षित व्यायाम योजना। सलाह के अनुसार ब्लड शुगर की निगरानी करें। कभी भी भोजन न छोड़ें।
        """,
        "note": "**नोट:** ये सामान्य दिशानिर्देश हैं, व्यक्तिगत चिकित्सा सलाह नहीं। भाग का आकार और विशिष्ट आवश्यकताएं अलग-अलग होती हैं। कस्टम भोजन योजना के लिए आहार विशेषज्ञ या डॉक्टर से सलाह लें।",
        "footer_disc": "अस्वीकरण: केवल शैक्षिक और सूचनात्मक उद्देश्यों के लिए। चिकित्सा सलाह नहीं। पिमा इंडियन डायबिटीज डेटासेट पर प्रशिक्षित मॉडल।",
        "footer_built": "Python, Streamlit, XGBoost, SHAP के साथ निर्मित।",
        "limitations": "मॉडल सीमाएं: पिमा इंडियन महिला डेटासेट पर प्रशिक्षित। पुरुषों या अन्य जातियों के लिए कम सटीक हो सकता है। केवल प्रारंभिक स्क्रीनिंग के लिए।"
    },
    "ko": {
        "title": "🩺 당뇨병 위험 예측기",
        "subtitle": "예방 건강 검진을 위한 설명 가능한 AI",
        "disclaimer": "⚠ **면책조항:** 이 도구는 통계적 위험만 예측합니다. 의학적 진단이 아닙니다. 항상 의료 전문가와 상담하세요.",
        "model_perf": "📊 모델 성능",
        "recall": "재현율", "recall_desc": "의료 검진에 최적화됨",
        "auc": "AUC-ROC", "auc_desc": "임상 임계값 > 0.7",
        "accuracy": "정확도", "accuracy_desc": "GridSearchCV로 튜닝됨",
        "project_links": "프로젝트 링크:", "tech_stack": "기술 스택:",
        "health_info": "📝 건강 정보",
        "age": "나이", "age_help": "현재 나이",
        "blood_sugar_header": "🩸 혈당 수준",
        "have_test": "혈당 검사 결과가 있나요?",
        "no_test": "아니요, 모르겠습니다", "yes_test": "예, 검사 결과가 있습니다",
        "type_fbs": "공복 혈당 수치 입력",
        "fbs_help": "검사 보고서에서 'Fasting Blood Sugar' 또는 'FBS' 확인",
        "no_test_title": "**검사 없음? 다음 3가지 질문에 답하세요:**",
        "thirsty": "항상 매우 목이 마릅니다",
        "tired": "8시간 자도 피곤합니다",
        "pee": "화장실을 자주 갑니다",
        "est_85": "예상 혈당: 85 (정상 범위)",
        "est_105": "예상 혈당: 105 (약간 높음)",
        "est_120": "예상 혈당: 120 (높음)",
        "est_140": "예상 혈당: 140 (매우 높음)",
        "cheat_sheet": "📋 이 숫자는 무엇을 의미하나요? 예시 보기",
        "cheat_table": """
        | 숫자 | 의미 | 실생활 예시 |
        | --- | --- | --- |
        | **70-99** | 정상 | 아침에 깨어났을 때 대부분의 건강한 사람 |
        | **100-125** | 당뇨 전단계 | 경고 신호와 같음. 지금 식단 변경 |
        | **126+** | 당뇨병 | 의사가 확인을 위해 2차 검사 요청 |

        **이 숫자를 얻는 방법:**
        1. **실험실 검사:** "공복 혈당" 검사 예약. 8시간 전 금식.
        2. **가정용 측정기:** 아침 식사 전 가장 먼저 검사/물 마시는 건 괜찮음.
        3. **검사 없음:** 위 3가지 질문 사용. 추정치일 뿐입니다.
        """,
        "body_measure": "📏 신체 측정",
        "height": "키 (cm)", "weight": "체중 (kg)",
        "bmi_calc": "계산된 BMI:", "normal": "정상", "overweight": "과체중", "obese": "비만",
        "health_bg": "❤ 건강 배경",
        "bp_status": "혈압 상태",
        "bp_low": "낮음", "bp_normal": "정상", "bp_high": "고혈압", "bp_not_sure": "모름",
        "pregnancies": "임신 횟수", "preg_help": "남성이거나 해당없으면 0 입력",
        "family": "부모, 형제자매, 자녀 중 당뇨병이 있나요?",
        "family_no": "없음", "family_1": "예, 1명", "family_2": "예, 2명 이상", "family_not_sure": "모름",
        "family_help": "유전적 위험 평가에 도움",
        "analyze_btn": "🔍 내 위험도 분석",
        "result_header": "📋 위험 평가 결과",
        "low_risk": "낮은 위험:", "low_desc": "통계적 위험이 낮습니다. 건강한 생활 습관 유지를 권장합니다.",
        "mod_risk": "중간 위험:", "mod_desc": "통계적 위험이 중간입니다. 생활습관 모니터링과 정기 검진을 고려하세요.",
        "high_risk": "높은 위험:", "high_desc": "통계적 위험이 높습니다. 추가 검사를 위해 의료 전문가 상담을 강력히 권장합니다.",
        "how_calc": "🔬 결과 계산 방법",
        "chart_caption": "차트는 위험 점수를 높이거나 낮춘 요인을 보여줍니다:",
        "chart_xlabel": "모델 출력에 미치는 영향",
        "chart_title": "위험 예측에 대한 특성 영향",
        "red_bars": "빨간 막대는 위험을 증가시킵니다. 녹색 막대는 위험을 감소시킵니다.",
        "health_tips": "💡 개인 맞춤형 건강 팁: 먹어야 할 것과 피해야 할 것",
        "tips_desc": "**WHO 및 ADA 지침 기반:**",
        "low_tips_title": "**계속 유지하세요! 다음에 집중하세요:**",
        "low_tips": """
        **✅ 더 많이 먹기:**
        - 통곡물: 현미, 귀리, 통밀빵
        - 채소: 시금치, 브로콜리, 당근, 여주
        - 과일: 사과, 오렌지, 구아바, 베리류 - 주스가 아닌 통째로 섭취
        - 단백질: 렌틸콩, 병아리콩, 생선, 달걀, 두부
        - 건강한 지방: 견과류, 씨앗, 올리브 오일

        **🏃 생활습관:** 주 5일 30분 걷기 + 7-8시간 수면
        """,
        "mod_tips_title": "**작은 변화가 큰 차이를 만듭니다. 여기서 시작하세요:**",
        "mod_tips": """
        **✅ 더 많이 먹기:**
        - 고섬유질: 귀리, 강낭콩, 매끼 채소
        - 단백질: 구운 닭고기/생선, 두부, 콩나물 - 혈당 스파이크 조절 도움
        - 좋은 간식: 아몬드 한 줌, 오이, 구운 병아리콩

        **❌ 줄이기/피하기:**
        - 단 음료: 탄산음료, 포장 주스, 단 차/커피
        - 흰 탄수화물: 흰 빵, 흰 쌀, 밀가루 - 현미/통곡물로 전환
        - 단 음식: 과자, 케이크, 쿠키 - 특별한 경우에만 제한
        - 튀긴 음식: 감자튀김, 칩 - 에어프라이어 또는 구운 것 시도

        **🏃 생활습관:** 매일 45분 빠르게 걷기 + 앉아있는 시간 줄이기. 6개월마다 혈당 검사.
        """,
        "high_tips_title": "**중요: 의사와 상담하세요. 이 팁은 의료 관리를 지원합니다:**",
        "high_tips": """
        **✅ 우선순위 식품:**
        - 비전분 채소: 접시의 50% - 시금치, 콜리플라워, 애호박
        - 저지방 단백질: 접시의 25% - 구운 생선, 닭가슴살, 두부
        - 복합 탄수화물: 접시의 25% - 퀴노아, 현미, 잡곡
        - 최고의 과일: 블루베리, 구아바, 사과, 배 - 하루 1회 제공량

        **❌ 엄격히 제한:**
        - 설탕: 설탕, 꿀, 시럽, 디저트
        - 정제 탄수화물: 흰 쌀, 흰 빵, 파스타, 감자
        - 가공 식품: 비스킷, 인스턴트 라면 - 숨겨진 설탕/소금 높음
        - 과일 주스: 100% 주스도 혈당 급상승 - 통째로 섭취
        - 알코올: 위험한 혈당 강하 유발 가능

        **🏃 생활습관:** 의사 감독 운동 계획. 권고에 따라 혈당 모니터링. 절대 식사 거르지 않기.
        """,
        "note": "**참고:** 이는 일반적인 지침이며 개인 맞춤 의료 조언이 아닙니다. 분량과 특정 요구사항은 다릅니다. 맞춤 식단은 영양사나 의사와 상담하세요.",
        "footer_disc": "면책조항: 교육 및 정보 제공 목적으로만 사용됩니다. 의학적 조언이 아닙니다. Pima Indian 당뇨병 데이터셋으로 훈련된 모델.",
        "footer_built": "Python, Streamlit, XGBoost, SHAP으로 구축.",
        "limitations": "모델 한계: Pima Indian 여성 데이터셋으로 훈련됨. 남성이나 다른 인종에는 덜 정확할 수 있음. 초기 검진용으로만 사용."
    }
}

# --- Language Selector ---
lang = st.sidebar.selectbox(
    "🌐 Language / भाषा / 언어",
    list(LANGUAGES.keys())
)
t = TEXT[LANGUAGES[lang]]


# --- Header ---
st.title(t["title"])
st.caption(t["subtitle"])
st.warning(t["disclaimer"])


# --- Load Model ---
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model, FEATURE_NAMES


model, feature_names = load_model()


# --- Sidebar ---
show_sidebar(t)


# --- User Input Form ---
age, glucose, bmi, bp, pregnancies, dpf, insulin, skin = show_input_form(t)


# --- Prediction ---
if st.button(
    t["analyze_btn"],
    type="primary",
    use_container_width=True,
):

    risk_percent, risk_level, color, input_df, top_reasons = predict_risk(
        model=model,
        feature_names=feature_names,
        pregnancies=pregnancies,
        glucose=glucose,
        bp=bp,
        skin=skin,
        insulin=insulin,
        bmi=bmi,
        dpf=dpf,
        age=age,
    )

    st.markdown("---")

    # --- Results ---
    show_results(t, risk_percent, risk_level, top_reasons, input_df)
    st.markdown("---")

       # --- SHAP Explainability ---
    st.subheader(t.get("how_calc", "How is this calculated?"))
    st.caption(t.get("chart_caption", "SHAP explanation of the prediction"))

    fig = create_shap_plot(
        top_reasons,
        t
    )
    st.pyplot(fig, use_container_width=True)

    st.caption(t.get("red_bars", "Red bars increase risk, blue bars decrease risk."))

    # --- Footer ---
    st.divider()
    st.caption(t.get("footer_disc", "This is not a medical diagnosis."))
    st.caption(t.get("footer_built", "Built with XGBoost + SHAP"))
    st.caption(t.get("limitations", "For educational purposes only."))
