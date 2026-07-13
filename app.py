import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

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
    "Français": "fr",
    "Español": "es", 
    "தமிழ்": "ta",
    "Deutsch": "de"
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
        "height": "Height (cm)", "weight": "Weight (kg)",
        "bmi_calc": "Calculated BMI:", "normal": "Normal", "overweight": "Overweight", "obese": "Obese",
        "health_bg": "❤️ Health Background",
        "bp_status": "Blood pressure status",
    },
        # --- User Input ---
st.header(t["health_info"]):

st.subheader(t["health_bg"]),
col5, col6 = st.columns(2)
bp_options = [t["bp_low"], t["bp_normal"], t["bp_high"], t["bp_not_sure"]]
bp_option = col5.selectbox(t["bp_status"], bp_options)
bp = 70 if bp_option == t["bp_low"] else 80 if bp_option == t["bp_normal"] else 100 if bp_option == t["bp_high"] else 85

pregnancies = col6.number_input(t["pregnancies"], 0, 20, 0, help=t["preg_help"])
        "pregnancies": "Number of Pregnancies", "preg_help": "Enter 0 if male or not applicable",
        "family": "Do any parents, siblings, or children have diabetes?",
        "family_no": "No", "family_1": "Yes, 1 family member", "family_2": "Yes, 2 or more family members", "family_not_sure": "Not Sure",
        "family_help": "This helps assess genetic risk",
        "analyze_btn": "🔍 Analyze My Risk",
        "result_header": "📋 Risk Assessment Result",
        "low_risk": "Lower Risk:", "low_desc": "Your statistical risk is low. Maintaining a healthy lifestyle is recommended.",
        "mod_risk": "Moderate Risk:", "mod_desc": "Your statistical risk is moderate. Consider lifestyle monitoring and regular health checkups.",
        "high_risk": "Elevated Risk:", "high_desc": "Your statistical risk is elevated. Consulting a healthcare professional for further testing is strongly advised.",
        "how_calc": "🔬 How This Result Was Calculated",
        "chart_caption": "The chart shows which factors increased or decreased your risk score:",
        "chart_xlabel": "Impact on Model Output",
        "chart_title": "Feature Impact on Risk Prediction",
        "red_bars": "Red bars increase risk. Green bars decrease risk.",
        "risk_factors": "📋 Key Risk Factors",
        "top_factors": "**Top 3 factors influencing your result:**",
        "explain_help": "*This explainability helps users and healthcare providers understand the prediction.*",
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
        "health_bg": "❤️ स्वास्थ्य पृष्ठभूमि",
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
        "risk_factors": "📋 मुख्य जोखिम कारक",
        "top_factors": "**आपके परिणाम को प्रभावित करने वाले शीर्ष 3 कारक:**",
        "explain_help": "*यह व्याख्या उपयोगकर्ताओं और स्वास्थ्य प्रदाताओं को भविष्यवाणी समझने में मदद करती है।*",
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
        "health_bg": "❤️ 건강 배경",
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
        "risk_factors": "📋 주요 위험 요인",
        "top_factors": "**결과에 영향을 준 상위 3가지 요인:**",
        "explain_help": "*이 설명은 사용자와 의료진이 예측을 이해하는 데 도움이 됩니다.*",
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
    },
    "fr": {
        "title": "🩺 Prédicteur de Risque de Diabète",
        "subtitle": "IA explicable pour le dépistage préventif",
        "disclaimer": "⚠ **Avertissement:** Cet outil prédit uniquement le risque statistique. Ce n'est PAS un diagnostic médical. Consultez toujours un professionnel de santé.",
        "model_perf": "📊 Performance du Modèle",
        "recall": "Rappel", "recall_desc": "Optimisé pour le Dépistage Médical",
        "auc": "AUC-ROC", "auc_desc": "Seuil clinique > 0.7",
        "accuracy": "Précision", "accuracy_desc": "Réglé via GridSearchCV",
        "project_links": "Liens du Projet:", "tech_stack": "Stack Technique:",
        "health_info": "📝 Informations de Santé",
        "age": "Âge", "age_help": "Votre âge actuel",
        "blood_sugar_header": "🩸 Niveau de Glycémie",
        "have_test": "Avez-vous un résultat de glycémie?",
        "no_test": "Non, je ne sais pas", "yes_test": "Oui, j'ai un résultat",
        "type_fbs": "Tapez votre glycémie à jeun",
        "fbs_help": "Vérifiez 'Fasting Blood Sugar' ou 'FBS' sur votre rapport",
        "no_test_title": "**Pas de test? Répondez à ces 3 questions:**",
        "thirsty": "J'ai très soif tout le temps",
        "tired": "Je me sens fatigué même après 8h de sommeil",
        "pee": "Je vais très souvent aux toilettes",
        "est_85": "Glycémie estimée: 85 (Normal)",
        "est_105": "Glycémie estimée: 105 (Légèrement élevée)",
        "est_120": "Glycémie estimée: 120 (Élevée)",
        "est_140": "Glycémie estimée: 140 (Très élevée)",
        "cheat_sheet": "📋 Que signifient ces chiffres? Cliquez pour des exemples",
        "cheat_table": """
        | Votre Nombre | Ce que ça Signifie | Exemple Réel |
        | --- | --- | --- |
        | **70-99** | Normal | La plupart des personnes en bonne santé au réveil |
        | **100-125** | Pré-diabète | Comme un signal d'alerte. Changez de régime maintenant |
        | **126+** | Diabète | Le médecin demandera un 2e test pour confirmer |
        
        **Comment obtenir ce nombre:**
        1. **Test Labo:** Réservez test "Glycémie à Jeun". Ne mangez pas 8h avant.
        2. **Lecteur Domicile:** Testez le matin avant de manger/boire de l'eau c'est OK.
        3. **Pas de Test:** Utilisez les 3 questions ci-dessus. C'est juste une estimation.
        """,
        "body_measure": "📏 Mesures Corporelles",
        "height": "Taille (cm)", "weight": "Poids (kg)",
        "bmi_calc": "IMC calculé:", "normal": "Normal", "overweight": "Surpoids", "obese": "Obésité",
        "health_bg": "❤️ Antécédents de Santé",
        "bp_status": "Statut de la Tension Artérielle",
        "bp_low": "Basse", "bp_normal": "Normale", "bp_high": "Hypertension", "bp_not_sure": "Incertain",
        "pregnancies": "Nombre de Grossesses", "preg_help": "Entrez 0 si homme ou non applicable",
        "family": "Parents, frères/sœurs ou enfants ont-ils le diabète?",
        "family_no": "Non", "family_1": "Oui, 1 membre", "family_2": "Oui, 2+ membres", "family_not_sure": "Incertain",
        "family_help": "Aide à évaluer le risque génétique",
        "analyze_btn": "🔍 Analyser Mon Risque",
        "result_header": "📋 Résultat de l'Évaluation du Risque",
        "low_risk": "Risque Faible:", "low_desc": "Votre risque statistique est faible. Maintenir un mode de vie sain est recommandé.",
        "mod_risk": "Risque Modéré:", "mod_desc": "Votre risque statistique est modéré. Envisagez surveillance du mode de vie et bilans réguliers.",
        "high_risk": "Risque Élevé:", "high_desc": "Votre risque statistique est élevé. Consulter un professionnel de santé est fortement conseillé.",
        "how_calc": "🔬 Comment ce Résultat a été Calculé",
        "chart_caption": "Le graphique montre quels facteurs ont augmenté ou diminué votre score:",
        "chart_xlabel": "Impact sur la Sortie du Modèle",
        "chart_title": "Impact des Caractéristiques sur la Prédiction du Risque",
        "red_bars": "Les barres rouges augmentent le risque. Les barres vertes diminuent le risque.",
        "risk_factors": "📋 Facteurs de Risque Clés",
        "top_factors": "**Top 3 des facteurs influençant votre résultat:**",
        "explain_help": "*Cette explicabilité aide utilisateurs et soignants à comprendre la prédiction.*",
        "health_tips": "💡 Conseils Santé Personnalisés: Que Manger & Éviter",
        "tips_desc": "**Basé sur les directives OMS & ADA:**",
        "low_tips_title": "**Continuez ainsi! Concentrez-vous sur le maintien:**",
        "low_tips": """
        **✅ MANGER PLUS:**
        - Céréales complètes: Riz brun, avoine, pain complet
        - Légumes: Épinards, brocoli, carottes, courge amère
        - Fruits : Pomme, orange, goyave, baies – consommez-les entiers, évitez les jus
        - Protéines : Lentilles (dal), pois chiches, poisson, œufs, paneer
        - Graisses saines : Noix, graines, huile d'olive
        
        **🏃 MODE DE VIE :** 30 min de marche 5 jours/semaine + 7 à 8 heures de sommeil
        """,
        "mod_tips_title": "**De petits changements font une grande différence. Commencez par ici :**",
        "mod_tips": """
        **✅ CONSOMMEZ DAVANTAGE DE :**
        - Aliments riches en fibres : Avoine, boulgour (daliya), haricots rouges (rajma), légumes à chaque repas
        - Protéines : Poulet ou poisson grillé, tofu, graines germées – aide à contrôler les pics de glycémie
        - Collations saines : Une poignée d'amandes, concombre, pois chiches grillés (chana)
        
        **❌ RÉDUISEZ/ÉVITEZ :**
        - Boissons sucrées : Sodas, jus industriels, thé ou café sucré
        - Glucides raffinés : Pain blanc, riz blanc, farine blanche (maida) – privilégiez les versions complètes ou intégrales
        - Sucreries : Pâtisseries traditionnelles (mithai), gâteaux, biscuits – à réserver aux occasions spéciales
        - Fritures : Samosas, pakoras, chips – préférez la cuisson à l'air chaud (friteuse sans huile) ou au four
        
        **🏃 MODE DE VIE :** 45 min de marche rapide par jour + réduisez le temps passé assis. Contrôlez votre glycémie tous les 6 mois.
        """,
        "high_tips_title": "**Important : Veuillez consulter un médecin. Ces conseils complètent le suivi médical :**",
        "high_tips": """
        **✅ PRIVILÉGIEZ CES ALIMENTS :**
        - Légumes non féculents : 50 % de l'assiette – épinards, chou-fleur, gombo (bhindi), calebasse (lauki)
        - Protéines maigres : 25 % de l'assiette – poisson grillé, blanc de poulet, lentilles (dal), paneer
        - Glucides complexes : 25 % de l'assiette – quinoa, riz complet, millets (bajra, jowar)
        - Fruits recommandés : Jamun, goyave, pomme, poire – 1 portion par jour
        
        **❌ À LIMITER STRICTEMENT :**
        - Sucre : Sucre de table, miel, sucre de canne complet (jaggery), confiseries, desserts
        - Glucides raffinés : Riz blanc, pain blanc, pâtes, pommes de terre
        - Aliments transformés : Biscuits, snacks salés (namkeen), nouilles instantanées – forte teneur en sucre/sel cachés
        - Jus de fruits : Même les jus 100 % pur jus provoquent un pic de glycémie – préférez le fruit entier
        - Alcool : Peut entraîner des chutes dangereuses du taux de sucre
        
        **🏃 MODE DE VIE :** Programme d'exercice supervisé par un médecin. Surveillez votre glycémie selon les recommandations. Ne sautez jamais de repas.
        """,
        "note": "**Remarque :** Il s'agit de recommandations générales et non de conseils médicaux personnalisés. La taille des portions et les besoins spécifiques varient. Consultez un diététicien ou un médecin pour un plan alimentaire sur mesure.",
        "footer_built": "Développé avec Python, Streamlit, XGBoost et SHAP.",
        "limitations": "Limites du modèle : Entraîné sur un jeu de données concernant des femmes amérindiennes (Pima). La précision peut être moindre pour les hommes ou d'autres groupes ethniques. Destiné uniquement à un dépistage initial."
    },
    "es": {
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
        **Cómo obtener este dato:**
        1. **Análisis de laboratorio:** Solicita una prueba de "glucosa en ayunas". No ingieras alimentos durante las 8 horas previas.
        2. **Medidor doméstico:** Realiza la prueba a primera hora de la mañana; está bien hacerla antes de comer o beber agua.
        3. **Sin prueba:** Responde a las 3 preguntas anteriores. Es solo una estimación.
        """,
        "body_measure": "📏 Medidas corporales",
        "height": "Estatura (cm)", "weight": "Peso (kg)",
        "bmi_calc": "IMC calculado:", "normal": "Normal", "overweight": "Sobrepeso", "obese": "Obesidad",
        "health_bg": "❤️ Antecedentes de salud",
        "bp_status": "Estado de la presión arterial",
        "bp_low": "Baja", "bp_normal": "Normal", "bp_high": "Presión arterial alta", "bp_not_sure": "No estoy seguro/a",
        "pregnancies": "Número de embarazos", "preg_help": "Ingresa 0 si eres hombre o si no aplica",
        "family": "¿Algún padre, hermano o hijo tiene diabetes?",
        "family_no": "No", "family_1": "Sí, 1 familiar", "family_2": "Sí, 2 o más familiares", "family_not_sure": "No estoy seguro/a",
        "family_help": "Esto ayuda a evaluar el riesgo genético",
        "analyze_btn": "🔍 Analizar mi riesgo",
        "result_header": "📋 Resultado de la evaluación de riesgo",
        "low_risk": "Riesgo bajo:", "low_desc": "Tu riesgo estadístico es bajo. Se recomienda mantener un estilo de vida saludable.",
        "mod_risk": "Riesgo moderado:", "mod_desc": "Tu riesgo estadístico es moderado. Considera vigilar tu estilo de vida y realizarte chequeos médicos regulares.",
        "high_risk": "Riesgo elevado:", "high_desc": "Tu riesgo estadístico es elevado. Se recomienda encarecidamente consultar a un profesional de la salud para realizar pruebas adicionales.",
        "how_calc": "🔬 Cómo se calculó este resultado",
        "chart_caption": "El gráfico muestra qué factores aumentaron o disminuyeron su puntuación de riesgo:",
        "chart_xlabel": "Impacto en el resultado del modelo",
        "chart_title": "Impacto de los factores en la predicción de riesgo",
        "red_bars": "Las barras rojas aumentan el riesgo. Las barras verdes disminuyen el riesgo.",
        "risk_factors": "📋 Factores de riesgo clave",
        "top_factors": "**Los 3 factores principales que influyen en su resultado:**",
        "explain_help": "*Esta explicación ayuda a los usuarios y a los profesionales de la salud a comprender la predicción.*",
        "health_tips": "💡 Consejos de salud personalizados: qué comer y qué evitar",
        "tips_desc": "**Basado en las pautas generales de prevención de la diabetes de la OMS y la ADA:**",
        "low_tips_title": "**¡Sigue así! Céntrate en mantener:**",
        "low_tips": """
        **✅ COME MÁS:**
        - Cereales integrales: arroz integral, avena, roti de trigo integral
        - Verduras: espinacas, brócoli, zanahorias, calabaza amarga (karela)
        - Frutas: manzana, naranja, guayaba, bayas (cómelas enteras, no en jugo)
        - Proteínas: lentejas (dal), garbanzos, pescado, huevos, paneer
        - Grasas saludables: frutos secos, semillas, aceite de oliva
        
        **🏃 ESTILO DE VIDA:** 30 minutos de caminata 5 días a la semana + 7-8 horas de sueño
        """,
        "mod_tips_title": "**Pequeños cambios marcan una gran diferencia. Empieza aquí:**",
        "mod_tips": """
        **✅ COME MÁS:**
        - Alto contenido en fibra: avena, daliya, rajma, verduras en cada comida
        - Proteínas: pollo o pescado a la parrilla, tofu, brotes (ayudan a controlar los picos de azúcar)
        - Snacks saludables: un puñado de almendras, pepino, garbanzos tostados (chana)
        
        **❌ REDUCE/EVITA:**
        - Bebidas azucaradas: refrescos, jugos envasados, té o café dulces
        - Carbohidratos refinados: pan blanco, arroz blanco, harina refinada (maida); cámbialos por integrales
        - Dulces: mithai, pasteles, galletas; limítalos a ocasiones especiales
        - Frituras: samosas, pakoras, papas fritas; prueba versiones hechas en freidora de aire u horneadas
        
        **🏃 ESTILO DE VIDA:** 45 minutos de caminata rápida al día + reduce el tiempo que pasas sentado. Revisa tu nivel de azúcar en sangre cada 6 meses.
        """,
        "high_tips_title": "**Importante: Consulta a un médico. Estos consejos complementan la atención médica:**",
        "high_tips": """
        **✅ PRIORIZA ESTOS ALIMENTOS:**
        - Verduras sin almidón: 50% de tu plato (espinacas, coliflor, bhindi, lauki)
        - Proteínas magras: 25% de tu plato (pescado a la parrilla, pechuga de pollo, dal, paneer)
        - Carbohidratos complejos: 25% de tu plato - quinua, arroz integral, mijo (bajra, jowar)
        - Mejores frutas: jambolán, guayaba, manzana, pera (1 porción al día)
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
    "ta": {
        "title": "🩺 நீரிழிவு நோய் அபாயத்தைக் கணிக்கும் கருவி",
        "subtitle": "தடுப்பு சுகாதாரப் பரிசோதனைக்கான விளக்கக்கூடிய AI",
        "disclaimer": "⚠ **பொறுப்புத் துறப்பு:** இந்தக் கருவி புள்ளிவிவர அடிப்படையிலான அபாயத்தை மட்டுமே கணிக்கிறது. இது மருத்துவ நோயறிதல் அல்ல. எப்போதும் ஒரு மருத்துவ நிபுணரை அணுகவும்.",
        "model_perf": "📊 மாதிரியின் செயல்திறன்",
        "recall": "Recall (மீட்புத் திறன்)", "recall_desc": "மருத்துவப் பரிசோதனைக்கு உகந்ததாக்கப்பட்டது",
        "auc": "AUC-ROC", "auc_desc": "மருத்துவ வரம்பு > 0.7",
        "accuracy": "துல்லியம்", "accuracy_desc": "GridSearchCV மூலம் மேம்படுத்தப்பட்டது",
        "project_links": "திட்ட இணைப்புகள்:", "tech_stack": "தொழில்நுட்பக் கட்டமைப்பு:",
        "health_info": "📝 சுகாதாரத் தகவல்கள்",
        "age": "வயது", "age_help": "உங்கள் தற்போதைய வயது",
        "blood_sugar_header": "🩸 இரத்தச் சர்க்கரை அளவு",
        "have_test": "உங்களிடம் இரத்தச் சர்க்கரை பரிசோதனை முடிவு உள்ளதா?",
        "no_test": "இல்லை, எனக்குத் தெரியாது", "yes_test": "ஆம், என்னிடம் பரிசோதனை முடிவு உள்ளது",
        "type_fbs": "உண்ணாவிரத இரத்தச் சர்க்கரை (Fasting Blood Sugar) அளவை உள்ளிடவும்",
        "fbs_help": "உங்கள் ஆய்வக அறிக்கையில் 'Fasting Blood Sugar' அல்லது 'FBS' என்பதைப் பார்க்கவும்",
        "no_test_title": "**பரிசோதனை இல்லையா? இந்த 3 கேள்விகளுக்குப் பதிலளிக்கவும்:**",
        "thirsty": "எப்போதும் அதிக தாகமாக உணர்கிறேன்",
        "tired": "8 மணிநேரம் தூங்கிய பிறகும் சோர்வாக உணர்கிறேன்",
        "pee": "அடிக்கடி சிறுநீர் கழிக்கச் செல்கிறேன்",
        "est_85": "மதிப்பிடப்பட்ட இரத்தச் சர்க்கரை: 85 (சாதாரண அளவு)",
        "est_105": "மதிப்பிடப்பட்ட இரத்தச் சர்க்கரை: 105 (சற்று அதிகம்)",
        "est_120": "மதிப்பிடப்பட்ட இரத்தச் சர்க்கரை: 120 (அதிகம்)",
        "est_140": "மதிப்பிடப்பட்ட இரத்தச் சர்க்கரை: 140 (மிக அதிகம்)",
        "cheat_sheet": "📋 இந்த எண்கள் எதைக் குறிக்கின்றன? உதாரணங்களுக்கு இங்கே கிளிக் செய்யவும்",
        "cheat_table": """
        | உங்கள் எண் | இதன் பொருள் | நிஜ வாழ்க்கை உதாரணம் |
        | --- | --- | --- |
        | **70-99** | இயல்பான அளவு | ஆரோக்கியமான பெரும்பாலானோருக்கு, தூங்கி எழுந்தவுடன் இருக்கும் அளவு |
        | **100-125** | நீரிழிவு நோய்க்கு முந்தைய நிலை (Pre-diabetes) | ஒரு எச்சரிக்கை அறிகுறி போன்றது; இப்போதே உணவுப் பழக்கத்தை மாற்றிக்கொள்ள வேண்டும் |
        | **126+** | நீரிழிவு நோய் | உறுதிப்படுத்த மருத்துவர் இரண்டாவது பரிசோதனையை மேற்கொள்ளச் சொல்வார் 
        **இந்த அளவைப் பெறுவது எப்படி:**
        1. **ஆய்வகப் பரிசோதனை:** "Fasting Blood Sugar" (உண்ணாவிரத இரத்தச் சர்க்கரை) பரிசோதனைக்கு முன்பதிவு செய்யவும். பரிசோதனைக்கு முந்தைய 8 மணிநேரம் எதுவும் சாப்பிடக்கூடாது.
        2. **வீட்டுப் பரிசோதனைக் கருவி:** காலையில் எழுந்தவுடன், உணவு அல்லது தண்ணீர் அருந்துவதற்கு முன் பரிசோதிப்பது சிறந்தது.
        3. **பரிசோதனை இல்லையெனில்:** மேலே உள்ள 3 கேள்விகளைப் பயன்படுத்தவும். இது ஒரு தோராயமான மதிப்பீடு மட்டுமே.
        """,
        "body_measure": "📏 உடல் அளவீடுகள்",
        "height": "உயரம் (செ.மீ)", "weight": "எடை (கி.கி)",
        "bmi_calc": "கணக்கிடப்பட்ட BMI:", "normal": "சாதாரண அளவு", "overweight": "அதிக எடை", "obese": "உடல் பருமன்",
        "health_bg": "❤️ சுகாதாரப் பின்னணி",
        "bp_status": "இரத்த அழுத்த நிலை",
        "bp_low": "குறைவு", "bp_normal": "சாதாரண அளவு", "bp_high": "அதிக இரத்த அழுத்தம்", "bp_not_sure": "தெரியவில்லை",
        "pregnancies": "கர்ப்பங்களின் எண்ணிக்கை", "preg_help": "ஆண் அல்லது பொருந்தாது எனில் 0 என உள்ளிடவும்",
        "family": "பெற்றோர், உடன்பிறப்புகள் அல்லது குழந்தைகளுக்கு நீரிழிவு நோய் உள்ளதா?",
        "family_no": "இல்லை", "family_1": "ஆம், குடும்பத்தில் ஒருவர்", "family_2": "ஆம், குடும்பத்தில் இருவர் அல்லது அதற்கு மேற்பட்டோர்", "family_not_sure": "தெரியவில்லை",
        "family_help": "இது மரபணு ரீதியான ஆபத்தை மதிப்பிட உதவுகிறது",
        "analyze_btn": "🔍 எனது ஆபத்து அளவை ஆராய்க",
        "result_header": "📋 ஆபத்து மதிப்பீட்டு முடிவு",
        "low_risk": "குறைந்த ஆபத்து:", "low_desc": "புள்ளிவிவரப்படி உங்களுக்கு ஆபத்து குறைவு. ஆரோக்கியமான வாழ்க்கை முறையைப் பின்பற்றுவது பரிந்துரைக்கப்படுகிறது.",
        "mod_risk": "மிதமான ஆபத்து:", "mod_desc": "புள்ளிவிவரப்படி உங்களுக்கு மிதமான ஆபத்து உள்ளது. வாழ்க்கை முறையைக் கண்காணிப்பதையும், வழக்கமான மருத்துவப் பரிசோதனைகளை மேற்கொள்வதையும் கருத்தில் கொள்ளவும்.",
        "high_risk": "அதிகரித்த ஆபத்து:", "high_desc": "புள்ளிவிவரப்படி உங்களுக்கு ஆபத்து அதிகமாக உள்ளது. மேலதிக பரிசோதனைகளுக்கு ஒரு சுகாதார நிபுணரை அணுகுமாறு கடுமையாகப் பரிந்துரைக்கப்படுகிறது.",
        "how_calc": "🔬 இந்த முடிவு எவ்வாறு கணக்கிடப்பட்டது",
        "chart_caption": "எந்த காரணிகள் உங்கள் ஆபத்து மதிப்பெண்ணை அதிகரித்தது அல்லது குறைத்தது என்பதை விளக்கப்படம் காட்டுகிறது:",
        "chart_xlabel": "மாதிரி வெளியீட்டின் மீதான தாக்கம்",
        "chart_title": "ஆபத்து கணிப்பில் காரணிகளின் தாக்கம்",
        "red_bars": "சிவப்பு நிறப் பட்டைகள் ஆபத்தை அதிகரிக்கின்றன. பச்சை நிறப் பட்டைகள் ஆபத்தைக் குறைக்கின்றன.",
        "health_tips": "💡 தனிப்பயனாக்கப்பட்ட சுகாதார குறிப்புகள்: என்ன சாப்பிட வேண்டும் & தவிர்க்க வேண்டும்",
        "tips_desc": "**WHO மற்றும் ADA வழிகாட்டுதல்களின் அடிப்படையில்:**",
        "low_tips_title": "**இப்போதைய நல்ல பழக்கங்களைத் தொடருங்கள்! இவற்றைக் கடைப்பிடிப்பதில் கவனம் செலுத்துங்கள்:**",
        "low_tips": """
        **✅ அதிகம் உண்ண வேண்டியவை:**
        - முழு தானியங்கள்: கைக்குத்தல் அரிசி, ஓட்ஸ், முழு கோதுமை சப்பாத்தி
        - காய்கறிகள்: கீரை வகைகள், ப்ரோக்கோலி, கேரட், பாகற்காய்
        - பழங்கள்: ஆப்பிள், ஆரஞ்சு, கொய்யா, பெர்ரி வகைகள் - பழச்சாறாக அல்லாமல் முழுப் பழமாக உண்ணவும்
        - புரதம்: பருப்பு வகைகள், கொண்டைக்கடலை, மீன், முட்டை, பனீர்
        - ஆரோக்கியமான கொழுப்புகள்: நட்ஸ், விதைகள், ஆலிவ் எண்ணெய்
        
        **🏃 வாழ்க்கை முறை:** வாரத்தில் 5 நாட்கள் 30 நிமிடம் நடைப்பயிற்சி + 7-8 மணிநேர தூக்கம்
        """,
        "mod_tips_title": "**சிறிய மாற்றங்கள் பெரிய பலனைத் தரும். இதிலிருந்து தொடங்குங்கள்:**",
        "mod_tips": """
        **✅ அதிகம் உண்ண வேண்டியவை:**
        - அதிக நார்ச்சத்து: ஓட்ஸ், கோதுமை ரவை, ராஜ்மா, ஒவ்வொரு வேளை உணவிலும் காய்கறிகள்
        - புரதம்: கிரில் செய்யப்பட்ட சிக்கன்/மீன், டோஃபு, முளைகட்டிய தானியங்கள் - ரத்த சர்க்கரை அளவு திடீரென உயர்வதைக் கட்டுப்படுத்த உதவும்
        - ஆரோக்கியமான சிற்றுண்டிகள்: ஒரு கைப்பிடி பாதாம், வெள்ளரிக்காய், வறுத்த கொண்டைக்கடலை
        
        **❌ குறைக்க/தவிர்க்க வேண்டியவை:**
        - சர்க்கரை கலந்த பானங்கள்: சோடா, பாக்கெட் பழச்சாறு, இனிப்பு கலந்த டீ/காபி
        - வெள்ளை மாவு சார்ந்த உணவுகள்: வெள்ளை பிரட், வெள்ளை அரிசி, மைதா - கைக்குத்தல் அரிசி அல்லது முழு தானியங்களுக்கு மாறவும்
        - இனிப்புகள்: மிட்டாய், கேக், குக்கீஸ் - விசேஷ நாட்களில் மட்டும் உட்கொள்ளவும்
        - பொரித்த உணவுகள்: சமோசா, பக்கோடா, சிப்ஸ் - 'ஏர்-ஃப்ரை' அல்லது பேக் செய்யப்பட்டவற்றை முயற்சிக்கவும்
        
        **🏃 வாழ்க்கை முறை:** தினமும் 45 நிமிடம் வேகமான நடைப்பயிற்சி + அமர்ந்திருக்கும் நேரத்தைக் குறைத்தல். 6 மாதங்களுக்கு ஒருமுறை ரத்த சர்க்கரை அளவை பரிசோதிக்கவும்.
        """,
        "high_tips_title": "**முக்கியம்: மருத்துவரை அணுகவும். இந்தக் குறிப்புகள் மருத்துவ சிகிச்சைக்குத் துணையாக அமையும்:**",
        "high_tips": """
        **✅ இந்த உணவுகளுக்கு முன்னுரிமை அளிக்கவும்:**
        - மாவுச்சத்து குறைவான காய்கறிகள்: தட்டில் 50% இருக்க வேண்டும் - கீரை, காலிஃபிளவர், வெண்டைக்காய், சுரைக்காய்
        - கொழுப்பு குறைந்த புரதம்: தட்டில் 25% இருக்க வேண்டும் - கிரில் செய்யப்பட்ட மீன், சிக்கன் மார்புப் பகுதி, பருப்பு, பனீர்
        - சிக்கலான மாவுச்சத்து: தட்டில் 25% இருக்க வேண்டும் - கினோவா, கைக்குத்தல் அரிசி, சிறுதானியங்கள் (கம்பு, சோளம்)
        - சிறந்த பழங்கள்: நாவல் பழம், கொய்யா, ஆப்பிள், பேரிக்காய் - ஒரு நாளைக்கு ஒரு பங்கு
        
        **❌ கண்டிப்பாகத் தவிர்க்கவும்:**
        - சர்க்கரை: சர்க்கரை, தேன், வெல்லம், இனிப்புகள், இனிப்புப் பண்டங்கள்
        - சுத்திகரிக்கப்பட்ட மாவுச்சத்து: வெள்ளை அரிசி, வெள்ளை ரொட்டி, பாஸ்தா, உருளைக்கிழங்கு
        - பொட்டலங்களில் அடைக்கப்பட்ட உணவு: பிஸ்கட்டுகள், நம்கீன், இன்ஸ்டன்ட் நூடுல்ஸ் - அதிக மறைமுக சர்க்கரை/உப்பு கொண்டது
        - பழச்சாறு: 100% பழச்சாறு கூட சர்க்கரை அளவை திடீரென அதிகரிக்கும் - அதற்குப் பதிலாக முழுப் பழங்களைச் சாப்பிடுங்கள்
        - மதுபானம்: ஆபத்தான சர்க்கரை வீழ்ச்சியை ஏற்படுத்தக்கூடும்
        
        **🏃 வாழ்க்கை முறை:** மருத்துவரின் மேற்பார்வையில் உடற்பயிற்சித் திட்டம். அறிவுறுத்தப்பட்டபடி இரத்தச் சர்க்கரை அளவைக் கண்காணிக்கவும். ஒருபோதும் உணவைத் தவிர்க்காதீர்கள்.
        """,
        "note": "**குறிப்பு:** இவை பொதுவான வழிகாட்டுதல்கள், தனிப்பயனாக்கப்பட்ட மருத்துவ ஆலோசனை அல்ல. பகுதி அளவு மற்றும் குறிப்பிட்ட தேவைகள் மாறுபடும். தனிப்பயனாக்கப்பட்ட உணவுத் திட்டத்திற்கு உணவியல் நிபுணர் அல்லது மருத்துவரை அணுகவும்.",
        "footer_disc": "பொறுப்புத் துறப்பு: கல்வி மற்றும் தகவல் நோக்கங்களுக்காக மட்டுமே. மருத்துவ ஆலோசனை அல்ல. பிமா இந்திய நீரிழிவு தரவுத்தொகுப்பில் பயிற்சி பெற்ற மாதிரி.",
        "footer_built": "பைதான், ஸ்ட்ரீம்லிட், எக்ஸ்ஜிபூஸ்ட், ஷாப் ஆகியவற்றைக் கொண்டு உருவாக்கப்பட்டது.",
        "limitations": "மாதிரியின் வரம்புகள்: பிமா இந்தியப் பெண் தரவுத்தொகுப்பில் பயிற்சி பெற்றது. ஆண்கள் அல்லது பிற இனத்தவருக்கு துல்லியம் குறைவாக இருக்கலாம். ஆரம்பகட்ட ஆய்வுக்கு மட்டுமே உத்தேசிக்கப்பட்டது."
    },
    "de": {
        "title": "🩺 Diabetes-Risiko-Vorhersage",
        "subtitle": "Erklärbare KI für präventive Gesundheitsuntersuchungen",
        "disclaimer": "⚠ **Hinweis:** Dieses Tool berechnet lediglich ein statistisches Risiko. Es handelt sich NICHT um eine medizinische Diagnose. Konsultieren Sie bei Fragen immer medizinisches Fachpersonal.",
        "model_perf": "📊 Modellleistung",
        "recall": "Sensitivität (Recall)", "recall_desc": "Optimiert für medizinisches Screening",
        "auc": "AUC-ROC", "auc_desc": "Klinischer Schwellenwert > 0,7",
        "accuracy": "Genauigkeit", "accuracy_desc": "Optimiert mittels GridSearchCV",
        "project_links": "Projekt-Links:", "tech_stack": "Technologie-Stack:",
        "health_info": "📝 Gesundheitsinformationen",
        "age": "Alter", "age_help": "Ihr aktuelles Alter",
        "blood_sugar_header": "🩸 Blutzuckerspiegel",
        "have_test": "Liegt Ihnen ein Blutzucker-Testergebnis vor?",
        "no_test": "Nein, ich weiß es nicht", "yes_test": "Ja, ich habe ein Testergebnis",
        "type_fbs": "Geben Sie Ihren Nüchternblutzucker-Wert ein",
        "fbs_help": "Suchen Sie in Ihrem Laborbericht nach „Nüchternblutzucker“ oder „FBS“",
        "no_test_title": "**Kein Test? Beantworten Sie diese 3 Fragen:**",
        "thirsty": "Ich habe ständig großen Durst",
        "tired": "Ich fühle mich auch nach 8 Stunden Schlaf müde",
        "pee": "Ich muss sehr häufig zur Toilette",
        "est_85": "Geschätzter Blutzucker: 85 (Normalbereich)",
        "est_105": "Geschätzter Blutzucker: 105 (Leicht erhöht)",
        "est_120": "Geschätzter Blutzucker: 120 (Erhöht)",
        "est_140": "Geschätzter Blutzucker: 140 (Sehr hoch)",
        "cheat_sheet": "📋 Was bedeuten diese Zahlen? Hier klicken für Beispiele",
        "cheat_table": """
        | Ihr Wert | Bedeutung | Beispiel aus dem Alltag |
        | --- | --- | --- |
        | **70–99** | Normal | Bei den meisten gesunden Menschen nach dem Aufwachen |
        | **100–125** | Prädiabetes | Wie ein Warnsignal; Ernährung jetzt umstellen |
        | **126+** | Diabetes | Der Arzt wird einen zweiten Test zur Bestätigung anordnen |
        
        **So ermitteln Sie diesen Wert:**
        1. **Labortest:** Buchen Sie einen Test auf „Nüchternblutzucker“. Essen Sie 8 Stunden vorher nichts.
        2. **Messgerät für zu Hause:** Messen Sie direkt morgens vor dem Essen; Wassertrinken ist erlaubt.
        3. **Kein Test:** Nutzen Sie die drei oben genannten Fragen. Dies ist nur eine Schätzung.
        """,
        "body_measure": "📏 Körpermaße",
        "height": "Körpergröße (cm)", "weight": "Gewicht (kg)",
        "bmi_calc": "Berechneter BMI:", "normal": "Normal", "overweight": "Übergewicht", "obese": "Adipositas",
        "health_bg": "❤️ Gesundheitlicher Hintergrund",
        "bp_status": "Blutdruckstatus",
        "bp_low": "Niedrig", "bp_normal": "Normal", "bp_high": "Hoher Blutdruck", "bp_not_sure": "Unsicher",
        "pregnancies": "Anzahl der Schwangerschaften", "preg_help": "Geben Sie 0 ein, wenn männlich oder nicht zutreffend",
        "family": "Haben Eltern, Geschwister oder Kinder Diabetes?",
        "family_no": "Nein", "family_1": "Ja, 1 Familienmitglied", "family_2": "Ja, 2 oder mehr Familienmitglieder", "family_not_sure": "Unsicher",
        "family_help": "Dies hilft bei der Einschätzung des genetischen Risikos",
        "analyze_btn": "🔍 Risiko analysieren",
        "result_header": "📋 Ergebnis der Risikobewertung",
        "low_risk": "Niedriges Risiko:", "low_desc": "Ihr statistisches Risiko ist gering. Ein gesunder Lebensstil wird empfohlen.",
        "mod_risk": "Mäßiges Risiko:", "mod_desc": "Ihr statistisches Risiko ist mäßig. Achten Sie auf Ihren Lebensstil und regelmäßige Gesundheitsuntersuchungen.",
        "high_risk": "Erhöhtes Risiko:", "high_desc": "Ihr statistisches Risiko ist erhöht. Es wird dringend empfohlen, für weitere Untersuchungen eine medizinische Fachkraft zu konsultieren.",
        "how_calc": "🔬 Wie dieses Ergebnis berechnet wurde",
        "chart_caption": "Das Diagramm zeigt, welche Faktoren Ihren Risikowert erhöht oder verringert haben:",
        "chart_xlabel": "Auswirkung auf das Modellergebnis",
        "chart_title": "Einfluss der Faktoren auf die Risikovorhersage",
        "red_bars": "Rote Balken erhöhen das Risiko. Grüne Balken verringern das Risiko.",
        "health_tips": "💡 Personalisierte Gesundheitstipps: Was Sie essen und meiden sollten",
        "tips_desc": "**Basierend auf allgemeinen Richtlinien der WHO und ADA zur Diabetes-Prävention:**",
        "low_tips_title": "**Machen Sie weiter so! Achten Sie darauf, Folgendes beizubehalten:**",
        "low_tips": """
        **✅ MEHR DAVON ESSEN:**
        - Vollkornprodukte: Naturreis, Hafer, Vollkorn-Roti
        - Gemüse: Spinat, Brokkoli, Karotten, Bittermelone (Karela)
        - Obst: Äpfel, Orangen, Guaven, Beeren – als ganze Frucht essen, nicht als Saft
        - Proteine: Linsen (Dal), Kichererbsen, Fisch, Eier, Paneer
        - Gesunde Fette: Nüsse, Samen, Olivenöl
        
        **🏃 LEBENSSTIL:** 30 Minuten Gehen an 5 Tagen pro Woche + 7–8 Stunden Schlaf
        """,
        "mod_tips_title": "**Kleine Änderungen bewirken viel. Fangen Sie hier an:**",
        "mod_tips": """
        **✅ MEHR DAVON ESSEN:**
        - Ballaststoffreich: Hafer, geschroteter Weizen (Daliya), Kidneybohnen (Rajma), Gemüse zu jeder Mahlzeit
        - Proteine: Gegrilltes Hähnchen/Fisch, Tofu, Sprossen – hilft, Blutzuckerspitzen zu kontrollieren
        - Gute Snacks: Eine Handvoll Mandeln, Gurke, geröstete Kichererbsen (Chana)
        
        **❌ REDUZIEREN/MEIDEN:**
        - Zuckerhaltige Getränke: Limonaden, Säfte aus der Packung, gesüßter Tee/Kaffee
        - Weißmehlprodukte: Weißbrot, weißer Reis, raffiniertes Mehl (Maida) – auf Vollkorn umsteigen
        - Süßigkeiten: Indische Süßspeisen (Mithai), Kuchen, Kekse – auf besondere Anlässe beschränken
        - Frittiertes: Samosas, Pakoras, Chips – stattdessen Heißluftfritteuse oder Backofen nutzen
        
        **🏃 LEBENSSTIL:** Täglich 45 Minuten zügiges Gehen + weniger Sitzen. Alle 6 Monate den Blutzucker überprüfen lassen.
        """,
        "high_tips_title": "**Wichtig: Bitte konsultieren Sie einen Arzt. Diese Tipps ergänzen die medizinische Behandlung:**",
        "high_tips": """
        **✅ DIESE LEBENSMITTEL BEVORZUGEN:**
        - Gemüse ohne viel Stärke: 50 % des Tellers – Spinat, Blumenkohl, Okra (Bhindi), Flaschenkürbis (Lauki)
        - Fettarme Proteine: 25 % des Tellers – gegrillter Fisch, Hähnchenbrust, Linsen (Dal), Paneer
        - Komplexe Kohlenhydrate: 25 % des Tellers - Quinoa, brauner Reis, Hirse (Bajra, Jowar)
        - Beste Obstsorten: Jamun, Guave, Apfel, Birne – 1 Portion pro Tag
         **❌ STRENG EINSCHRÄNKEN:**
        - Zucker: Haushaltszucker, Honig, Jaggery, Süßigkeiten, Desserts
        - Raffinierte Kohlenhydrate: Weißer Reis, Weißbrot, Nudeln, Kartoffeln
        - Fertigprodukte: Kekse, salzige Snacks (z. B. Namkeen), Instant-Nudeln – hoher Gehalt an verstecktem Zucker/Salz
        - Fruchtsäfte: Selbst 100 % Saft lässt den Blutzuckerspiegel stark ansteigen – essen Sie stattdessen ganze Früchte
        - Alkohol: Kann gefährliche Blutzuckerabfälle verursachen
        
        **🏃 LEBENSSTIL:** Ärztlich betreuter Trainingsplan. Blutzucker wie empfohlen überwachen. Niemals Mahlzeiten ausfallen lassen.
        """,
        "note": "**Hinweis:** Dies sind allgemeine Richtlinien, keine individuelle medizinische Beratung. Portionsgrößen und spezifische Bedürfnisse variieren. Konsultieren Sie eine Ernährungsfachkraft oder einen Arzt für einen maßgeschneiderten Ernährungsplan.",
        "footer_disc": "Haftungsausschluss: Nur zu Bildungs- und Informationszwecken. Keine medizinische Beratung. Das Modell wurde mit dem „Pima Indian Diabetes Dataset“ trainiert.",
        "footer_built": "Erstellt mit Python, Streamlit, XGBoost, SHAP.",
        "limitations": "Modellbeschränkungen: Trainiert mit einem Datensatz von Frauen der Pima-Indianer. Die Genauigkeit bei Männern oder anderen ethnischen Gruppen kann geringer sein. Nur für ein erstes Screening gedacht."
    },

# --- Language Selector ---
lang = st.sidebar.selectbox("🌐 Language / भाषा / 언어", list(LANGUAGES.keys()))
t = TEXT[LANGUAGES[lang]]

# --- Header ---
st.title(t["title"])
st.caption(t["subtitle"])
st.warning(t["disclaimer"])

# --- Load/Train Model ---
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(url, names=names)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scale_pos_weight = len(y[y==0]) / len(y[y==1])
    model = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = load_model()

# --- Sidebar ---
with st.sidebar:
    st.header(t["model_perf"])
    st.metric(t["recall"], "67.3%", t["recall_desc"])
    st.metric(t["auc"], "0.76", t["auc_desc"])
    st.metric(t["accuracy"], "69.5%", t["accuracy_desc"])
    st.markdown("---")
    st.markdown(f"**{t['project_links']}**")
    st.markdown("[📓 Kaggle Research](https://www.kaggle.com/code/kashish0000/notebookb6b8ef2c97)")
    st.markdown("[💻 GitHub Code](https://github.com/kashish-alt0786/Medical-IT-Diabetes-AI-Project)")
    st.markdown("---")
    st.markdown(f"**{t['tech_stack']}** `Python` `XGBoost` `SHAP` `Streamlit`")

# --- User Input ---
st.header(t["health_info"])

col1, col2 = st.columns(2)
age = col1.number_input(t["age"], 1, 120, 30, help=t["age_help"])

# --- Blood Sugar Section ---
st.subheader(t["blood_sugar_header"])
knows_glucose = st.radio(t["have_test"], [t["no_test"], t["yes_test"]], horizontal=True)

if knows_glucose == t["yes_test"]:
    glucose = st.number_input(t["type_fbs"], 50, 300, 90, help=t["fbs_help"])
else:
    st.markdown(t["no_test_title"])
    thirsty = st.checkbox(t["thirsty"])
    tired = st.checkbox(t["tired"])
    pee = st.checkbox(t["pee"])
    symptom_count = sum([thirsty, tired, pee])
    if symptom_count == 0:
        glucose = 85
        st.success(t["est_85"])
    elif symptom_count == 1:
        glucose = 105
        st.warning(t["est_105"])
    elif symptom_count == 2:
        glucose = 120
        st.warning(t["est_120"])
    else:
        glucose = 140
        st.error(t["est_140"])

with st.expander(t["cheat_sheet"]):
    st.markdown(t["cheat_table"])

st.subheader(t["body_measure"])
col3, col4 = st.columns(2)
height = col3.number_input(t["height"], 100, 250, 165)
weight = col4.number_input(t["weight"], 30, 200, 65)
bmi = weight / ((height/100)**2)
bmi_label = t["normal"] if bmi<25 else t["overweight"] if bmi<30 else t["obese"]
st.info(f"{t['bmi_calc']} {bmi:.1f} | {bmi_label}")

st.subheader(t["health_bg"])
col5, col6 = st.columns(2)
bp_options = [t["bp_normal"], t["bp_high"], t["bp_not_sure"]]
bp_option = col5.selectbox(t["bp_status"], bp_options)
bp = 80 if bp_option == t["bp_normal"] else 100 if bp_option == t["bp_high"] else 85

pregnancies = col6.number_input(t["pregnancies"], 0, 20, 0, help=t["preg_help"])

family_options = [t["family_no"], t["family_1"], t["family_2"], t["family_not_sure"]]
family_history = st.radio(t["family"], family_options, horizontal=True, help=t["family_help"])

if family_history == t["family_no"]:
    dpf = 0.15
elif family_history == t["family_1"]:
    dpf = 0.5
elif family_history == t["family_2"]:
    dpf = 1.2
else:
    dpf = 0.3

insulin = 80
skin = 20

st.divider()

# --- Prediction ---
if st.button(t["analyze_btn"], type="primary", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    risk_proba = model.predict_proba(input_df)[0][1]
    risk_percent = risk_proba * 100
    
    st.markdown("---")
    st.header(t["result_header"])
    
    if risk_percent < 30:
        risk_level = "Low"
        st.success(f"**{t['low_risk']} {risk_percent:.1f}%**")
        st.markdown(t["low_desc"])
        st.markdown("---")
        with st.expander(t["health_tips"], expanded=True):
            st.markdown(t["tips_desc"])
            st.markdown(t["low_tips_title"])
            st.markdown(t["low_tips"])
            st.caption(t["note"])
    elif risk_percent < 70:
        risk_level = "Moderate"
        st.warning(f"**{t['mod_risk']} {risk_percent:.1f}%**")
        st.markdown(t["mod_desc"])
        st.markdown("---")
        with st.expander(t["health_tips"], expanded=True):
            st.markdown(t["tips_desc"])
            st.markdown(t["mod_tips_title"])
            st.markdown(t["mod_tips"])
            st.caption(t["note"])
    else:
        risk_level = "High"
        st.error(f"**{t['high_risk']} {risk_percent:.1f}%**")
        st.markdown(t["high_desc"])
        st.markdown("---")
        with st.expander(t["health_tips"], expanded=True):
            st.markdown(t["tips_desc"])
            st.markdown(t["high_tips_title"])
            st.markdown(t["high_tips"])
            st.caption(t["note"])
    
    st.markdown("---")
    
    # SHAP Explainability
    st.subheader(t["how_calc"])
    st.caption(t["chart_caption"])
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Family History', 'Age']
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in shap_values[0]]
    
    sns.barplot(x=shap_values[0], y=feature_labels, palette=colors, ax=ax)
    ax.set_title(t["chart_title"], fontsize=14, fontweight='bold')
    ax.set_xlabel(t["chart_xlabel"])
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.caption(t["red_bars"])

# --- Footer ---
st.divider()
st.caption(t["footer_disc"])
st.caption(t["footer_built"])
st.caption(t["limitations"])
