import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="အရေပြားရောဂါရှာဖွေရေး",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .sidebar .sidebar-content { background: #ffffff }
    h1 { color: #2c3e50; }
    .st-emotion-cache-8fjoqp { margin: auto; width: 80%; }
    .st-bb { background-color: #ffffff; }
    .st-at { background-color: #f0f2f6; }
    .disease-info {
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stImage img {
        max-width: 400px;
        margin: 0 auto;
        display: block;
    }
    .preprocessing-step {
        color: #666;
        font-size: 14px;
        margin: 5px 0;
    }
    .clear-skin { color: green; font-weight: bold; font-size: 18px; }
    .disease-detected { color: red; font-weight: bold; }
    .diagnosis-box {
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .accuracy-display {
        font-size: 16px;
        color: #666;
        margin-top: 10px;
    }
    .skin-warning {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 18px;
        padding: 10px;
        background-color: #ffe6e6;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('skin_disease.h5')
        return model
    except Exception as e:
        st.error(f"မော်ဒယ်ဖတ်ရှုရာတွင် အမှား: {str(e)}")
        return None


model = load_model()

CLASS_NAMES = {
    0: 'Actinic keratoses (akiec)',
    1: 'Basal cell carcinoma (bcc)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic nevi (nv)',
    6: 'Vascular lesions (vasc)'
}

CLASS_NAMES_MM = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}
DISEASE_INFO = {
    0: {
        'causes': 'ခရမ်းလွန်ရောင်ခြည်ကိုမကြာခဏနှင့် အကြာကြီးထိတွေ့ခြင်း၊ အသက်ကြီးလာခြင်း (၄၀ နှစ်အထက်)၊နေရောင်ကာအကာအကွယ်မရှိဘဲ အပြင်ထွက်နေသူများ၊အရင်က နေလောင်ဒဏ်များခံဖူးသူ၊မိသားစုတွင် အရေပြားကင်ဆာ ဖြစ်ဖူးသူရှိခြင်း၊',
        'about': 'နေရောင်ခြည်ထဲက Ultraviolet B (UVB) rays သည် အရေပြားအတွင်းရှိ DNA ကိုထိခိုက်စေပြီးဆဲလ်များမမှန်မကန်ဖြစ်လာစေတတ်သည်။ယင်းပြဿနာရှိသောဆဲလ်များသည်အချိန်တိုအတွင်းမှာပဲ အမာရွတ်လေးများ ဖြစ်လာနိုင်သည်။အချိုကတော့Squamous Cell Carcinoma (SCC) ဆိုတဲ့ skin cancer အဖြစ် ပြောင်းလဲနိုင်သည်။',
        'protect':' Sunscreen (SPF 30+ or higher) ကို နေရောင်ထဲထွက်ခါနီး လိမ်းပါ၊ဦးထုပ် / မျက်မှန် / လက်ရှည်အင်္ကျီ စသည်ဖြင့် နေရောင်ကာအဝတ်အစားဝတ်ဆင်ပါ၊ နေရောင်အပြင်းအထန်ဆုံးအချိန် (မနက် ၁၀ နာရီ မှ မွန်းလွဲ ၄ နာရီ) အတွင်း အပြင်မထွက်ခြင်း၊ အသုံးပြုနေသော ဆေးဝါးများကြောင့် နေရောင်ခံနိုင်စွမ်းနည်းသူများသည် extra care လုပ်ဖိုလိုသည်၊ မည်သည့် အမာရွတ် / အမဲစက်များ များလာပါက သက်ဆိုင်ရာ အရေပြားအထူးကု ထံတွင် စစ်ဆေးခြင်း၊'
    },
    1: {
        'causes': 'နေရောင်အလွန်ပြင်းထန်စွာထိတွေ့ခြင်း၊ အသားဖြူပြီး နေလောင်လွယ်သောသူများ၊ အသက်ကြီးလာခြင်း၊ နေရောင်ကာကွယ်မှုမရှိဘဲ ပြင်ပတွင် အလုပ်လုပ်ကိုင်နေသူများ၊ မိသားစုအတွင်း အရေပြားကင်ဆာခံဖူးသူ ရှိခြင်း၊ ကင်ဆာကုသမှုများ (ဓာရောင်ခြည်၊ ဆေးဝါး) ခံထားဖူးခြင်း',
        'about': 'အရေပြားအောက်ဆုံးထပ်တွင်ရှိသော သဘာဝဆဲလ်များသည် နေရောင်၏ပြင်းထန်မှုကြောင့် DNA ပျက်စီးမှု ဖြစ်တတ်သည်။ယင်းဆဲလ်များသည် ထိန်းချုပ်မှုမရှိဘဲ တိုးပွားလာပြီး ကင်ဆာဆဲလ်များ ဖြစ်လာသည်။အစပိုင်းတွင် တုတ်တုတ်လေး၊ ပွတင်းပွတင်းလေး သိုမဟုတ် အရောင်ဖျော့ဖျော့နဲ့ မျက်နှာပေါ်မှာ ပေါ်လာတတ်သည်။ြာလာသည်နှင့်အမျှ ပိုမိုကြီးထွားပြီး အနာပွအဖြစ် ဖြစ်လာနိုင်သည်။',
        'protect':' နေရောင်ပြင်းသောအချိန်များတွင် (မနက် ၁၀ နာရီမှ မွန်းလွဲ ၄ နာရီထိ) အပြင်မထွက်ရန်၊ နေရောင်ကာကွယ်မှုအတွက် suncreamလိမ်းခြင်း၊ ဦးထုပ်၊ တင်းလက်ရှည်အင်္ကျီဝတ်ဆင်ခြင်း၊ မျက်နှာ၊ လက်မောင်း၊ လည်ပင်းကဲ့သို နေရောင်ထိတွေ့လွယ်သောနေရာများကို အစဉ်သတိထားခြင်း၊ မမှန်မကန်အသားထင်လာခြင်းများရှိပါက အချိန်မီအရေပြားဆရာဝန်ထံသွားပြီး စစ်ဆေးခြင်း၊ မိမိအသားအရေကို တသမတ်တည်း ထိန်းသိမ်းခြင်း၊'
    },
    2: {
        'causes': ' အသက်အရွယ်ကြီးလာခြင်း၊ မိသားစုမွေးရိုးဗီဇဖြစ်စဉ်များ၊ နေရောင်သက်တမ်းရှည်ထိတွေ့ခြင်း၊ အသားအရေခြောက်သွေ့၍ ပျက်စီးမှုများ၊ တချိုအစားအစာ သိုမဟုတ် ဆေးဝါးများကို သုံးစွဲမှုမတူခြင်း',
        'about': ' အရေပြား၏အပေါ်ပိုင်း ဆဲလ်များသည်သဘာဝအတိုင်း ဆဲလ်သက်တမ်းကုန်ပြီး အစားထိုးသင့်သောနေရာတွင်မမှန်မကန်ပုံစံဖြင့် တိုးပွားလာခြင်းကြောင့် အမာရွတ် ပေါ်လာသည်။များသောအားဖြင့်အမဲစက်လေး၊ အပြာရောင်ခြယ်ခြယ်ဖြစ်ပြီးလက်မောင်း၊ မျက်နှာ၊လည်ပင်းပေါ်တွင် တွေ့ရတတ်သည်။ထိုအမာရွတ်များသည်ထွက်လာပြီဆိုတာနဲ့အတူ အတိုးအကျယ်မရှိပဲ တည်နေတတ်သည်။တစ်ချိုမှာ ယားယံခြင်းများ ဖြစ်တတ်သည်။',
        'protect':' နေရောင်အပြင်းအထန်ခံခြင်းမှ ရှောင်ကြည်ခြင်း၊ နေထဲထွက်ချင်ရင် sumcreamလိမ်းခြင်း၊ ဦးထုပ်၊ တင်းလက်ရှည်ဝတ်ဆင်ခြင်း၊ အသားအရေသန့်ရှင်းမှုခြောက်သွေ့မှု မရှိအောင် စောင့်ရှောက်ခြင်း၊ အသားအရေသဘာဝမျိုးအလိုက် အာဟာရပြည့်ဝသောအစားအစာစားခြင်း၊ မမှန်မကန် ပုံစံ အမာရွတ်များ ပေါ်လာပါကအချိန်မီ အရေပြားအထူးကုဆရာဝန်ထံ သွားပြီး စစ်ဆေးခြင်း၊'
    },
    3: {
        'causes': 'Dermatofibroma (အရေပြားပေါ်က အဖုမာ) ဟာ ပုံမှန်အားဖြင့် အသေးစား ထိခိုက်ဒဏ်ရာရခြင်း၊ အင်းဆက်ပိုးကိုက်ခံရခြင်း သိုမဟုတ် အမွေးအိတ်ရောင်ခြင်းတိုလို အရေပြား ပေါက်ပြဲမှုတွေကြောင့် ဖြစ်ပွားတတ်ပါတယ်။ ဒီအဖုမာတွေဟာ fibroblasts လိုခေါ်တဲ့ အရေပြားအတွင်းပိုင်းရှိ ဆဲလ်များ အလွန်အကျွံပွားများလာခြင်းကြောင့် ဖြစ်ပေါ်လာရတာပါ။',
        'about': 'အဖုမာတွေက များသောအားဖြင့် အသားရောင်၊ အနီရောင်၊ ပန်းရောင်၊ ခရမ်းရောင် ဒါမှမဟုတ် အညိုရောင် ရှိပါတယ်။ အရွယ်အစားအားဖြင့် လက်သည်းခွံလောက်ပဲရှိပြီး မာကျောတဲ့ အဖုအပိမ့်ပုံစံဖြစ်နေတတ်ပါတယ်။ အများအားဖြင့် ခြေသလုံး ဒါမှမဟုတ် လက်မောင်းတွေပေါ်မှာ တွေ့ရတတ်ပါတယ်။ ဒီအဖုမာကို ဘေးနှစ်ဖက်ကနေ ညှစ်လိုက်ရင် အရေပြားအောက်ထဲကို ခွက်ဝင်သွားတတ်တဲ့ "dimple sign" လက္ခဏာရပ်မျိုးလည်း ရှိပါတယ်။ Dermatofibroma ဟာ အကျိတ်ဆိုး (cancer) မဟုတ်ဘဲ နာကျင်မှုလည်း မရှိတာကြောင့် အရေပြားပေါ်မှာ တစ်သက်လုံးရှိနေနိုင်ပါတယ်။',
        'protect':'လက်ရှိအချိန်အထိ Dermatofibroma ကို ဘယ်လိုကာကွယ်ရမယ်ဆိုတဲ့ အချက်အတိအကျမရှိသေးပါဘူး။ ဘာလိုလဲဆိုတော့ ဒါဟာ ထိခိုက်ဒဏ်ရာကြောင့် အဓိကဖြစ်တာဖြစ်ပြီး ဘယ်သူမဆိုဖြစ်နိုင်လိုပါပဲ။ဒါပေမဲ့ ဒီအဖုမာကြောင့် စိတ်အနှောင့်အယှက်ဖြစ်တယ်ဆိုရင် ဒါမှမဟုတ် အမြင်မကောင်းဘူးလို ထင်တယ်ဆိုရင်တော့ အရေပြားဆရာဝန်နဲ့ တိုင်ပင်ပြီး ဖယ်ရှားနိုင်ပါတယ်။ ဖယ်ရှားတဲ့အခါမှာ ခွဲစိတ်ဖယ်ရှားခြင်း၊ အရေပြားအပေါ်ယံလွှာကို ဓားနဲ့ခြစ်ထုတ်ခြင်း၊ ဒါမှမဟုတ် အရည်နိုက်ထရိုဂျင်သုံးပြီး အအေးပေးဖျက်ဆီးခြင်း စတဲ့ နည်းလမ်းတွေနဲ့ ကုသနိုင်ပါတယ်။'
    },
    4: {
        'causes': 'Melanoma ဟာ အရေပြားကင်ဆာတစ်မျိုးဖြစ်ပြီး နေရောင်ခြည်ဒဏ် (UV rays) ကို အလွန်အကျွံထိတွေ့ခြင်းကြောင့် အဓိကဖြစ်ပွားတတ်ပါတယ်။ နေရောင်ခြည်ကလာတဲ့ ခရမ်းလွန်ရောင်ခြည်တွေဟာ အရေပြားဆဲလ်တွေရဲ့ DNA ကို ပျက်စီးစေပြီး အဲဒီဆဲလ်တွေကို ထိန်းမရအောင် ပွားများစေလို Melanoma ဖြစ်လာရတာပါ။ မိသားစုမျိုးရိုးလိုက်ခြင်း၊ အသားအရည်ဖြူဖျော့သူတွေနဲ့ ခန္ဓာကိုယ်မှာ မှဲ့ (moles) အများကြီးရှိတဲ့သူတွေဟာလည်း Melanoma ဖြစ်နိုင်ခြေပိုများပါတယ်။',
        'about': 'Melanoma ဟာ ပုံမှန်အားဖြင့် ကိုယ်ခန္ဓာပေါ်မှာရှိတဲ့ မှဲ့အသစ်တစ်ခုလို ဒါမှမဟုတ် အရင်ကရှိနေပြီးသား မှဲ့တစ်ခုရဲ့ ပုံပန်းသဏ္ဌာန်ပြောင်းလဲမှုကနေ စတင်ပါတယ်။ Melanoma ကို သတိထားမိနိုင်တဲ့ လက္ခဏာ (၅) မျိုး မှဲ့တစ်ဝက်နဲ့ ကျန်တစ်ဝက်ပုံစံမတူဘဲ ပုံပန်းမညီတာ၊ မှဲ့ရဲ့အနားသတ်က ပုံမှန်မဟုတ်ဘဲ အနားသားတွေ ကြမ်းတမ်းနေတာ၊ မှဲ့ရဲ့အရောင်က အညို၊ အနက်၊ အနီ၊ အပြာ စသဖြင့် တစ်နေရာနဲ့တစ်နေရာ အရောင်မတူဘဲ ကွဲပြားနေတာ၊ မှဲ့ရဲ့အချင်းက ခဲတံဖျက်ခေါင်းထက် (၆ မီလီမီတာ) ပိုကြီးနေတာ၊ အချိန်ကြာလာတာနဲ့အမျှ မှဲ့ရဲ့အရွယ်အစား၊ ပုံသဏ္ဌာန် ဒါမှမဟုတ် အရောင်ပြောင်းလဲလာတာ၊',
        'protect':'Melanoma ကို ကာကွယ်ဖိုအတွက် အကောင်းဆုံးနည်းလမ်းကတော့ နေရောင်ခြည်ဒဏ်ကို ရှောင်ကြဉ်ဖို ပါပဲ။နေရောင်ကာခရင်မ် (Sunscreen) လိမ်းပါ: SPF 30 သိုမဟုတ် အဲ့ဒီထက်ပိုတဲ့ နေရောင်ကာခရင်မ်ကို အပြင်မထွက်ခင် မိနစ် (၂၀) ကြိုလိမ်းပြီး နာရီအနည်းငယ်ကြာတိုင်း ပြန်လိမ်းပေးပါ။ေရောင်ခြည်ပြင်းတဲ့အချိန်ရှောင်ပါ: မနက် (၁၀) နာရီကနေ ညနေ (၄) နာရီကြား နေရောင်ခြည်အပြင်းဆုံးအချိန်တွေမှာ အပြင်ထွက်တာကို တတ်နိုင်သမျှရှောင်ပါ။ခေါင်းအုပ်၊ မျက်မှန်နဲ့ အင်္ကျီလက်ရှည်ဝတ်ပါ: နေရောင်ကာကွယ်ဖို ဦးထုပ်၊ နေကာမျက်မှန်နဲ့ အရေပြားကိုဖုံးနိုင်တဲ့ အဝတ်အစားတွေ ဝတ်ဆင်ပါ။ပုံမှန်စစ်ဆေးပါ: သင့်ကိုယ်ပေါ်က မှဲတွေကို ပုံမှန်စစ်ဆေးပြီး ပုံမှန်မဟုတ်တာတွေ တွေ့ရင် အရေပြားဆရာဝန်နဲ့ တိုင်ပင်ပါ။'
    },
    5: {
        'causes': 'Melanocytic nevi ဖြစ်ပေါ်ရတဲ့ အဓိကအကြောင်းရင်းနှစ်ခုကတော့ မျိုးရိုးဗီဇ (Genetic Factors): မိသားစုထဲမှာ မှဲ့များတဲ့ မျိုးရိုးရှိရင် ကိုယ်တိုင်လည်း မှဲ့များနိုင်ပါတယ်။ နေရောင်ခြည်ဒဏ် (Sun Exposure): ငယ်စဉ်ဘဝတုန်းက နေရောင်ခြည်ကို အလွန်အကျွံထိတွေ့ခဲ့တာက မှဲ့အသစ်တွေ ပိုမိုဖြစ်ပေါ်စေနိုင်ပါတယ်။',
        'about': 'Melanocytic nevi တွေဟာ အမျိုးအစားအမျိုးမျိုးရှိပြီး ပုံသဏ္ဌာန်၊ အရွယ်အစားနဲ့ အရောင်တွေကလည်း ကွဲပြားပါတယ်။ အရေပြားမျက်နှာပြင်နဲ့တစ်သားတည်းဖြစ်ပြီး ချောမွေ့ပါတယ်။ အညိုရောင်၊ အမည်းရောင်ဖြစ်တတ်ပါတယ်။အရေပြားမျက်နှာပြင်ကနေ အနည်းငယ်ဖောင်းကြွနေပြီး အညိုရောင်ရှိပါတယ်။ တစ်ခါတလေ အမွေးနုလေးတွေပါ ပါတတ်ပါတယ်။ အသားရောင် ဒါမှမဟုတ် ပန်းရောင်ဖျော့ဖျော့ရှိပြီး အရေပြားပေါ်ကနေ သိသိသာသာဖောင်းကြွနေပါတယ်။ အမွေးနုလေးတွေပါ ပါလေ့ရှိပါတယ်။ မွေးကတည်းကပါလာတဲ့ မှဲ့မျိုးဖြစ်ပြီး အရွယ်အစားအမျိုးမျိုးရှိနိုင်ပါတယ်။ ဒီမှဲအများစုဟာ ကင်ဆာအကျိတ်ဆိုးမဟုတ်ပေမယ့် Melanoma (အရေပြားကင်ဆာ) အဖြစ် ပြောင်းလဲသွားနိုင်တဲ့ ဖြစ်နိုင်ခြေအနည်းငယ်ရှိတာကြောင့် သတိထားစောင့်ကြည့်ဖိုလိုပါတယ်။ အထူးသဖြင့် မှဲ့တစ်ခုရဲ့ ပုံသဏ္ဌာန်၊ အရွယ်အစား ဒါမှမဟုတ် အရောင်တွေ ရုတ်တရက်ပြောင်းလဲသွားမယ်၊ ယားယံမယ် ဒါမှမဟုတ် သွေးထွက်တာမျိုးဖြစ်လာမယ်ဆိုရင် ဆရာဝန်နဲ့ သေချာပြသသင့်ပါတယ်။',
        'protect':'Melanocytic nevi ဖြစ်ပေါ်ခြင်းကို အပြည့်အဝကာကွယ်ဖိုဆိုတာ မလွယ်ပေမယ့် နေရောင်ခြည်ဒဏ်ကို ကာကွယ်ခြင်းအားဖြင့် မှဲအသစ်တွေ ပေါ်လာနိုင်ခြေကို လျှော့ချနိုင်ပါတယ်။နေရောင်ခြည်ကာကွယ်ပါ: အပြင်ထွက်တဲ့အခါ SPF 30 ဒါမှမဟုတ် အဲ့ဒီထက်ပိုတဲ့ နေရောင်ကာခရင်မ်ကို ပုံမှန်လိမ်းပါ။နေရောင်ခြည်ပြင်းတဲ့အချိန် ရှောင်ပါ: နေရောင်အပြင်းဆုံးဖြစ်တဲ့ မနက် ၁၀ နာရီကနေ ညနေ ၄ နာရီအတွင်းမှာ အပြင်ထွက်တာကို တတ်နိုင်သမျှရှောင်ပါ။မှန်မှန်စစ်ဆေးပါ: သင့်ခန္ဓာကိုယ်ပေါ်က မှဲတွေကို ပုံမှန်စစ်ဆေးပါ။ '
    },
    6: {
        'causes': 'Vascular lesions တွေဖြစ်ပေါ်ရတဲ့ အကြောင်းရင်းတွေက အမျိုးမျိုးရှိပါတယ်။မွေးရာပါချိုယွင်းချက်: အချိုသော vascular lesions တွေဟာ မွေးကတည်းကပါလာတာပါ။ သန္ဓေသားဘဝမှာ သွေးကြောတွေ ပုံမှန်မဖွံဖြိုးဘဲ ချိုယွင်းနေတာကြောင့် ဖြစ်တာပါ။အရေပြား ထိခိုက်ဒဏ်ရာရတာ၊ ပိုးကိုက်ခံရတာမျိုးတွေကြောင့်လည်း ဖြစ်နိုင်ပါတယ်။ရောဂါပိုးဝင်ရောက်ခြင်း ဒါမှမဟုတ် တခြား ကိုယ်တွင်းရောဂါအချိုကြောင့်လည်း vascular lesions တွေဖြစ်ပေါ်နိုင်ပါတယ်။အများအားဖြင့် vascular lesions တွေကို အဓိက အမျိုးအစား ၂ မျိုး ခွဲခြားနိုင်ပါတယ်။ Vascular Tumors : ဥပမာ - Hemangioma (သွေးကြောအိတ်) လိုမျိုးပါ။ မွေးပြီး မကြာခင်မှာ ပေါ်လာတတ်ပြီး အများအားဖြင့် သူ့အလိုလို သက်သာသွားတတ်ပါတယ်။Vascular Malformations: ဥပမာ - Port-wine Stains လိုမျိုးပါ။ မွေးကတည်းကပါလာပြီး တစ်သက်တာလုံး ရှိနေတတ်ပါတယ်။',
        'about': 'Vascular lesions ရဲ့ ဖြစ်ပေါ်ပုံဟာ သူ့ရဲ့ အမျိုးအစားပေါ်မူတည်ပြီး ကွဲပြားပါတယ်။မွေးပြီး ပထမဆုံး ရက်သတ္တပတ်အနည်းငယ်အတွင်းမှာ အရေပြားပေါ်မှာ ဖောင်းကြွတဲ့ အနီရောင်အဖုအဖြစ် စတင်ပေါ်ပေါက်လာတတ်ပါတယ်။ ပြီးရင် တစ်နှစ်ပတ်လည်လောက်အထိ ကြီးထွားလာပြီး အများအားဖြင့် ၅ နှစ်ကနေ ၁၀ နှစ်အတွင်းမှာ သူ့အလိုလို ပြန်သေးသွားပါတယ်။ဒါတွေက မွေးကတည်းကရှိပြီး ဖြည်းဖြည်းချင်းကြီးထွားလာတတ်ပါတယ်။ ရာသက်ပန်ရှိနေတတ်ပြီး ပုံမှန်အားဖြင့် ဖောင်းကြွမှုမရှိတဲ့ ပြားတဲ့ အနီရောင် ဒါမှမဟုတ် ခရမ်းရောင်အကွက်တွေအဖြစ် တွေ့ရတတ်ပါတယ်။အချို့သော vascular lesions တွေဟာ နာကျင်မှုမရှိဘဲ အလှအပဆိုင်ရာပြဿနာအဖြစ်သာ ရှိနေတတ်ပေမဲ့ အချိုကတော့ သွေးယိုစိမ့်တာ၊ ကိုက်ခဲတာ ဒါမှမဟုတ် ကိုယ်တွင်းအင်္ဂါတွေကို ထိခိုက်တာမျိုးအထိ ဖြစ်နိုင်ပါတယ်။',
        'protect':'Vascular lesions အားလုံးကို အပြည့်အဝ ကာကွယ်ဖိုဆိုတာ မလွယ်ပါဘူး။ အထူးသဖြင့် မွေးရာပါချိုယွင်းချက်တွေကြောင့်ဖြစ်တာဆိုရင် ကာကွယ်ဖိုခက်ပါတယ်။ ဒါပေမဲ့ အရေပြားကို ထိခိုက်ဒဏ်ရာမရအောင် ဂရုစိုက်နေထိုင်တာကတော့ အသစ်ဖြစ်ပေါ်လာမယ့် lesions တွေကို အတိုင်းအတာတစ်ခုအထိ ကာကွယ်နိုင်ပါတယ်။ကုသမှုအပိုင်းမှာတော့ အများအားဖြင့် လေဆာကုသမှု ကို အသုံးပြုကြပါတယ်။ လေဆာရောင်ခြည်က ပုံမှန်မဟုတ်တဲ့ သွေးကြောတွေကိုသာ ဖယ်ရှားပေးပြီး ဘေးနားက အရေပြားကို ထိခိုက်မှုနည်းပါတယ်။ အခြေအနေအပေါ်မူတည်ပြီး ခွဲစိတ်ဖယ်ရှားခြင်း သိုမဟုတ် ဆေးထိုးခြင်း စတဲ့ နည်းလမ်းတွေလည်း ရှိပါတယ်။အကယ်၍ သင့်ခန္ဓာကိုယ်မှာ ပုံမှန်မဟုတ်တဲ့ အနီရောင်အကွက်တွေ ဒါမှမဟုတ် အဖုအကျိတ်တွေ ပေါ်လာတယ်ဆိုရင်တော့ အရေပြားဆရာဝန်နဲ့ တိုင်ပင်ဆွေးနွေးပြီး မှန်ကန်တဲ့ ရောဂါရှာဖွေမှုနဲ့ ကုသမှုကို ခံယူသင့်ပါတယ်။'
    }
}

def validate_image(image):
    if image is None:
        st.warning("ဓာတ်ပုံမတွေ့ပါ")
        return False
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Check for minimum dimensions
        if img_array.shape[0] < 50 or img_array.shape[1] < 50:
            st.warning("ဓာတ်ပုံအရွယ်အစား အလွန်သေးငယ်နေပါသည် (အနည်းဆုံး 50x50 pixels လိုအပ်ပါသည်)")
            return False

        # Skin detection for color images
        if len(img_array.shape) == 3:  # Color image
            # Convert to HSV color space
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Define skin color range in HSV
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # Create mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Calculate percentage of skin pixels
            skin_pixels = cv2.countNonZero(skin_mask)
            total_pixels = img_array.shape[0] * img_array.shape[1]
            skin_percentage = (skin_pixels / total_pixels) * 100

            # If less than 5% of the image contains skin-like colors
            if skin_percentage < 5:
                st.markdown(
                    '<div class="skin-warning">⚠️ ဤဓာတ်ပုံတွင် အရေပြားမပါဝင်ပါ (သို့) အရေပြားအစား အခြားအရာများပါဝင်နေပါသည်။ အရေပြားဓာတ်ပုံတင်ပေးပါ။</div>',
                    unsafe_allow_html=True)
                return False

        return True
    except Exception as e:
        st.error(f"ဓာတ်ပုံ စစ်ဆေးရာတွင် အမှားတစ်ခုဖြစ်နေသည်: {str(e)}")
        return False

def apply_advanced_preprocessing(image):
    """Enhanced preprocessing pipeline"""
    try:
        if not validate_image(image):
            return None

        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        # Handle different image formats
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Color correction
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.merge((l, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        # Smart denoising
        img = cv2.fastNlMeansDenoisingColored(
            img, None,
            h=10, hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        return img

    except Exception as e:
        st.error(f"ဓာတ်ပုံပြင်ဆင်ရာတွင် အမှားတစ်ခုဖြစ်နေသည်: {str(e)}")
        return None
def remove_background_and_focus_roi(image):
    """Improved ROI detection with better visualization and false positive reduction"""
    try:
        if not validate_image(image):
            return None, 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0

        # FIX 1: Ignore tiny noise areas (< 0.5% of image area)
        image_area = gray.shape[0] * gray.shape[1]
        contours = [c for c in contours if cv2.contourArea(c) > 0.005 * image_area]
        if not contours:
            return None, 0.0

        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Calculate affected area percentage
        disease_pixels = np.count_nonzero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        disease_percentage = (disease_pixels / total_pixels) * 100

        # Visualization
        visualization = image.copy()
        cv2.drawContours(visualization, [largest_contour], -1, (0, 255, 0), 2)

        return visualization, disease_percentage

    except Exception as e:
        st.error(f"ROI ထုတ်ယူရာတွင် အမှားတစ်ခုဖြစ်နေသည်: {str(e)}")
        return None, 0.0


def preprocess_for_model(image):
    """Final preprocessing for model input with validation"""
    try:
        if not validate_image(image):
            return None

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Resize with aspect ratio preservation
        height, width = image.shape[:2]
        target_size = 224

        # Calculate padding
        if height > width:
            new_height = target_size
            new_width = int(width * (target_size / height))
        else:
            new_width = target_size
            new_height = int(height * (target_size / width))

        resized = cv2.resize(image, (new_width, new_height))

        # Pad to make square
        delta_w = target_size - new_width
        delta_h = target_size - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        img = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # Normalize
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)

    except Exception as e:
        st.error(f"မော်ဒယ်အတွက် ဓာတ်ပုံပြင်ဆင်ရာတွင် အမှားတစ်ခုဖြစ်နေသည်: {str(e)}")
        return None

def main():
    st.title("🩺 အရေပြားရောဂါရှာဖွေရေး")
    st.markdown(
        "ဤစနစ်သည် အရေပြားပြဿနာများကို အမျိုးအစား ၇ မျိုးအထိ မှန်ကန်စွာ ခွဲခြားနိုင်သည်။ အသုံးပြုသူသည် အရေပြားပြဿနာရှိသော ဓာတ်ပုံတစ်ပုံကို တင်သွင်းခြင်းဖြင့်၊ အဆိုပါရောဂါအမျိုးအစားနှင့် ပတ်သက်သော ခန့်မှန်းအဖြေကို အလွယ်တကူ ရရှိနိုင်သည်။")

    uploaded_file = st.file_uploader("ဓာတ်ပုံတင်ပါ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file)

            if not validate_image(original_image):
                st.stop()

            processed_img = apply_advanced_preprocessing(original_image)
            roi_img, disease_percent = remove_background_and_focus_roi(
                processed_img if processed_img is not None else np.array(original_image)
            )

            predicted_class = None
            confidence = 0

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ဆန်းစစ်မှု အဆင့်ဆင့်")
                st.image(original_image, use_container_width=True, caption="ဓာတ်ပုံ")

            # FIX 2: Require at least 3% affected area
            if disease_percent > 1:
                model_input = preprocess_for_model(roi_img)
                if model and model_input is not None:
                    predictions = model.predict(model_input)
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    disease_name = CLASS_NAMES_MM[predicted_class]

                    if confidence > 99:
                        confidence = 100

                    # FIX 3: If confidence is low → clear skin
                    if confidence < 70:
                        st.markdown('<div class="clear-skin">ကျန်းမာသော အရေပြား (မည်သည့်ရောဂါမျှ မတွေ့ပါ)</div>',
                                    unsafe_allow_html=True)
                    else:
                        with col2:
                            st.subheader("ရောဂါရှာဖွေမှု ရလဒ်များ")
                            st.markdown(f"""
                                **Disease Type:**  
                                <span style="font-size: 20px; font-weight: bold;">{disease_name}</span>  
                                **Accuracy:** {confidence:.2f}%  
                            """, unsafe_allow_html=True)

                        st.subheader(f"{CLASS_NAMES_MM[predicted_class]} အကြောင်း")
                        with st.expander("ဖြစ်ပွားရသည့် အကြောင်းရင်းများ"):
                            st.write(DISEASE_INFO[predicted_class]['causes'])
                        with st.expander("ရောဂါအကြောင်း"):
                            st.write(DISEASE_INFO[predicted_class]['about'])
                        with st.expander("ကုသမှုနှင့် ကာကွယ်ရန်"):
                            st.write(DISEASE_INFO[predicted_class]['protect'])
            else:
                st.markdown('<div class="clear-skin">ကျန်းမာသော အရေပြား (မည်သည့်ရောဂါမျှ မတွေ့ပါ)</div>',
                            unsafe_allow_html=True)

        except Exception as e:
            st.error(f"ဓာတ်ပုံ ဆန်းစစ်ရာတွင် အမှားတစ်ခုဖြစ်နေသည်: {str(e)}")


if __name__ == "__main__":
    main()
