import streamlit as st
import pandas as pd
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import os

DB_FILE = "users_data.csv"

if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=["username", "password", "xp"])
    df.to_csv(DB_FILE, index=False)

def get_data():
    return pd.read_csv(DB_FILE)

def update_data(df):
    df.to_csv(DB_FILE, index=False)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'xp' not in st.session_state:
    st.session_state.xp = 0

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

def process_sign(label):
    instructions = {
        "SPEED_LIMIT_5": "Խիստ դանդաղ! Արագությունը մինչև 5 կմ/ժ (Max 5 km/h)",
        "SPEED_LIMIT_15": "Դանդաղեցրեք ընթացքը՝ մինչև 15 կմ/ժ (Max 15 km/h)",
        "SPEED_LIMIT_20": "Դանդաղեցրեք ընթացքը՝ մինչև 20 կմ/ժ (Max 20 km/h)",
        "SPEED_LIMIT_30": "Արագությունը՝ մինչև 30 կմ/ժ (Max 30 km/h)",
        "SPEED_LIMIT_40": "Արագությունը՝ մինչև 40 կմ/ժ (Max 40 km/h)",
        "SPEED_LIMIT_50": "Արագությունը՝ մինչև 50 կմ/ժ (Max 50 km/h)",
        "SPEED_LIMIT_60": "Քաղաքային արագություն՝ մինչև 60 կմ/ժ (Max 60 km/h)",
        "SPEED_LIMIT_70": "Արագությունը՝ մինչև 70 կմ/ժ (Max 70 km/h)",
        "SPEED_LIMIT_80": "Արագությունը՝ մինչև 80 կմ/ժ (Max 80 km/h)",
        "RESTRICTION_ENDS": "Սահմանափակումների ավարտ (Restrictions End)",

        "STOP": "ԿԱՆԳ! Կանգնեք 3 վայրկյան (Full Stop for 3 sec)",
        "GIVE_WAY": "Զիջեք ճանապարհը բոլորին (Give Way / Yield)",
        "NO_ENTRY": "ՄՈՒՏՔՆ ԱՐԳԵԼՎԱԾ Է! (No Entry)",
        "NO_PARKING": "Կայանումն արգելված է (No Parking)",
        "NO_STOPPING_OR_STANDING": "Կանգառն արգելված է! (No Stopping)",
        "STRAIGHT_PROHIBITED": "Երթևեկությունն ուղիղ արգելված է (Straight Prohibited)",
        "LEFT_TURN_PROHIBITED": "Ձախ շրջադարձն արգելված է (No Left Turn)",
        "RIGHT_TURN_PROHIBITED": "Աջ շրջադարձն արգելված է (No Right Turn)",
        "U_TURN_PROHIBITED": "Հետադարձն արգելված է (No U-Turn)",
        "OVERTAKING_PROHIBITED": "Վազանցն արգելված է (Overtaking Prohibited)",
        "HORN_PROHIBITED": "Ձայնային ազդանշանն արգելված է (No Honking)",
        "ALL_MOTOR_VEHICLE_PROHIBITED": "Մեքենաների մուտքն արգելված է (All Motor Vehicles Prohibited)",
        "TRUCK_PROHIBITED": "Բեռնատարների մուտքն արգելված է (Trucks Prohibited)",
        "CYCLE_PROHIBITED": "Հեծանիվների մուտքն արգելված է (Cycles Prohibited)",
        "PEDESTRIAN_PROHIBITED": "Հետիոտների մուտքն արգելված է (Pedestrians Prohibited)",
        "BULLOCK_PROHIBITED": "Լծասայլերի մուտքն արգելված է (Animal-drawn vehicles Prohibited)",
        "BULLOCK_AND_HANDCART_PROHIBITED": "Սայլակների մուտքն արգելված է (Carts Prohibited)",
        "HANDCART_PROHIBITED": "Ձեռնասայլակների մուտքն արգելված է (Handcarts Prohibited)",
        "TONGA_PROHIBITED": "Կառքերի մուտքն արգելված է (Tonga Prohibited)",

        "COMPULSARY_AHEAD": "Պարտադիր ուղիղ (Must go Straight)",
        "TURN_RIGHT": "Պարտադիր աջ (Turn Right only)",
        "COMPULSARY_TURN_LEFT": "Պարտադիր ձախ (Turn Left only)",
        "COMPULSARY_TURN_RIGHT_AHEAD": "Շրջադարձը միայն աջ (Turn Right Ahead)",
        "COMPULSARY_TURN_LEFT_AHEAD": "Շրջադարձը միայն ձախ (Turn Left Ahead)",
        "COMPULSARY_AHEAD_OR_TURN_RIGHT": "Ուղիղ կամ աջ (Straight or Right)",
        "COMPULSARY_AHEAD_OR_TURN_LEFT": "Ուղիղ կամ ձախ (Straight or Left)",
        "COMPULSARY_KEEP_RIGHT": "Պահպանեք աջ կողմը (Keep Right)",
        "COMPULSARY_KEEP_LEFT": "Պահպանեք ձախ կողմը (Keep Left)",
        "PASS_EITHER_SIDE": "Անցեք երկու կողմից էլ (Pass either side)",
        "ROUNDABOUT": "Շրջանաձև երթևեկություն (Roundabout)",
        "COMPULSARY_SOUND_HORN": "Պարտադիր ձայնային ազդանշան (Sound Horn)",
        "COMPULSARY_CYCLE_TRACK": "Հեծանվային ուղի (Cycle Track)",
        "COMPULSARY_MINIMUM_SPEED": "Նվազագույն արագություն (Minimum Speed)",

        "PEDESTRIAN_CROSSING": "ԶԳՈՒՅՇ! Հետիոտնային անցում (Pedestrian Crossing)",
        "SCHOOL_AHEAD": "Դպրոց! Դանդաղեցրեք ընթացքը (School Ahead)",
        "CATTLE": "Անասուններ ճանապարհին (Cattle Ahead)",
        "TRAFFIC_SIGNAL": "Լուսացույց առջևում (Traffic Light Ahead)",
        "CROSS_ROAD": "Խաչմերուկ առջևում (Cross Road Ahead)",
        "T_INTERSECTION": "T-աձև խաչմերուկ (T-Intersection)",
        "Y_INTERSECTION": "Y-աձև խաչմերուկ (Y-Intersection)",
        "STAGGERED_INTERSECTION": "Շեղված խաչմերուկ (Staggered Intersection)",
        "SIDE_ROAD_RIGHT": "Միացող ճանապարհ աջից (Side Road Right)",
        "SIDE_ROAD_LEFT": "Միացող ճանապարհ ձախից (Side Road Left)",
        "GAP_IN_MEDIAN": "Բացվածք բաժանարար գոտում (Gap in Median)",
        "MAJOR_ROAD_AHEAD": "Գլխավոր ճանապարհի հատում (Major Road Ahead)",
        "NARROW_ROAD_AHEAD": "Ճանապարհի նեղացում (Narrow Road Ahead)",
        "NARROW_BRIDGE": "Նեղ կամուրջ (Narrow Bridge)",
        "SLIPPERY_ROAD": "Սահուն ճանապարհ (Slippery Road)",
        "LOOSE_GRAVEL": "Խճի արտանետում (Loose Gravel)",
        "CYCLE_CROSSING": "Հեծանվորդների հատում (Cycle Crossing)",
        "FALLING_ROCKS": "Քարաթափման վտանգ (Falling Rocks)",
        "DANGEROUS_DIP": "Վտանգավոր գոգավորություն (Dangerous Dip)",
        "HUMP_OR_ROUGH_ROAD": "Անհարթ ճանապարհ (Rough Road)",
        "STEEP_ASCENT": "Վտանգավոր վերելք (Steep Ascent)",
        "STEEP_DESCENT": "Վտանգավոր վայրէջք (Steep Descent)",
        "LEFT_HAND_CURVE": "Վտանգավոր շրջադարձ ձախ (Left Curve)",
        "RIGHT_HAND_CURVE": "Վտանգավոր շրջադարձ աջ (Right Curve)",
        "LEFT_HAIR_PIN_BEND": "Կտրուկ ձախ շրջադարձ (Left Hairpin Bend)",
        "RIGHT_HAIR_PIN_BEND": "Կտրուկ աջ շրջադարձ (Right Hairpin Bend)",
        "LEFT_REVERSE_BEND": "Ոլորաններ ձախ (Left Reverse Bend)",
        "RIGHT_REVERSE_BEND": "Ոլորաններ աջ (Right Reverse Bend)",
        "MEN_AT_WORK": "Ճանապարհային աշխատանքներ (Men at Work)",
        "QUAY_SIDE_OR_RIVER_BANK": "Ելք դեպի առափնյա (River Bank)",
        "FERRY": "Լաստանավային անցում (Ferry Ahead)",
        "BARRIER_AHEAD": "Ուղեփակոց առջևում (Barrier Ahead)",
        "GUARDED_LEVEL_CROSSING": "Երկաթուղային անցում՝ ուղեփակոցով (Guarded Level Crossing)",
        "UNGUARDED_LEVEL_CROSSING": "Երկաթուղային անցում՝ առանց ուղեփակոցի (Unguarded Level Crossing)",

        "LOAD_LIMIT": "Քաշի սահմանափակում (Load Limit)",
        "AXLE_LOAD_LIMIT": "Սռնու վրա ծանրաբեռնվածություն (Axle Load Limit)",
        "WIDTH_LIMIT": "Լայնության սահմանափակում (Width Limit)",
        "HEIGHT_LIMIT": "Բարձրության սահմանափակում (Height Limit)",
        "LENGTH_LIMIT": "Երկարության սահմանափակում (Length Limit)",
        "PRIORITY_FOR_ONCOMING_VEHICLES": "Զիջեք հանդիպակաց մեքենաներին (Priority for oncoming)",
        "ROAD_WIDENS_AHEAD": "Ճանապարհի լայնացում (Road Widens Ahead)",
        "DIRECTION": "Ուղղություն (Direction Sign)"
    }
    return instructions.get(label, f"Նշան: {label}")

def auth_page():
    st.title("🚦 Traffic Sign Game")
    choice = st.sidebar.radio("Գործողություն", ["Մուտք", "Գրանցում"])
    
    df = get_data()
    
    st.subheader(choice)
    user = st.text_input("Օգտանուն")
    pwd = st.text_input("Գաղտնաբառ", type="password")
    
    if st.button("Հաստատել"):
        if choice == "Մուտք":
            user_record = df[(df['username'] == user) & (df['password'].astype(str) == pwd)]
            if not user_record.empty:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.session_state.xp = int(user_record.iloc[0]['xp'])
                st.rerun()
            else:
                st.error("Սխալ օգտանուն կամ գաղտնաբառ")
        else:
            if user in df['username'].values:
                st.warning("Այս օգտանունը զբաղված է")
            else:
                new_row = pd.DataFrame([{"username": user, "password": pwd, "xp": 0}])
                updated_df = pd.concat([df, new_row], ignore_index=True)
                update_data(updated_df)
                st.success("Գրանցումը հաջողվեց! Մուտք գործեք:")

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.5)
        
        for r in results:
            for box in r.boxes:
                lbl = model.names[int(box.cls)]
                instr = process_sign(lbl)
                cv2.putText(img, instr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return results[0].plot()

def game_page():
    st.sidebar.title(f"👤 {st.session_state.username}")
    st.sidebar.header(f"🏆 XP: {st.session_state.xp}")
    
    if st.sidebar.button("Ելք"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🚗 Real-time Recognition")
    
    webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("✅ Ճիշտ կատարեցի (+20 XP)"):
        st.session_state.xp += 20
        save_progress()
    if col2.button("❌ Սխալվեցի (-10 XP)"):
        st.session_state.xp -= 10
        save_progress()

def save_progress():
    df = get_data()
    df.loc[df['username'] == st.session_state.username, 'xp'] = st.session_state.xp
    update_data(df)
    st.toast("Առաջընթացը պահպանվեց!")

if not st.session_state.logged_in:
    auth_page()
else:
    game_page()
