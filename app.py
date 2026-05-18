import streamlit as st
import pandas as pd
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av  # Պետք է տեղադրել pip install av
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
if 'detected_sign' not in st.session_state:
    st.session_state.detected_sign = "Նշան դեռևս չի հայտնաբերվել"


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
        "NO_PARKING": "Կայանումն արգελված է (No Parking)",
        "NO_STOPPING_OR_STANDING": "Կանգառն արգելված է! (No Stopping)",
        "PEDESTRIAN_CROSSING": "ԶԳՈՒՅՇ! Հետիոտնային անցում (Pedestrian Crossing)",
        "SCHOOL_AHEAD": "Դպրոց! Դանդաղեցրեք ընթացքը (School Ahead)"
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


# Օպտիմալացված տեսահոլովակի մշակող դաս
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.5)

        # Գտնում ենք նշանները և թարմացնում տեքստը
        for r in results:
            for box in r.boxes:
                lbl = model.names[int(box.cls)]
                st.session_state.detected_sign = process_sign(lbl)

        # Վերադարձնում ենք YOLO-ի կողմից գծված կադրը
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def game_page():
    st.sidebar.title(f"👤 {st.session_state.username}")
    st.sidebar.header(f"🏆 XP: {st.session_state.xp}")

    if st.sidebar.button("Ելք"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🚗 Real-time Recognition")

    # Միացնում ենք WebRTC սթրիմը
    webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)

    # Ցուցադրում ենք հայերեն հրահանգը գեղեցիկ տեսքով
    st.info(f"💡 **Հրահանգ վարորդին:** {st.session_state.detected_sign}")

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
    st.rerun()


if not st.session_state.logged_in:
    auth_page()
else:
    game_page()