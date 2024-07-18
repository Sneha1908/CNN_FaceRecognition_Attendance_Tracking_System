import streamlit as st
import pandas as pd
from datetime import datetime
import os
import csv

def load_attendance(date):
    try:
        df = pd.read_csv(f"Attendance/Attendance_{date}.csv")
        return df
    except FileNotFoundError:
        st.error(f"No attendance records found for {date}.")
        return None

def parse_id_name(folder_name):
    parts = folder_name.split('-')
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def main():
    st.title("Attendance Record")
    ts = datetime.now()
    date = ts.strftime("%d-%m-%Y")
    df_attendance = load_attendance(date)

    if df_attendance is not None:
        st.write(f"Attendance for {date}")
        present_names = df_attendance['NAME'].unique()
        dataset_path = r'C:\Users\Sneha\Desktop\FACE RECOGNITION SYSTEM\data'

        for folder_name in os.listdir(dataset_path):
            id, name = parse_id_name(folder_name)
            if id is not None and name is not None and name not in present_names:
                pass

        st.dataframe(df_attendance.style.highlight_max(axis=0), width=None, height=None)
    else:
        st.write("No attendance records found for today.")

if __name__ == "__main__":
    main()

#Streamlit run attendance_app.py