import os

import influxdb_client
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import mySTAMP
import mySTUMP
from DataPoint import DataPoint
from TD_Join import TD_Join
import pandas as pd
from DataPoint import DataPoint

import stumpy

from influxdb_client import InfluxDBClient, WriteOptions
import pandas as pd
# Initialize session state variables
if 'data1' not in st.session_state:
    st.session_state.data1 = None
if 'data2' not in st.session_state:
    st.session_state.data2 = None
if 'value1' not in st.session_state:
    st.session_state.value1 = None
if 'value2' not in st.session_state:
    st.session_state.value2 = None

# Title and description
st.title("TD-Join system for temporal dependencies across time series")
# st.write("This app compares two time series and extract the most similar subsequences in the input time series that follows Allen's interval relations.")

# Define color mapping for each column
color_map_best = {
    'equal': 'green',  # Best value
    'overlaps': 'orange',
    'meets': 'blue',
    'before': 'red'  # Red
}
color_map_all = {
    'equal': '#90EE90',  # Light Green
    'overlaps': '#FFD580',  # Light Orange
    'meets': '#ADD8E6',  # Light Blue
    'before': '#FFB6C1'  # Light Red/Pink
}

def get_time_series_data(measurement_name):
    url = os.getenv("url")
    token = os.getenv("token")
    org = os.getenv("org")
    bucket = os.getenv("bucket")

    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: 1970)  // Adjust the time range as needed
      |> filter(fn: (r) => r._measurement == "{measurement_name}")
      |> keep(columns: ["_time", "_value"])
    '''
    tables = query_api.query(query)

    # Extract timestamps and values
    time_series_data = []
    for table in tables:
        for record in table.records:
            time_series_data.append({"_time": record.get_time(), "_value": record.get_value()})

    # Convert to DataFrame
    return pd.DataFrame(time_series_data)


def write_time_series_data(file):
    write_api = client.write_api(write_options=WriteOptions(batch_size=1000, flush_interval=10_000))
    data = load_time_series(file)
    field_key = data.columns[1]
    if data is not None:
        for _, row in data.iterrows():
            point = influxdb_client.Point(file.name.split('.')[0]).field(field_key, float(row[1])).time(row[0])
            write_api.write(bucket=os.getenv("bucket"), org=os.getenv("org"), record=point)
    write_api.close()

def update_dropdowns():
    query_api = client.query_api()
    query = f'''
    import "influxdata/influxdb/schema"
    schema.measurements(bucket: "{os.getenv("bucket")}")
    '''
    tables = query_api.query(query)
    return [record.get_value() for table in tables for record in table.records]

# Function to apply styles
def get_min_values(df):
    min_values = {}
    for col in df.columns:
        valid_numbers = [float(x[2]) for x in df[col] if x is not None and len(x) > 2 and x[2] is not None]
        min_values[col] = min(valid_numbers) if valid_numbers else None
        # print(f"Column: {col}, Min Value: {min_values[col]}")
    return min_values

# Load the time series data
def load_time_series(file):
    try:
        data = pd.read_csv(file)
        if len(data.columns) < 2:
            st.error("The file must have at least two columns (e.g., Time, Value).")
            return None
        return data
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        return None

# Dropdown menu for Allen's interval relations  (simplified names)
relations = ["before","meets","equal","overlaps"]
algorithms = ["TD_Join (our)","STAMP","STUMP"]

selected = option_menu(
    menu_title=None,  # Hide the menu title
    options=["Data import","Filtering", "Augmenting"],
    icons=["filter", "bar-chart"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

load_dotenv()
if selected == "Data import":
    url=os.getenv("url")
    token=os.getenv("token")
    org=os.getenv("org")
    bucket=os.getenv("bucket")
    client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

    # Extract measurement names
    measurements =  update_dropdowns()

    # File uploaders for time series files
    st.header("Upload Time Series Files")
    uploaded_files = st.file_uploader("Upload Time Series Files (CSV) to InFluxDB", type=["csv"], accept_multiple_files=True)




    if uploaded_files and st.button("Upload Time Series"):
        for i, file in enumerate(uploaded_files):
            write_time_series_data(file)
        measurements = update_dropdowns()

    selected_file1 = st.selectbox("Select a Time Series", measurements)
    selected_file2 = st.selectbox("Select a Time Series ", measurements)
    data1 = get_time_series_data(selected_file1)
    data2 = get_time_series_data(selected_file2)
    st.session_state.data1 = data1
    st.session_state.data2 = data2
    # Check if both files are uploaded
    if data1 is not None and data2 is not None:
        if st.button("Plot time series"):
            # Extract columns for plotting
            st.session_state.value1 = st.session_state.data1.iloc[:, 1].astype(np.float64)
            st.session_state.value2 = st.session_state.data2.iloc[:, 1].astype(np.float64)

            # Generate plots
            st.subheader("Time Series Comparison")
            col1, col2 = st.columns(2)
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
            # Plot for Time Series 1
            with col1:
                ax1.plot(st.session_state.value1, label="Time Series 1")
                ax1.set_xlabel("Time",fontsize=16)
                ax1.set_ylabel("Values", fontsize=16)
                ax1.set_title(selected_file1.split(".")[0],fontsize=22)
                ax1.legend()
                st.pyplot(fig1)

            # Plot for Time Series 2
            with col2:
                ax2.plot(st.session_state.value2, label="Time Series 2", color="orange")
                ax2.set_xlabel("Time",fontsize=16)
                ax2.set_ylabel("Values", fontsize=16)
                ax2.set_title(selected_file2.split(".")[0],fontsize=22)
                ax2.legend()
                st.pyplot(fig2)
    else:
        st.warning("No time series present. Please upload time series files.")

elif selected == "Augmenting":
    if st.session_state.value1 is not None and st.session_state.value2 is not None:
        selected_algorithm = st.selectbox("Select an algorithm", algorithms)
        subsequence_length = st.slider("Choose the size of the window", min_value=3, max_value=max(len(st.session_state.value1)-1, len(st.session_state.value2)-1), value=3)
        check_if_pressed = False
        data = []
        if st.button("Compute all Allen's Motifs"):
            check_if_pressed = True
            fig2, axs2 = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
            axs2[0].plot(st.session_state.value1)
            axs2[1].plot(st.session_state.value2, c='orange')
            axs2[1].set_xlabel('Time',fontsize=12)

            axs2[0].set_ylabel("Values",fontsize=12)
            axs2[1].set_ylabel("Values",fontsize=12)

            ylim_lower1 = st.session_state.value1.min() - 10
            ylim_upper1 = st.session_state.value1.max() + 10
            ylim_lower2 = st.session_state.value2.min() - 10
            ylim_upper2 = st.session_state.value2.max() + 10

            axs2[0].set_ylim(ylim_lower1, ylim_upper1)
            axs2[1].set_ylim(ylim_lower2, ylim_upper2)

            ts1_timestamp = st.session_state.data1.iloc[:, 0]
            ts1_value = st.session_state.data1.iloc[:, 1].astype(np.float64)
            ts2_timestamp = st.session_state.data2.iloc[:, 0]
            ts2_value = st.session_state.data2.iloc[:, 1].astype(np.float64)
            # Create a list of datapoints for each time series
            ts1= [ DataPoint(ts1_timestamp[i],ts1_value[i]) for i in range(len(ts1_timestamp))]
            ts2= [ DataPoint(ts2_timestamp[i],ts2_value[i]) for i in range(len(ts2_timestamp))]

            if selected_algorithm == "TD_Join (our)":
                ap = TD_Join(T_A=ts1, m=subsequence_length, T_B=ts2)
                data = ap
            elif selected_algorithm == "STUMP":
                ap = mySTUMP.MYSTUMP(T_A=ts1_value, m=subsequence_length, T_B=ts2_value)
                data = ap
            elif selected_algorithm == "STAMP":
                ap = mySTAMP.MYSTAMP(T_A=ts1_value, m=subsequence_length, T_B=ts2_value)
                data = ap


            seq_A_rect = None
            seq_B_rect = None
            for key in ap:
                values = [item[2] for item in ap[key]]
                if values:
                    index = min(range(len(values)), key=lambda i: values[i])
                else:
                    st.warning(f"No subsequences found for the {key} relation. Try with a different subsequence length.")
                    continue
                # Find the index of the minimum value
                index = min(range(len(values)), key=lambda i: values[i])

                # Get the first element (item[0]) corresponding to the minimum value
                seq_A_index = int(ap[key][index][0])
                seq_B_index = int(ap[key][index][1])
                if key == "before":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='red')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='red')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                elif key == "meets":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='blue')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='blue')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                elif key == "equal":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='green')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='green')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                elif key == "overlaps":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='orange')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='orange')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                axs2[0].add_patch(seq_A_rect)
                axs2[1].add_patch(seq_B_rect)

            if selected_algorithm == "TD_Join (our)":
                axs2[0].set_xticks(range(0, len(ts1_timestamp), 10))
                axs2[0].set_xticklabels([ts1_timestamp[i] for i in range(0, len(ts1_timestamp), 10)], rotation=90, fontsize=8)
                axs2[1].set_xticks(range(0, len(ts2_timestamp), 10))
                axs2[1].set_xticklabels([ts2_timestamp[i] for i in range(0, len(ts2_timestamp), 10)], rotation=90, fontsize=8)

            st.pyplot(fig2)

            # Find the maximum length of any column
            max_len = max(len(v) for v in ap.values())

            for key in ap:
                while len(ap[key]) < max_len:
                    ap[key].append([None, None,None])

            df = pd.DataFrame(ap)
            min_values = get_min_values(df)
            first_occurrence = {col: True for col in df.columns}  # Track first occurrence for each column

            def highlight_min(val, col):
                if val and len(val) > 2 and  min_values[col] is not None and val[2] == min_values[col] and first_occurrence[col]:  # Ensure val has at least 3 elements
                    first_occurrence[col] = False  # Mark the minimum as already highlighted
                    return f"background-color: {color_map_best[col]}"
                return None  # Return None to keep the base color

            def highlight_col(x):
                df1 = pd.DataFrame('', index=x.index, columns=x.columns)
                for col in x.columns:
                    df1[col] = f'background-color: {color_map_all[col]}'
                return df1

            styled_df = df.style.apply(highlight_col, axis=None)
            styled_df = styled_df.apply(
                lambda x: [
                    highlight_min(v, x.name) or f'background-color: {color_map_all[x.name]}' for v in x
                ],
                axis=0
            )
            st.header(f"Time-Dependent Matrix Profile")
            st.dataframe(styled_df)
    else:
        st.write("No time series present. Please upload time series files.")
elif selected == "Filtering":
    if st.session_state.value1 is not None and st.session_state.value2 is not None:
        selected_relation = st.selectbox("Choose Allen's Relation", relations)
        selected_algorithm = st.selectbox("Select an algorithm", algorithms)
        subsequence_length = st.slider("Choose the size of the window", min_value=3, max_value=max(len(st.session_state.value1)-1, len(st.session_state.value2)-1), value=3)
        if st.button("Compute Temporal Motifs"):
            value1 = st.session_state.value1
            value2 = st.session_state.value2
            fig2, axs2 = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
            axs2[0].plot(value1, c ='olive')
            axs2[1].plot(value2, c='purple')
            if selected_algorithm == "TD_Join (our)":
                axs2[1].set_xlabel('Time',fontsize=12)
            else:
                axs2[1].set_xlabel('Value',fontsize=12)

            axs2[0].set_ylabel('Values',fontsize=12)
            axs2[1].set_ylabel('Values',fontsize=12)


            ylim_lower1 = value1.min() - 10
            ylim_upper1 = value1.max() + 10
            ylim_lower2 = value2.min() - 10
            ylim_upper2 = value2.max() + 10
            axs2[0].set_ylim(ylim_lower1, ylim_upper1)
            axs2[1].set_ylim(ylim_lower2, ylim_upper2)

            ts1_timestamp = st.session_state.data1.iloc[:, 0]
            ts1_value = st.session_state.data1.iloc[:, 1].astype(np.float64)
            ts2_timestamp = st.session_state.data2.iloc[:, 0]
            ts2_value = st.session_state.data2.iloc[:, 1].astype(np.float64)
            # Create a list of datapoints for each time series
            ts1= [ DataPoint(ts1_timestamp[i],ts1_value[i]) for i in range(len(ts1_timestamp))]
            ts2= [ DataPoint(ts2_timestamp[i],ts2_value[i]) for i in range(len(ts2_timestamp))]

            if selected_algorithm == "TD_Join (our)":
                ap = TD_Join(T_A=ts1, m=subsequence_length, T_B=ts2, Allen_relation=selected_relation)
            elif selected_algorithm == "STUMP":
                ap = mySTUMP.MYSTUMP(T_A=ts1_value, m=subsequence_length, T_B=ts2_value, Allen_relation=selected_relation)
            elif selected_algorithm == "STAMP":
                ap = mySTAMP.MYSTAMP(T_A=ts1_value, m=subsequence_length, T_B=ts2_value, Allen_relation=selected_relation)

            if len(ap[selected_relation]) != 0:
                print(f"Run with algorithm {selected_algorithm}\n", ap[selected_relation])
                values = np.array([item[2] for item in ap[selected_relation]])
                # Find the index of the minimum value

                # values = [item[2] for item in ap[selected_relation]]
                # Find the index of the minimum value
                index = min(range(len(values)), key=lambda i: values[i])

                # Get the first element (item[0]) corresponding to the minimum value
                seq_A_index = int(ap[selected_relation][index][0])
                seq_B_index = int(ap[selected_relation][index][1])


                # data = [[f"{row[0]:.1f}  {row[1]:.5f}"] for row in ap[selected_relation]]
                data = ap
                df = pd.DataFrame({selected_relation: data})
                # Create a DataFrame with an empty first column and the concatenated values in the second column



                if selected_relation == "before":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='red')
                    axs2[0].axvline(x=seq_A_index+subsequence_length, linestyle="dashed", color='red')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='red')
                    axs2[1].axvline(x=seq_B_index+subsequence_length, linestyle="dashed", color='red')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[selected_relation], alpha=0.3)
                elif selected_relation == "meets":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='blue')
                    axs2[0].axvline(x=seq_A_index+subsequence_length, linestyle="dashed", color='blue')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='blue')
                    axs2[1].axvline(x=seq_B_index+subsequence_length, linestyle="dashed", color='blue')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[selected_relation], alpha=0.3)
                elif selected_relation == "equal":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='green')
                    axs2[0].axvline(x=seq_A_index+subsequence_length, linestyle="dashed", color='green')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='green')
                    axs2[1].axvline(x=seq_B_index+subsequence_length, linestyle="dashed", color='green')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor= color_map_best[selected_relation], alpha=0.3)
                elif selected_relation ==  "overlaps":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='orange')
                    axs2[0].axvline(x=seq_A_index+subsequence_length, linestyle="dashed", color='orange')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='orange')
                    axs2[1].axvline(x=seq_B_index+subsequence_length, linestyle="dashed", color='orange')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor= color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor= color_map_best[selected_relation], alpha=0.3)

                axs2[0].add_patch(seq_A_rect)
                axs2[1].add_patch(seq_B_rect)


                if selected_algorithm == "TD_Join (our)":
                    axs2[0].set_xticks(range(0, len(ts1_timestamp), 10))
                    axs2[0].set_xticklabels([ts1_timestamp[i].date() for i in range(0, len(ts1_timestamp), 10)], rotation=90, fontsize=8)
                    axs2[1].set_xticks(range(0, len(ts2_timestamp), 10))
                    axs2[1].set_xticklabels([ts2_timestamp[i].date() for i in range(0, len(ts2_timestamp), 10)], rotation=90, fontsize=8)


                st.pyplot(fig2)

                df = pd.DataFrame({selected_relation: ap[selected_relation]})

                # Get minimum values for the column
                min_values = get_min_values(df)

                # if selected_relation == "meets":
                #     print(min_values, selected_algorithm)

                first_occurrence = {selected_relation: True}  # Track first occurrence for the column

                # Function to highlight the minimum value
                def highlight_min(val, col):
                    if val and len(val) > 2 and val[2] == min_values[col] and first_occurrence[col]:
                        first_occurrence[col] = False
                        return f"background-color: {color_map_best[col]}"
                    return None

                # Function to apply base color to all cells
                def highlight_col(x):
                    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
                    df1[selected_relation] = f'background-color: {color_map_all[selected_relation]}'
                    return df1

                # Apply styling to the DataFrame
                styled_df = df.style.apply(highlight_col, axis=None)
                styled_df = styled_df.apply(
                    lambda x: [
                        highlight_min(v, x.name) or f'background-color: {color_map_all[x.name]}' for v in x
                    ],
                    axis=0
                )

                # Display the styled DataFrame
                st.header(f"Filtered Time-Dependent Matrix Profile")
                st.dataframe(styled_df)
            else:
                st.write("No subsequences found for the selected relation. Try with different subsequence length.")

    else:
        st.write("No time series present. Please upload time series files.")

