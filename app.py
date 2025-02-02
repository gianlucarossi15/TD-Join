import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from TD_Join import TD_Join
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
    'Equals': 'green',  # Best value
    'Overlaps': 'orange',
    'Meets': 'blue',
    'Before': 'red'  # Red
}
color_map_all = {
    'Equals': '#90EE90',  # Light Green
    'Overlaps': '#FFD580',  # Light Orange
    'Meets': '#ADD8E6',  # Light Blue
    'Before': '#FFB6C1'  # Light Red/Pink
}
# Function to apply styles


def get_min_values(df):
    min_values = {}
    for col in df.columns:
        valid_numbers = [float(x[1]) for x in df[col] if x[1] is not None]
        min_values[col] = min(valid_numbers) if valid_numbers else None
    # print(df.columns)
    # print(df[col][0])
    # print(min_values)
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
relations = ["Before","Meets","Equals","Overlaps"]
algorithms = ["TD_Join (our)","STAMP","STOMP"]

selected = option_menu(
    menu_title=None,  # Hide the menu title
    options=["Data import","Filtering", "Augmenting"],
    icons=["filter", "bar-chart"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

names = []
if selected == "Data import":
    # File uploaders for time series files
    st.header("Upload Time Series Files")
    uploaded_files = st.file_uploader("Upload Time Series Files (CSV) to InFluxDB", type=["csv"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) >= 2:
        for i, file in enumerate(uploaded_files):
            names.append(file.name)
            if i == 0:
                st.session_state.data1 = load_time_series(file)
            if i == 1:
                st.session_state.data2 = load_time_series(file)

        selected_file1 = st.selectbox("Select a Time Series", names)
        selected_file2 = st.selectbox("Select a Time Series ", names)
        # Check if both files are uploaded
        if st.session_state.data1 is not None and st.session_state.data2 is not None:
            if st.button("Plot time series"):
                # Extract columns for plotting
                st.session_state.value1 = st.session_state.data1.iloc[:, 1].astype(np.float64)
                st.session_state.value2 = st.session_state.data2.iloc[:, 1].astype(np.float64)

                # Generate plots
                st.subheader("Time Series Comparison")
                col1, col2 = st.columns(2)
                if selected_file1 == "AAPL.csv" and selected_file2 == "AMD.csv":
                    fig1, ax1 = plt.subplots()
                    fig2, ax2 = plt.subplots()
                    # Plot for Time Series 1
                    with col1:
                        ax1.plot(st.session_state.value1, label="Time Series 1")
                        ax1.set_xlabel("Points")
                        ax1.set_ylabel("Values")
                        ax1.set_title("Time Series 1")
                        ax1.legend()
                        st.pyplot(fig1)

                    # Plot for Time Series 2
                    with col2:
                        ax2.plot(st.session_state.value2, label="Time Series 2", color="orange")
                        ax2.set_xlabel("Points")
                        ax2.set_ylabel("Values")
                        ax2.set_title("Time Series 2")
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
            axs2[1].set_xlabel('Points')

            axs2[0].set_ylabel('Values')
            axs2[1].set_ylabel('Values')

            ylim_lower1 = st.session_state.value1.min() - 10
            ylim_upper1 = st.session_state.value1.max() + 10
            ylim_lower2 = st.session_state.value2.min() - 10
            ylim_upper2 = st.session_state.value2.max() + 10

            axs2[0].set_ylim(ylim_lower1, ylim_upper1)
            axs2[1].set_ylim(ylim_lower2, ylim_upper2)

            ap = TD_Join(T_A=st.session_state.value1, m=subsequence_length, T_B=st.session_state.value2)
            data = ap

            seq_A_rect = None
            seq_B_rect = None
            for key in ap:
                values = [ap[key][i] for i in range(len(ap[key]))]
                if len(values) == 0:
                    continue
                values = np.array(values)[:, 1]
                seq_A_index = np.argmin(values)
                seq_B_index = ap[key][np.argmin(values)][0]
                if key == "Before":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='red')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='red')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                elif key == "Meets":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='blue')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='blue')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                elif key == "Equals":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='green')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='green')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                elif key == "Overlaps":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='orange')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='orange')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[key], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[key], alpha=0.3)
                axs2[0].add_patch(seq_A_rect)
                axs2[1].add_patch(seq_B_rect)
            st.pyplot(fig2)

            # Find the maximum length of any column
            max_len = max(len(v) for v in ap.values())

            # Pad shorter lists with [None, None]
            for key in ap:
                while len(ap[key]) < max_len:
                    ap[key].append([None, None])

            df = pd.DataFrame(ap)
            # Get the minimum second value for each column
            min_values = get_min_values(df)
            # def highlight_min(val, col):
            #     if val and val[1] == min_values[col]:  # Check if second value matches min
            #         return f"background-color: {color_map_best[col]}"
            #     return ""
            # styled_df = df.style.apply(lambda x: [highlight_min(v, x.name) for v in x], axis=0)
            # st.header(f"Time-Dependent Matrix Profile")
            # st.dataframe(styled_df)
            def highlight_min(val, col):
                if val and val[1] == min_values[col]:  # Check if second value matches min
                    return f"background-color: {color_map_best[col]}"
                return ""

            def highlight_col(x):
                df1 = pd.DataFrame('', index=x.index, columns=x.columns)
                for col in x.columns:
                    df1[col] = f'background-color: {color_map_all[col]}'
                return df1

            styled_df = df.style.apply(highlight_col, axis=None)
            styled_df = styled_df.apply(lambda x: [highlight_min(v, x.name) for v in x], axis=0)
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
            axs2[0].plot(value1)
            axs2[1].plot(value2, c='orange')
            axs2[1].set_xlabel('Points')

            axs2[0].set_ylabel('Values')
            axs2[1].set_ylabel('Values')


            ylim_lower1 = value1.min() - 10
            ylim_upper1 = value1.max() + 10
            ylim_lower2 = value2.min() - 10
            ylim_upper2 = value2.max() + 10
            axs2[0].set_ylim(ylim_lower1, ylim_upper1)
            axs2[1].set_ylim(ylim_lower2, ylim_upper2)
            ap = TD_Join(T_A = value1,
                              m = subsequence_length,
                              T_B = value2, Allen_relation=selected_relation)
            if len(ap[selected_relation]) != 0:
                values = np.array([item[1] for item in ap[selected_relation]])
                # Find the index of the minimum value
                seq_A_index = np.argmin(values)
                seq_B_index = ap[selected_relation][seq_A_index][0]

                # data = [[f"{row[0]:.1f}  {row[1]:.5f}"] for row in ap[selected_relation]]
                data = [[row[0],row[1]] for row in ap[selected_relation]]
                df = pd.DataFrame({selected_relation: data})
                # Create a DataFrame with an empty first column and the concatenated values in the second column



                if selected_relation == "Before":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='red')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='red')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[selected_relation], alpha=0.3)
                elif selected_relation == "Meets":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='blue')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='blue')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor=color_map_best[selected_relation], alpha=0.3)
                elif selected_relation == "Equals":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='green')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='green')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor=color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor= color_map_best[selected_relation], alpha=0.3)
                elif selected_relation ==  "Overlaps":
                    axs2[0].axvline(x=seq_A_index, linestyle="dashed", color='orange')
                    axs2[1].axvline(x=seq_B_index, linestyle="dashed", color='orange')
                    seq_A_rect = plt.Rectangle((seq_A_index, ylim_lower1), subsequence_length, ylim_upper1 - ylim_lower1, facecolor= color_map_best[selected_relation], alpha=0.3)
                    seq_B_rect = plt.Rectangle((seq_B_index, ylim_lower2), subsequence_length, ylim_upper2 - ylim_lower2, facecolor= color_map_best[selected_relation], alpha=0.3)

                axs2[0].add_patch(seq_A_rect)
                axs2[1].add_patch(seq_B_rect)
                st.pyplot(fig2)


                min_values = get_min_values(df)

                def highlight_min(val, col):
                    if val and val[1] == min_values[col]:  # Check if second value matches min
                        return f"background-color: {color_map_best[col]}"
                    return ""

                def highlight_col(x):
                    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
                    for col in x.columns:
                        df1[col] = f'background-color: {color_map_all[col]}'
                    return df1

                styled_df = df.style.apply(highlight_col, axis=None)
                styled_df = styled_df.apply(lambda x: [highlight_min(v, x.name) for v in x], axis=0)
                st.header(f"Time-Dependent Matrix Profile")
                st.dataframe(styled_df)

            else:
                st.write("No subsequences found for the selected relation. Try with shorter subsequence.")

    else:
        st.write("No time series present. Please upload time series files.")