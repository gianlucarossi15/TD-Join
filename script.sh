#!/bin/bash

# Start the InfluxDB server in the background
influxd &
INFLUXD_PID=$!
echo "InfluxDB server started with PID $INFLUXD_PID"

# Start the Streamlit app in the background
streamlit run app.py &
STREAMLIT_PID=$!
echo "Streamlit app started with PID $STREAMLIT_PID"

# Wait for both processes to complete
wait $INFLUXD_PID $STREAMLIT_PID

# Cleanup on exit
trap "kill $INFLUXD_PID $STREAMLIT_PID" EXIT