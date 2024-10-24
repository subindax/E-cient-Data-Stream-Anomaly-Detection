import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Data Stream Simulation
def data_stream_simulation(n=1000): 
    # This function simulates a data stream of 1000 values.
    # Each value follows a sine wave pattern with added noise and occasional anomalies.
    for i in range(n):
        value = 10 * np.sin(0.1 * i) + np.random.uniform(-1, 1)  # generate a value with a sine wave pattern + noise
        if np.random.random() < 0.05:  # introduce an anomaly 5% of the time
            value += np.random.uniform(20, 30)  # anomaly is a sudden large spike in the value           
        yield value  # yield one value at a time to simulate streaming data

# Z-Score Anomaly Detector
class ZScoreAnomalyDetector:
    # Anomaly detection based on Z-Score, using a sliding window to calculate mean and standard deviation.
    def __init__(self, window_size=50, threshold=3):
        # window_size: the size of the sliding window to calculate mean and standard deviation
        # threshold: Z-Score threshold to consider a value as an anomaly
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)  # a deque (double-ended queue) to store data in a fixed-size window
        
    def detect(self, value):
        # Detect anomalies based on the z-score of the data in the sliding window.
        try:
            self.data_window.append(value)
            if len(self.data_window) < self.window_size:
                return False, None  # Not enough data yet for Z-Score calculation
            
            mean = np.mean(self.data_window)  # Calculate mean of the window
            std_dev = np.std(self.data_window)  # Calculate standard deviation of the window
            
            if std_dev == 0:
                # If standard deviation is zero, raise an error since Z-Score can't be computed
                return ValueError("Standard Deviation is Zero. Cannot compute Z-Score.")
            
            z_score = (value - mean) / std_dev  # Calculate the Z-Score
            # Return True if the Z-Score exceeds the threshold, indicating an anomaly
            return abs(z_score) > self.threshold, z_score
        except ValueError as ve:
            print(f"Error: {ve}")
            return False, None


# Isolation Forest Anomaly Detector
class IsolationForestAnomalyDetector:
    # Anomaly detection using Isolation Forest, an unsupervised learning algorithm.
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)  # contamination=0.05 assumes 5% of data are anomalies
        self.data = []  # Data storage for initial training
    
    def detect(self, value):
        # Append incoming value to data and detect anomalies once enough data is collected.
        self.data.append([value])
        if len(self.data) < 50:  # Wait until there are at least 50 data points
            return False
        if len(self.data) == 50:
            self.model.fit(self.data)  # Train the Isolation Forest model with initial data
        
        prediction = self.model.predict([[value]])  # Predict if the latest value is an anomaly (-1)
        return prediction == -1  # Return True if anomaly is detected


# One-Class SVM Detector
class OneClassSVMAnomalyDetector:
    # Anomaly detection using One-Class SVM, a type of Support Vector Machine for outlier detection.
    def __init__(self):
        self.model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # nu=0.05 assumes 5% of data are anomalies
        self.data = []  # Data storage for initial training
    
    def detect(self, value):
        # Append incoming value to data and detect anomalies once enough data is collected.
        self.data.append([value])
        if len(self.data) < 50:  # Wait until there are at least 50 data points
            return False
        if len(self.data) == 50:
            self.model.fit(self.data)  # Train the One-Class SVM model with initial data
        
        prediction = self.model.predict([[value]])  # Predict if the latest value is an anomaly (-1)
        return prediction == -1  # Return True if anomaly is detected


# Visualization of the Data Stream with Anomalies
def visualize_stream(data_stream, detector):
    # This function visualizes the data stream with detected anomalies in real-time.
    x_data, y_data, anomaly_points = [], [], []  # Arrays to store x (time), y (values), and anomaly points
    fig, ax = plt.subplots()  # Set up the figure and axes for plotting
    ax.set_xlim(0, 100)  # x-axis limits (rolling window of 100 data points)
    ax.set_ylim(-15, 35)  # y-axis limits
    
    line, = ax.plot([], [], lw=2)  # Line plot for the normal data stream
    scatter, = ax.plot([], [], 'ro')  # Scatter plot for detected anomalies (red dots)
    
    def update(frame):
        # Update the plot with new data and check for anomalies.
        x_data.append(len(x_data))  # Append current time (index)
        y_data.append(frame)  # Append current data value
        
        if len(x_data) > 100:
            ax.set_xlim(len(x_data)-100, len(x_data))  # Move the x-axis window forward as more data comes in
        
        line.set_data(x_data, y_data)  # Update the line plot with new data
        
        is_anomaly = detector.detect(frame)  # Check if the current value is an anomaly
        if is_anomaly:
            anomaly_points.append((len(x_data), frame))  # If anomaly detected, store the point
            
        scatter.set_data([p[0] for p in anomaly_points], [p[1] for p in anomaly_points])  # Update scatter plot with anomalies
        return line, scatter
    
    ani = FuncAnimation(fig, update, frames=data_stream, blit=True, interval=100)  # Create animation of the data stream
    plt.show()  # Display the plot


# Main Script
if __name__ == "__main__":
    data_stream = data_stream_simulation()  # Start data stream simulation
    
    # Choose which anomaly detector to use
    print("Choose anomaly detector:")
    print("1: Z-Score")
    print("2: Isolation Forest")
    print("3: One-Class SVM")
    choice = input("Enter 1, 2, or 3: ")  # Get user input for the anomaly detector
    
    # By selecting, initialize an anomaly detection mechanism   
    if choice == '1':
        detector = ZScoreAnomalyDetector(window_size=50, threshold=3)  
    elif choice == '2':
        detector = IsolationForestAnomalyDetector()  
    elif choice == '3':
        detector = OneClassSVMAnomalyDetector()  
    else:
        print("Invalid choice. Using Z-Score as default.")
        detector = ZScoreAnomalyDetector(window_size=50, threshold=3)  # Default to Z-Score detector

    visualize_stream(data_stream, detector)  # Start visualization with the chosen anomaly detector