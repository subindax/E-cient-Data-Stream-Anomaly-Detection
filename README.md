# Efficient-Data-Stream-Anomaly-Detection

This project simulates a real-time data stream and detects anomalies using three algorithms: Z-Score, Isolation Forest, and One-Class SVM. The algorithms monitor the stream and identify unusual patterns or outliers in the data. A visualization displays the data stream and highlights detected anomalies in real-time. This system offers flexible anomaly detection, balancing simplicity and computational efficiency depending on the chosen algorithm.

•	Data Stream Simulation: This function generates a stream of values, with occasional anomalies added randomly.
•	Z-Score Anomaly Detector: Uses a sliding window to compute mean and standard deviation, and detects anomalies based on the Z-Score.
•	Isolation Forest Anomaly Detector: Uses Isolation Forest to identify anomalies based on the assumption that anomalies are few and different from the rest of the data.
•	One-Class SVM Detector: Uses Support Vector Machines to identify anomalies by learning from the data and separating the normal points from anomalies.
•	Visualization: A real-time plot that shows the data stream, marking anomalies with red dots as they are detected.
