# Earthquake-Prediction
Earthquake Prediction using Random Forest Classifier (CPU, Parallel, GPU)
Project Overview
This project is focused on building an earthquake prediction model using the Random Forest Classifier. The primary goal is to classify earthquakes based on two outputs: Depth and Magnitude. The depth is categorized as either Shallow or Deep, and the magnitude is categorized as either Low Magnitude or High Magnitude. The project compares the performance of Random Forest Classifier in three different computing environments: standard CPU-based execution, parallel processing using all CPU cores, and GPU-accelerated execution using cuML’s RandomForestClassifier.

The model is designed to help predict potential earthquake severity and depth based only on the geospatial input features, namely Latitude and Longitude. This classification-based approach is practical when real-time or near-real-time predictions are required in a risk management scenario, where understanding if an earthquake is shallow and high magnitude can trigger different emergency protocols compared to deep and low magnitude events.

Dataset Overview
The dataset used in this project contains earthquake records with their Latitude and Longitude as input features. The Magnitude and Depth are both converted into binary labels for classification. The depth is labeled as 0 for Shallow and 1 for Deep, while the magnitude is labeled as 0 for Low Magnitude and 1 for High Magnitude. These labels allow the model to perform binary classification for both targets.

Features of the Project
The workflow starts with data preprocessing, where the dataset is cleaned by handling missing values and creating labels for classification tasks. Exploratory Data Analysis (EDA) is performed to visualize the earthquake data. This includes scatter plots of earthquake locations (Latitude vs Longitude) colored by both Depth and Magnitude labels, as well as a correlation heatmap to understand the relationships between the features.

For the modeling part, a Random Forest Classifier is trained in three versions. First, a non-parallel version is trained using the CPU with default settings. Second, a parallelized version is trained using n_jobs=-1, which allows the model to utilize all available CPU cores for faster training. Third, a GPU-based Random Forest Classifier is trained using cuML, leveraging CUDA-enabled GPUs for high-speed computation.

The performance of each version is evaluated based on accuracy for both the Depth Label and the Magnitude Label, and the execution time is recorded to compare computational efficiency. Despite similar accuracy across the CPU, parallel CPU, and GPU versions (around 90% accuracy for both tasks), the GPU implementation increased the
training time but got around 90% accuracy.

How to Run the Code
To run the project, first clone the repository from GitHub. Then, open the notebook in Google Colab or any compatible Jupyter environment. For Colab, enable GPU runtime by going to Runtime → Change runtime type → GPU. The required libraries can be installed directly in the Colab environment using simple pip commands. The standard scikit-learn library is used for the CPU and parallel versions of Random Forest, while the cuML library from NVIDIA’s RAPIDS ecosystem is used for the GPU implementation.

In Colab, you can install cuML with:
bash
!pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
Other dependencies include scikit-learn, matplotlib, and seaborn for visualization.

Evaluation and Results
The model achieves approximately 85% accuracy for both the Depth classification and the Magnitude classification tasks across all three execution modes. However, the main difference lies in the execution time. The CPU version takes the longest time to train, the parallel version is faster due to multi-threading with n_jobs=-1, and the GPU version using cuML is the fastest, achieving a significant reduction in computation time. Confusion matrices and classification reports are generated in the notebook to visualize the prediction performance and identify any potential issues with class balance or misclassification.

Visualizations Included
The project includes several visualizations for better understanding the data and results. These include scatter plots of earthquake locations, categorized by depth and magnitude, as well as a correlation heatmap to observe feature relationships. Additionally, confusion matrices are plotted to show the performance of the classifier for both labels.

Project Structure
The core of the project is contained in a single Jupyter notebook called MLPC_.ipynb. This notebook includes the full workflow from data loading and cleaning, through model training and evaluation, to visualization of the results. The notebook is self-contained and ready to run in Google Colab or on local machines equipped with GPU support.

License
This project is distributed under the MIT License, allowing open use, modification, and distribution for academic or commercial purposes.

Author
This project was developed by Prapanna Upadhyay. You can find more of my work on GitHub.

Final Notes
This project focuses only on classification using the Random Forest Classifier, and compares non-parallel CPU, parallel CPU, and GPU-based training. It does not include regression models. The dual output classification (for both magnitude and depth) and the GPU comparison make this project suitable for real-time applications where both accuracy and speed are critical. The current workflow also provides a reproducible pipeline for evaluating Random Forest classifiers under different computational configurations.



