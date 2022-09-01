# Core Box Bounding Box Regression

Creation, training, and deployment of a Keras Convolutional Neural Network (CNN)
for the prediction of the bounding box coordinates and core numbers of core box pictures.

### The project is structured in four subsections:

1. Data Preparation
2. Model Building
3. Model Verification
4. Model API

-----------

## 01 - Data Preparation
The prediction labels used for model training, as well as the image file paths are querried from a PostgreSQL database server. The identification token is provided in a JSON file. Additionally the images described in the metadata are copied to a local folder for later usage. Metadata entries without connected image files are deleted.

## 02 - Model Building
The metadata is filtered and rescaled to suit the requirements for the machine learning model. A deep convolutional neural network is built, based on a pretrained Keras application. The model is configured with user defined settings. All images are transformed to suit the models input layer requirements.

The neural network is trained on the labeled training dataset. There are either eight or nine output values to be predicted. The trained model is saved in .h5 format.

*All transformation operations are performed using multi threading*

## 03 - Model Verification
The trained model is used to predict a set of preselected images. The same transformation steps as in the model building process are applied. The predicted values are inversly scaled and rounded to restore the orginal scaling.

A matplotlib figure is plotted showing the core box picture, the predicted core box (red), and the true labeled core box corners (white).

## 04 - Model API
A RestAPI is deployed in a Docker image. It is accessible with a post request and predicts the bounding box coordinates and core number from an image sent in binary format in the request body. The response object is in JSON format. The API supports a setup of the predictor mode and extensive exception tracing.

----------
### There are numerous refinements possible and __suggested__ for the neural network training process:

- Dockerization of model building project for VM training deployment in [Azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python), Datacenter or [Azure Clusters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python).
- Use of MLOps platform for run managment ([Weights & Biases](https://wandb.ai/site), [mlflow](https://mlflow.org/), [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/overview-what-is-machine-learning-studio))
- Implementation of Hyperparameter Optimization ([Weights & Biases](https://docs.wandb.ai/guides/sweeps), [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters), [Ray Tune](https://docs.ray.io/en/latest/tune/index.html))
- Further development of RestAPI with implementation of model update procedures
- Implementation of SSL Encryption for RestAPI
- Transfer of RestAPI to Azure cloud
- Use more and **better labeled** images for model training
- Implement further object recognition tasks like:
  - Removal of color distortion / perspective curvature from core box pictures
  - Color recognition of cm sections
  - Recognition of bedding / bedding direction / bedding thickness
  - Recognition of fractures and brittleness 

------
Overall target is the automated extraction of new data from cores and storage in Azure cloud based database systems for further analysis (e.g. clustering project).

--------
## Docker Guide
1. Make sure docker is installed and running on server system
2. Transfer api docker folder to server system (scp e.g.)
3. Build docker image from Dockerfile 
4. Run docker container with binding external port (internal port: 8000)
5. Make sure container is running when closing ssh session