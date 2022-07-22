# Sales Rate Prediction Using Recurrent Neural Network

Sales rate prediction is very important and have a deep impact in every company’s economic development. A rise or fall in the amount of sales has an 
immense role in determining the amount of production. This determination is not limited only for Shampoo product but it can be possible to every kind of 
industrial productions. However, this work is restricted for Sales of Shampoo products. The dataset (Sales of Shampoo) is a single featured dataset 
(There have just a single value i.e., the amount of sales of Shampoo per month). I used 2 most important Recurrent Neural Networks named LSTM and GRU for 
time series data, and finally made comparison between them. Here, sales amount prediction has been done for next 5 months. By comparing the results of the prediction 
figures and mean squared errors, it is concluded that GRU architecture performs better than LSTM one.

##	Dataset and Data Preprocessing

The Dataset is collected from [Time Series Data Library (TSDL)](https://www.kaggle.com/datasets/dougcresswell/shampoo-sales-2001-2003). Time series datasets are most popular in recent years. The Sales of shampoo over a three years period is one of them. It is last updated 1 Feb 2014. The data in that dataset is collected for 3 years i.e., 36 months sales information. The time range in this time series data is per month. So the dataset has 36 samples and only 2 features- date and amount. The date feature is for time series data. Among the samples 31 data samples are used for training and the rest 5 are used to find the prediction performance. 

## Models we Tested

![LSTM](https://github.com/nazmul729/Deep_Learning_Project/blob/main/LSTM.png)

![GRU](https://github.com/nazmul729/Deep_Learning_Project/blob/main/GRU.png)

## Empirical Evaluation


![LSTM_Prediction](https://github.com/nazmul729/Deep_Learning_Project/blob/main/LSTM_Prediction.png)

![GRU_Prediction](https://github.com/nazmul729/Deep_Learning_Project/blob/main/GRU_Prediction.png)

## Credits

- Special thanks to Professor [Dr Anthony S. Maida](https://people.cmix.louisiana.edu/maida/)
- Implemented this project using Python frameworks: Numpy, Pandas, Scikit-learn.
