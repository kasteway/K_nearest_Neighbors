# K_Nearest_Neighbors
## from sklearn.neighbors import KNeighborsClassifier


### Summary:


The k-Nearest Neighbors (k-NN) algorithm is a straightforward and versatile machine learning method used for both classification and regression tasks. It operates on the principle that similar data points are usually close to each other in the feature space. The core parameter of k-NN is 'k', which represents the number of nearest neighbors considered for making predictions. The algorithm determines these neighbors using distance metrics, such as Euclidean or Manhattan distance. Unlike many machine learning algorithms, k-NN does not require an explicit training phase; instead, it uses the entire training dataset during the prediction phase. 

While k-NN is known for its simplicity and ease of implementation, it has certain drawbacks. It can be computationally intensive, especially with large datasets, as it requires calculating the distance to every training data point. Also, it has a high memory requirement due to the need to store the entire dataset. The algorithm's performance can be affected by irrelevant features and the scale of the data, making feature scaling a necessary step. Additionally, selecting the optimal number of neighbors ('k') can be challenging and significantly impacts the algorithm's effectiveness. Despite these challenges, k-NN is a popular choice for many applications due to its intuitiveness and effectiveness in scenarios where the dataset is representative of the space it models.

In summary, while k-NN is valued for its simplicity, lack of a training phase, and effectiveness in certain scenarios, it also faces challenges such as computational inefficiency, sensitivity to irrelevant features, and the need for careful tuning and preprocessing of data. K-NN is a versatile and easy-to-understand algorithm for classification and regression, but its simplicity can also lead to challenges, particularly in terms of computational efficiency and sensitivity to the data's scale and quality.






---

### Advantages & Disadvantages:

#### Advantages:
- Simplicity and Intuitiveness: k-NN is straightforward to understand and implement, making it an excellent starting point for beginners in machine learning.

- No Training Phase: Unlike many algorithms, k-NN doesn't require a training phase. It stores the training dataset and makes predictions by searching through this dataset, which can be an advantage in certain scenarios.

- Versatility: It can be used for both classification (assigning labels) and regression (predicting continuous values).

- Flexibility with Distance Function: k-NN allows for the flexibility of choosing the type of distance metric (like Euclidean, Manhattan, etc.), which can be tuned based on the type of data.

- Naturally Handles Multi-Class Cases: The algorithm can easily handle classification in scenarios with multiple classes.

- Effective with Sufficient Representative Data: When the dataset is large enough to capture the diversity of possible cases, k-NN can be quite effective.



#### Disadvantages:
- Computationally Intensive: As the dataset grows, the prediction step becomes slower because the algorithm searches through the entire dataset for the nearest neighbors.

- High Memory Requirement: k-NN requires storing the entire training dataset for use during the prediction phase, which can be memory-intensive.

- Sensitive to Irrelevant Features: Since it relies on the distance between data points, k-NN can be negatively impacted by features that do not contribute to the overall patterns but are included in the distance calculation.

- Sensitivity to Imbalanced Data: In classification tasks, if one class is significantly more frequent than others, k-NN can be biased towards this majority class.

- Feature Scaling Required: k-NN is sensitive to the scale of the data. Features need to be normalized or standardized, so they contribute equally to the distance calculations.

- Difficulty in Choosing the Right 'k': Selecting the optimal number of neighbors (k) is crucial and can be challenging. A value too small can be noisy and subject to the effects of outliers, while a value too large may smooth over the data's actual structure.

- Boundary Problem: In areas where the class boundary is not well-defined, the algorithm might struggle to make accurate predictions.


---
---

### Data:

The data set used for this analysis comes from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom). This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525).  Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended.  This latter class was combined with the poisonous one.  The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

- Dataset Characteristics -> Multivariate

- Subject Area -> Biology

- Associated Tasks -> Classification

- Feature Type -> Categorical

- Instances -> 8124

- Features -> 22



---

### Tips:

- No Model training as it uses entire data set to learn
- Feature Scaling Required: k-NN is sensitive to the scale of the data therefore is always good to scale
- Sensitivity to Imbalanced Data: In classification tasks, if one class is significantly more frequent than others, k-NN can be biased towards this majority class.
- K should be odd: Difficulty in Choosing the Right 'k' use grid research
- 
- knn_model = KNeighborsClassifier(n_neighbors=1)



---
### Model:

knn_model = KNeighborsClassifier(n_neighbors=1)

test_error_rates = []

for k in range(1,30):

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train,y_train) 
   
    y_pred_test = knn_model.predict(scaled_X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)


plt.figure(figsize=(10,6),dpi=200)
plt.plot(range(1,30),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")

---
