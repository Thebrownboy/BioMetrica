# BioMetrica

## Problem Definition 
The problem at hand is to develop a machine learning solution for "Body Level Classification" based on a given dataset. The dataset comprises various attributes related to the physical, genetic, and habitual conditions of individuals. These attributes consist of both categorical and continuous variables. The goal is to accurately classify the body level of a person into one of four distinct classes.

With a total of 1477 data samples, it is important to address the class imbalance issue in the dataset. The distribution of classes is uneven, meaning that certain classes may have significantly more or fewer instances than others. Therefore, it is necessary to build models that can effectively adapt to this class imbalance while aiming to achieve the best possible classification results.


## Data Visualization 
- **Imbalanced target feature** 
  <img src="Images/Body_Level.png" style="width:80%" align='center'>
- **Some features also exhibit imbalances, where the majority of their values tend to be skewed towards a single value.**
    <br>
    <img src ='Images/Meal_Count.png' style="width:80%" align='center'>
    <br>
    <img src ='Images/Transport.png' style="width:80%" align='center' >
- **Furthermore, the presence of skewness can also be observed in certain data instances.**
    <br>
    <img src ='Images/Age.png' style="width:80%" align='center' >
    
    
## Data preprocessing 

To prepare the data for analysis, the following preprocessing techniques will be applied:

- Standardization: The feature values will be standardized to have a mean of 0 and a standard deviation of 1, ensuring consistent scaling across different features.

- Log Transformation (for skewed data): When data exhibits skewness, a logarithmic transformation will be applied to reduce the impact of extreme values and achieve a more normal distribution.

- Oversampling (to tackle the imbalancing problem): To address the imbalance in certain features, oversampling techniques such as Synthetic Minority Over-sampling Technique (SMOTE) could be employed.**But we found that the regular random oversampling was a good choice** 

By implementing these preprocessing steps, we aim to improve the quality and suitability of the data for the subsequent stages of the project.

## Insights
**The feature importance analysis reveals that weight, age, and height of the person are the primary features that any model would prioritize in learning their significance. These features can be combined to calculate the Body Mass Index (BMI). However, it is important to note that BMI represents the true function in this problem. Therefore, including it as a feature in the model would be redundant or unnecessary. As a result, we do not require a machine learning model to solve this particular problem.**
<img src="Images/feature_Importance.png" style="width:80%" align='center'>

## Models
- Logistic Regression
- Random forest regression
- SVM
- NN (Neural Network)

### Logistic Regression

**Firstly, we begin with a basic implementation of Logistic Regression without any additional techniques. This initial step allows us to tune the hyperparameters and identify the optimal configuration.**

- Tuning the 'C' Hyperparameter
  <img src='Images/best-c-for-logisticpng' style="width:80%" align='center'>
- Exploring various approaches(class weights (CW) , over-sampling(OS))
  <img src='Images/different_approch.png' style="width:80%" align='center'>
  
  
### Focal Loss (Modifying the learning methodology)


**Instead of relying solely on preprocessing steps to address the issue of imbalanced data, we can explore the possibility of incorporating the knowledge of this problem directly into the learning process itself. By explicitly informing the loss function about the imbalanced nature of the data, we enable it to handle this situation more effectively. This approach can potentially alleviate the need for extensive preprocessing steps specifically aimed at dealing with the imbalance problem**

#### Problems with imbalance dataset 
- No learning due to easy negatives 
- cumulative effect of many easy negatives 
- Cross entropy does not handle the two problems above let's see how can focal loss helps in solving them 
balance between easy and hard examples: 
#### Handling easy examples problems
- The idea is that if a sample is already well-classified, we can significantly decrease or down weigh its contribution to the loss.
- gamma is the modulating factor 
  <img src='Images/Focal_loss1.png' style="width:80%" align='center'>
  
#### cumulative effect of many easy negatives
- To do so, we add a weighting parameter (α), which is usually the inverse class frequency. α  is the weighted term whose value is α for positive class and 1-α for negative 
 <img src='Images/Focal_loss_2.png' style="width:80%" align='center'>

## Results 
<table>
  <thead>
    <tr>
      <td>Model</td>
      <td>train-accuracy</td>
      <td>val-accuracy</td>
      <td>test-accuracy</td>
    </tr>
  </thead>
  
  <tbody>
    <tr>
      <td>before-sampling</td>
      <td>0.9878</td>
      <td>0.983</td>
      <td>0.9715</td>
    </tr>
    <tr>
      <td>after-sampling</td>
      <td>0.9965</td>
      <td>0.9863</td>
      <td>0.993</td>
    </tr>
  </tbody> 
</table>

## Interpreting the Results

As demonstrated earlier, the focal loss approach has yielded the best model performance in terms of accuracy and F1-score, even without any preprocessing steps applied to the data. However, when we further applied oversampling with a 0.5 ratio, we observed even better results in terms of accuracy and F1-score. This indicates that combining the benefits of focal loss with a controlled oversampling strategy can lead to further improvements in model performance. By balancing the class distribution while maintaining the benefits of focal loss, we can effectively address the challenges posed by imbalanced data and achieve enhanced accuracy and F1-score.

