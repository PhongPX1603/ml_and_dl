# early_stopping-

## 1. The Problem of Training Just Enough

* When training a large network, there will be a point during training when the model will stop generalizing and start learning the statistical noise in the training dataset.
* This overfitting of the training dataset will result in an increase in generalization error, making the model less useful at making predictions on new data.
* One approach to solving this problem is to treat the number of training epochs as a hyperparameter and train the model multiple times with different values, then select the number of epochs that result in the best performance on the train or a holdout test dataset.

## 2. Stop Training When Generalization Error Increases
* An alternative approach is to train the model once for a large number of training epochs.
* During training, the model is evaluated on a holdout validation dataset after each epoch. If the performance of the model on the validation dataset starts to degrade (e.g. loss begins to increase or accuracy begins to decrease), then the training process is stopped.
* Training can therefore be stopped at the point of smallest error with respect to the validation data set
* Regularization may also be implicit as is the case with early stopping.


## 3. How to Stop Training Early
* Early stopping requires that you configure your network to be under constrained, meaning that it has more capacity than is required for the problem.
* When training the network, a larger number of training epochs is used than may normally be required, to give the network plenty of opportunity to fit, then begin to overfit the training dataset.
* There are three elements to using early stopping: Monitoring model performance, Trigger to stop training, The choice of model to use.

### Monitoring Performance (Theo dõi hiệu suất)
* Validation dataset ís used to monitor performance of the model during training
* It is also common to use the loss on a validation dataset as the metric to monitor, although you may also use prediction error in the case of regression, or accuracy in the case of classification.
* Performance of the model is evaluated on the validation set at the end of each epoch, which adds an additional computational cost during training. This can be reduced by evaluating the model less frequently, such as every 2, 5, or 10 training epochs.

### Early Stopping Trigger (Kích hoạt dừng sớm)
* Once a scheme for evaluating the model is selected, a trigger for stopping the training process must be chosen.
* In the simplest case, training is stopped as soon as the performance on the validation dataset decreases as compared to the performance on the validation dataset at the prior training epoch (e.g. an increase in loss). But some delay in stopping is almost always a good idea.

```Results indicate that “slower” criteria, which stop later than others, on the average lead to improved generalization compared to “faster” ones. However, the training time that has to be expended for such improvements is rather large on average and also varies dramatically when slow criteria are used.```

### Model Choice
* At the time that training is halted, the model is known to have slightly worse generalization error than a model at a prior epoch.
* As such, some consideration may need to be given as to exactly which model is saved. Specifically, the training epoch from which weights in the model that are saved to file.
* This will depend on the trigger chosen to stop the training process. For example, if the trigger is a simple decrease in performance from one epoch to the next, then the weights for the model at the prior epoch will be preferred.
* If the trigger is required to observe a decrease in performance over a fixed number of epochs, then the model at the beginning of the trigger period will be preferred.
* Perhaps a simple approach is to always save the model weights if the performance of the model on a holdout dataset is better than at the previous epoch. That way, you will always have the model with the best performance on the holdout set.

```
Every time the error on the validation set improves, we store a copy of the model parameters. When the training algorithm terminates, we return these parameters, rather than the latest parameters.
```

## 4. Examples of Early Stopping

## 5. Tips for Early Stopping

### When to Use Early Stopping
### Plot Learning Curves to Select a Trigger
### Monitor an Important Metric
### Suggested Training Epochs
### Early Stopping With Cross-Validation 
### Overfit Validation
### Further Reading

## 6. Summary
* You discovered that stopping the training of neural network early before it has overfit the training dataset can reduce overfitting and improve the generalization of deep neural networks.

* The challenge of training a neural network long enough to learn the mapping, but not so long that it overfits the training data.
* Model performance on a holdout validation dataset can be monitored during training and training stopped when generalization error starts to increase.
* The use of early stopping requires the selection of a performance measure to monitor, a trigger for stopping training, and a selection of the model weights to use.
