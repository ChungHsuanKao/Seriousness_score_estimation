# Feature enginering

def respond_mean(dataframe):
    temp=[]
    newstart = 0
    for j in range(newstart, len(dataframe)):
        if dataframe.loc[j, 'type'] == 'Outgoing':
            out_time = dataframe.loc[j, 'createdAt']
            
            for t in range(j, len(dataframe)):
                if dataframe.loc[t, 'type'] == 'Incoming':
                    in_time = dataframe.loc[t, 'createdAt']
                    responding = in_time - out_time
                    temp.append(responding.total_seconds())
                    newstart = t
                    break
    
    return statistics.mean(temp)


def respond_median(dataframe):
    temp=[]
    newstart = 0
    for j in range(newstart, len(dataframe)):
        if dataframe.loc[j, 'type'] == 'Outgoing':
            out_time = dataframe.loc[j, 'createdAt']
            
            for t in range(j, len(dataframe)):
                if dataframe.loc[t, 'type'] == 'Incoming':
                    in_time = dataframe.loc[t, 'createdAt']
                    responding = in_time - out_time
                    temp.append(responding.total_seconds())
                    newstart = t
                    break    
    
    return statistics.median(temp)


# Validation
def coef_importance(X_train, model):
    temp = pd.DataFrame(np.std(X_train, 0), columns=['std'])
    temp['coef'] = model.coef_[0].tolist()
    temp['weight'] = temp['std']*temp['coef']
    temp_sort = temp.sort_values('weight', ascending=False)['weight']
    plt.bar(temp_sort.index, temp_sort)
    plt.xticks(rotation=90)
    
def feature_importance(model, X_train):
    # get importances of each feature
    importances = model.feature_importances_
    # get feature name of each column
    feat_labels = X_train.columns
    # arrange the order of importances from larget to small and extract its index
    indices = np.argsort(importances)[::-1]
    plt.style.use('ggplot')
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature importance')

def confusion_matrix_plot(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:\n', conf_mat)

    labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()

def precision_recall_plot(true_label, prediction_prob):
    from inspect import signature
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(true_label, prediction_prob)
    
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(true_label, prediction_prob)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Binary Precision-Recall curve: AP={0:0.2f}'.format(average_precision))



# Visualization

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')

def box_dist_plot(dataframe, column):
    plt.subplot(1,2,1)
    dataframe.boxplot(column)
    
    plt.subplot(1,2,2)
    sns.distplot(dataframe[column])
    
    plt.tight_layout()