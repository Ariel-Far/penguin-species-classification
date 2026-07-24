#!/usr/bin/env python
# coding: utf-8

# ## Group contributions
# Ariel Farzan, Ethan Crofut, Phineas Fritsch
# 
# Contribution statement: We all collaborated equally in the project. We all worked on examining and cleaning the data equally. Ariel focused on perceptron modelling, Ethan did logistic regression and Phineas did LDA modeling. As well we all collaborated on deciding and coding the statistics of the data set and tables we created to examine the data in a quantitative manner. Ariel worked on the scatter plot,Ethan worked on the box plot and bar chart, and Phineas worked on the histogram. We all worked together on identifying from the quantitative data which of the parameters are the best to choose for our final models. We all worked together on the model evaluation function to compare all the models and feature choices and choose the best one. Ethan worked on the accuracy testing for our models.

# ## Data import
# First we must retrieve the data from GitHub and store it in a DataFrame. We'll need to import some packages first, then use pd.read_csv.

# In[ ]:


import numpy as np
import pandas as pd

# I also need to import ssl and change how it handles SSL certificates, otherwise I cannot import the data. This is not a problem for everyone but it is for me, so I must add that here.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# In[ ]:


#retrieve the data and store it in a dataframe.
penguins = pd.read_csv("https://raw.githubusercontent.com/liaochunyang/PIC16/refs/heads/main/PIC16A/data/palmer_penguins.csv")


# ### Remark
# 
# Let's get an idea for how the data looks, i.e. how the rows are organized, what features there are, etc.

# In[ ]:


penguins


# ### Remark
# 
# There seems to be a lot of useless information and NaN values. For good practice, we will refrain from handling NaNs until after train-test split; to decide whether we can just dropna or have to fill in NaN values with values such as averages, we will need to see how much data would remain if we dropna, and if this would be sufficient for modeling. First we can simply count NaNs to get an idea for this.

# In[ ]:


#find number of nan values using .isna and .info
nan_counts = penguins.isna().sum()
print("NAN_COUNTS")
print(nan_counts)
print("\nPENGUINS.INFO")
print(penguins.info())


# ### Remark
# That's not too bad; while there are tons of NaNs in the Comments column, these will be useless for modeling. We will be able to simply dropna without having to worry about more complex handling.
# Granted, if we were to dropna right now, we would have almost no data left due to the tremendous amount of NaNs in comments. For that matter, when we do our final modeling, we will want to drop NaN only after selecting features (so that we do not eliminate data based on columns that will not end up being used anyway), but for exploratory analysis of the data we may as well just take out "Comments" and then dropna so that we can see things like correlation matrices. There should be enough data for this.
# 
# We will now move forward with exploratory analysis.

# ## Data Cleaning
# Now we've successfully imported our data, we need to perform some rudementory data cleaning before analyzing the data. Let's first print the head of our dataframe and a list of the columns

# In[ ]:


print(penguins.head())
pen_cols = penguins.columns
print(f"The columns for the current dataframe are: {pen_cols}")


# Looking at the outputs above, we can categorize columns into categorical, numerical, and to be dropped

# In[ ]:


columns_to_drop = [
    'studyName',
    'Individual ID',
    'Comments',
    'Sample Number'
]

categorical_columns = [
    'Species',
    'Clutch Completion',
    'Date Egg',
    'Island',
    'Sex',
]

# Based on the dataframe structure, these are the remaining columns that are numeric:
numerical_columns = [
    'Culmen Length (mm)',
    'Culmen Depth (mm)',
    'Flipper Length (mm)',
    'Body Mass (g)',
    'Delta 15 N (o/oo)',
    'Delta 13 C (o/oo)'
]

# Optionally, drop the unwanted columns from the dataframe
penguins.drop(columns=columns_to_drop, inplace=True)
penguins


# Now this looks much better, but more cleaning is needed. We need to make sure we don't have any constant values across the dataframe. To ensure this we can use a useful function *nunique()* which will count the number of unique entries in a row/column.

# In[ ]:


constant_cols = penguins.columns[penguins.nunique() <= 1]
list_constant_cols = constant_cols.tolist()
penguins.drop(columns=list_constant_cols, inplace=True)
penguins_original = penguins.copy() # Need to copy for later use, something happens inplace that changes penguins somehow
penguins


# This now results in a sufficiently cleaned dataframe which is now ready for exploratory analysis.

# ## Exploratory analysis
# We would like to do a correlation matrix to see which features are important for predicting the chosen label ("Species"). This won't work given the current format of the data; we will need to handle NaN and encode categorical variables.
# 
# Let's use some additional methods from sklearn.

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# ### Remark
# 
# For the purposes of our correlation matrix, there are also many useless columns. While it'd be ideal to look at everything, we will only end up needing a few features for modeling; we can take out some before doing the correlation matrix, as we would have an incredible excess of columns after one-hot encoding if otherwise.

# In[ ]:


def one_hot(df, cols):
    """
    One-hot encodes a dataframe.
    df: pandas DataFrame
    param: cols a list of columns to encode 
    return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

def label(df, cols):
    """
    Label encodes desired features of a dataframe.
    df: input dataframe
    cols: columns to encode
    Returns:
    A dataframe with the encoded features. 
    """
    
    for each in cols:
        
        label_encoder = LabelEncoder()
        #dummies = label_encoder.fit_transform(df[each])
        #dummies = pd.DataFrame(dummies, columns=[each])
        #df = pd.concat([df, dummies], axis=1)
        
        #label_encoder = LabelEncoder()
        df[each] = label_encoder.fit_transform(df[each])
        
    return df

penguins_one_hot = one_hot(penguins, categorical_columns)
penguins_label_encoded = label(penguins, categorical_columns)


# In[ ]:


penguins_one_hot


# In[ ]:


#We have a lot of categorical features for which there isn't a reasonable way of ranking them, so let's try one hot encoding.
penguins_label_encoded


# Our encoded data still has NaN values, which will break .corr(). Let us take out comments (which is unnecessary) - let us fill NaN values with averages so the results of corr are easily interpretable.
# of course, it would be ideal to not have to use previously NaN averages to look at correlation; we can also plot the correlation with the NaN filled in as 0, as most of the values are not usually 0 (except some encoded categorical variables, but these are not NaN anyway). This way, we can filter out the NaN on the graphs by seeing which ones are at 0.

# In[ ]:


penguins_label_encoded.corr()


# ### Remark
# By looking at the Species row (or column), we are looking for values far away from 0 (these mean a strong correlation, or negative correlation for negative values, both of which would mean that the particular feature might be useful in predicting Species).
# 
# In one-hot encoding there are too many columns to see all of them this way, so let's just extract an array from this table, print that, and then find the most magnitudinous values.

# In[ ]:


corr = penguins_label_encoded.corr()
species_corr = corr.loc["Species"]
sorted_corr = species_corr.abs().sort_values(ascending=False)
sorted_corr


# ### Remark
# It seems that Flipper Length (mm), Body Mass (g), Culmen Depth (mm), Culmen Length (mm) are the most useful numerical values, while Island is the most useful categorical variable, based on these correlations. Let's further analyze culmen lenth, cumlen depth, flipper length, and body mass to determine which of those we use in as our model parameters

# # Table Analysis

# In[ ]:


pd.set_option('display.width', 1000)

summary_stats = penguins_original[['Flipper Length (mm)', 'Body Mass (g)', 'Culmen Length (mm)', 'Culmen Depth (mm)']].describe()
print("Table 1: Summary Statistics of Numerical Variables")
print(summary_stats)

# Table 2: Count of penguins by Species and Island (using groupby)
grouped_counts = penguins_original.groupby(['Species', 'Island']).size().reset_index(name='Count')
print("\nTable 2: Count of Penguins by Species and Island")
print(grouped_counts)

# Table 3: Average measurements by Species (using groupby)
grouped_avg = penguins_original.groupby('Species')[['Flipper Length (mm)', 'Body Mass (g)', 'Culmen Length (mm)', 'Culmen Depth (mm)']].mean().reset_index()
print("\nTable 3: Average Measurements by Species")
print(grouped_avg)


# Here we get a better understanding of what are data looks like and how species differ. Notably we now understand why Island has such a high correlation to species, two of the three species are only on a single island!

# ## Graph Analysis
# To get an even better understanding of our data, we should graph some variables.plt.figure(figsize=(8,6))
# sns.boxplot(data=train_data, x='Species', y='Culmen Depth (mm)', ax=axs[0], hue = 'Island')
# ax.set_title("Boxplot of Culmen Depth by Species")
# ax.set_xlabel("Species")
# ax.set_ylabel("Culmen Depth (mm)")
# ax.set_xticklabels(["Adeline", "Gentoo", "Chinstrap"])

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Figure 1: Boxplot of Culmen Depth by Species
plt.figure(figsize=(8, 6))
sns.boxplot(data=penguins_original, x='Species', y='Culmen Depth (mm)', hue='Island')
plt.title("Boxplot of Culmen Depth by Species")
plt.xlabel("Species")
plt.ylabel("Culmen Depth (mm)")
plt.xticks(ticks=[0, 1, 2], labels=["Adelie", "Gentoo", "Chinstrap"])
plt.show()

# Figure 2: Histogram of Flipper Length
plt.figure(figsize=(8, 6))
sns.histplot(data=penguins_original, x='Flipper Length (mm)', bins=20, kde=True)
plt.title("Distribution of Flipper Length")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Frequency")
plt.show()


# Based on these graphs we can determine two things:
# 1. Based on Culmen Depth alone does a reasonable job at separating Chinstrap from Gentoo and Adelie. However further analysis is needed to see if this is true across multiple variables.
# 2. Flipper Length is bimodal, suggesting that there are likely two different populations, like perhaps species, in the dataset. In other words, the histogram suggests that one can differentiate different populations based on flipper length alone, making it a strong contender for a numerical variable we use
# 
# To help narrow down our numerical choices, we can look at all possible scatter plots comparing different combinations of variables to find how variables interact with eachother in predicting species.

# In[ ]:


import matplotlib.pyplot as plt

# Define the variables to compare.
variables = ['Flipper Length (mm)', 'Culmen Depth (mm)', 'Culmen Length (mm)', 'Body Mass (g)']
n = len(variables)

# Extract species information.
species = penguins_original['Species']
unique_species = species.unique()
colors = ['r', 'g', 'b']
color_map = dict(zip(unique_species, colors))

# Create a figure with a 4x4 grid of subplots.
fig, axs = plt.subplots(n, n, figsize=(16, 16))

# Define a function that plots a scatter plot for any pair of variables.
def scatter_pair(var_x, var_y, ax):
    for sp in unique_species:
        mask = species == sp
        ax.scatter(penguins_original[var_x][mask], penguins_original[var_y][mask],
                   color=color_map[sp], label=sp, s=50, edgecolor='k', alpha=0.7)
    ax.tick_params(axis='both', labelsize=8)

# Loop over every combination of variables.
for i, var_y in enumerate(variables):
    for j, var_x in enumerate(variables):
        ax = axs[i, j]
        scatter_pair(var_x, var_y, ax)
        # Set x-axis label only on the bottom row.
        if i == n - 1:
            ax.set_xlabel(var_x, fontsize=10)
        # Set y-axis label only on the left column.
        if j == 0:
            ax.set_ylabel(var_y, fontsize=10)

plt.tight_layout()
plt.show()


# Here we see many potential promising contenders. Particularly Flipper Length and Culmen Length, Culmen Length and Body Mass, Culmen Length and Flipper Length, and Culmen Length and Culmen Depth. These combinations provide the least overlap between species. However to better assess the true accuracy, it is best to make a decision region.

# ## Feature selection
# 
# While our feature selection is not yet final, we narrowed them down enough to start training models, which can inform our final decision. We will arbitrarily choose classification models, different combinations of features, and see which ones generally perform the best.
# Of course, we won't go in completely blind; it would be smart to begin with plotting decision regions for our models, a this can give us an idea of what pair of numeric features will work well together.
# 
# We'll start with a logistic regression, then plot decision regions for all four of our candidate numeric features. We can judge how effective each of these features is by how well the scatterplot matches up with the model's decision regions. To make sure we're not biased to the abilities of the logistic regression only, we can also try other classification models like support vector machine.
# 
# We will accomplish this by writing a function that can take user-specified features and model type, then plot the penguin species as different colors as a scatterplot, with the model decision regions defined by different colored boundaries. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def plot_decision_regions(data, features_and_label, label="Species", model=LogisticRegression(solver='lbfgs', max_iter=5000)):
    """
    Plot decision regions of a classification model for all pairs of features specified by the user.
    data: source dataframe
    features_and_label: list of strings, all features + label
    label: string, the name of the label
    Returns:
    None
    """
    # Ensure the input list contains at least two features + the label
    if len(features_and_label) < 3:
        raise ValueError("features_and_label must contain the names of at least two features and one label")

    # label encoder alias
    le = preprocessing.LabelEncoder()

    # Select only desired features, label, drop na
    data_sel = data[features_and_label].dropna()
    label_data = data_sel[label]
    feat_data = data_sel.drop(columns=[label])

    # Make tuples of all combinations of features
    #feature_pairs = [(feat_data.columns[i], feat_data.columns[j]) for i in range(len(feat_data.columns)) for j in range(i + 1, len(feat_data.columns))]
    feature_pairs = [(feat_data.columns[i], feat_data.columns[j]) for i in range(len(feat_data.columns)) for j in range(len(feat_data.columns)) if i != j]
    
    # Set up subplots according to feature amount
    n_plots = len(feature_pairs)
    n_cols = min(3, n_plots)  # Limit columns to at most 3 so they don't get too squished
    n_rows = (n_plots // n_cols) + (n_plots % n_cols > 0)  # Calculate required rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of a single row/column

    # Label encoding
    y = le.fit_transform(label_data)

    # Collect handles and labels for the legend
    handles_list = []
    labels_list = []

    # Plot decision region graphs for each pair of features
    for idx, (feature_x, feature_y) in enumerate(feature_pairs):
        X = feat_data[[feature_x, feature_y]]
        model.fit(X, y)

        # Grid
        grid_x = np.linspace(X[feature_x].min(), X[feature_x].max(), 501)
        grid_y = np.linspace(X[feature_y].min(), X[feature_y].max(), 501)
        xx, yy = np.meshgrid(grid_x, grid_y)

        # Model prediction
        XY = pd.DataFrame({feature_x: xx.ravel(), feature_y: yy.ravel()})
        p = model.predict(XY).reshape(xx.shape)

        # Plotting
        ax = axes[idx]
        contour = ax.contourf(xx, yy, p, cmap="jet", alpha=0.2)
        scatter = ax.scatter(X[feature_x], X[feature_y], c=y, cmap="jet", edgecolor="k")
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(f"{feature_x} vs {feature_y}")

        # Collect handles and labels for the legend
        if idx == 0:  # Only create the legend on the first subplot
            species_labels = le.classes_  # Get the unique species names from the label encoder
            handles, labels = scatter.legend_elements()
            handles_list.extend(handles)
            labels_list.extend(species_labels)

    # Remove empty subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Create the single legend outside of the plot area
    fig.legend(handles_list, labels_list, title=label, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()


# In[ ]:


plot_decision_regions(penguins, ["Flipper Length (mm)","Body Mass (g)","Culmen Length (mm)","Culmen Depth (mm)","Species"], "Species", LogisticRegression(solver='lbfgs', max_iter=5000))


# These plots show that specific pairs of features work well with each other. Culmen length seems to be particularly good, as every plot that involves culmen length looks reasonably accurate. The three plots involving culmen length look similar, so we will try training models and calculating the accuracy scores for these three combinations of features.
# 
# We will accomplish this by writing a function where the features and model type are user-specified.
# 
# Further, to ensure our comparison of the models and feature selection are accurate, we will need to perform cross-validation, or in other words split the data into blocks ("folds"), then take different combinations of those data blocks for the training and test data. This will gives us a bunch of different individual models from each iteration, and we can gauge the quality of the model as well as the features by taking the average accuracy across these iterations.
# We will also examine a confusion matrix (a table reporting the classification accuracies for each class) to judge accuracy.

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# In[ ]:


# cross validation and confusion matrix

def test_clf_model_acc_by_cross_validation(data, features=["Flipper Length (mm)","Body Mass (g)","Culmen Length (mm)","Culmen Depth (mm)"], label='Species', k=10, model=LogisticRegression(solver='lbfgs', max_iter=10000)):
    """
    Trains a classification model and obtains average accuracy using a cross validation strategy.
    Args:
    data: dataframe containing input data
    features: a list of strings, the names of the desired features
    label: a string, name of the desired label
    k: number of folds to divide data during cross validation
    model: classification model to be used
    Returns:
    a list containing the accuracy of the model for each fold, the average accuracy overall, and a list containing the confusion matrices for each model
    """
    
    #accuracy list
    acc_list = [] 

    #confusion matrix list
    conf_mat = []

    #generate kfold indices
    kf = KFold(n_splits=k, shuffle=True)

    #make feature and label array, also drop nans as kfold indices are made already
    X = data[features].dropna()
    y = data.loc[X.index, label] #get only the species at the indices with dropped na
    
    #Split according to kfolds
    for train_idx, test_idx in kf.split(X): #where X is the ... cleaned? feature array. 

        #train test split according to kf
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model.fit(X_train, y_train)

        # Test model
        y_pred = model.predict(X_test)

        # Compute accuracy
        acc = accuracy_score(y_test, y_pred)
        acc_list.append(acc)
        conf_mat.append(confusion_matrix(y_test, y_pred))
    
    return acc_list, np.mean(acc_list), conf_mat


# In[ ]:


accuracies, mean, conf = test_clf_model_acc_by_cross_validation(penguins, k=10, model=LogisticRegression(solver='lbfgs', max_iter=10000), features=['Culmen Length (mm)',"Culmen Depth (mm)", "Island"], label='Species')
print(accuracies)
print(mean)
for i in range(len(conf)):
    print(conf[i])


# In[ ]:


accuracies, mean, conf = test_clf_model_acc_by_cross_validation(penguins, k=10, model=LinearSVC(C=20,loss="hinge"), features=['Culmen Length (mm)',"Culmen Depth (mm)", "Island"], label='Species')
print(accuracies)
print(mean)
for i in range(len(conf)):
    print(conf[i])


# Seems that Culmen Length, Culmen Depth, and Island are sufficient for decent models. The support vector machine does not perform as well (albeit we did not optimize the hyperparameter C), but the logistic regression does. With some other models, we should be able to get some more accurate predictions.
# 
# While we did use simple models like logistic regression and support vector machines in the previous section, this was only to avoid bias, and our main goal was feature selection. We still have many more models to choose from, and we should test these to find the best model so that we can make the most accurate predictions.
# 
# We did find that logistic regression was reasonably accurate (>97% average), so we will use this as one of our three models.
# 
# We will now move on to testing the accuracy of different models provided in the sklearn package to find other high performing models, given our selected features (Culmen Length, Culmen Depth, and Island).

# ## Modeling
# We can start with logistic regression, which we know works. We have already seen from cross validation and the confusion matrix that logistic regression is pretty good - the average accuracy scores are consistently good, and the confusion matrices show few misclassified items.
# 
# However, we should further test its ability to extrapolate on unseen data. We will now do a bunch of random train test splits to see if the logistic regression can be consistently accurate.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def test_clf_model_acc(it, data, model=LogisticRegression(solver='lbfgs', max_iter=10000), features_and_label=['Flipper Length (mm)','Body Mass (g)','Culmen Length (mm)','Culmen Depth (mm)','Species'], label='Species'):
    """
    Tests the average accuracy of a classification model over a user-specified number of random train-test splits.
    Args:
    it: int, number of iterations
    data: input dataframe
    model: the type of model to be used. Should be for classification, such as logistic regression or SVM.
    features_and_label: a list containing the names of all features and the label to be considered.
    label: a string indicating the label.
    Returns: a list of the model accuracy (each element corresponds to one iteration from a random split), and the mean accuracy computed from that list.
    """

    #accuracy list
    acc_list = [] 

    #get features only
    feature_list = [f for f in features_and_label if f != label]
    
    for i in range(it):
        
        #train test split with all features
        train_data, test_data = train_test_split(data[features_and_label], test_size=0.2)

        # Post-split drop nan values
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        #
        X_train = train_data[feature_list]
        y_train = train_data[label]

        X_test = test_data[feature_list]
        y_test = test_data[label]

        # Train model
        model.fit(X_train, y_train)

        # Test model
        y_pred = model.predict(X_test)

        # Compute accuracy
        acc = accuracy_score(y_test, y_pred)
        acc_list.append(acc)
    
    return acc_list, np.mean(acc_list)


# In[ ]:


accs, avg = test_clf_model_acc(10, penguins, features_and_label = ['Island','Flipper Length (mm)','Body Mass (g)','Culmen Length (mm)','Species'], model=LogisticRegression(solver='lbfgs', max_iter=10000))
print(f'List of accuracies is {accs} with an average accuracy of {avg}.')


# Even with this test, the accuracy is pretty good. Not all models hold up to this, i.e. we are filtering out the support vector machine which does not perform well enough.
# We'll plot decision regions for these restricted features to see what it's getting wrong.

# In[ ]:


plot_decision_regions(penguins, ["Culmen Length (mm)","Culmen Depth (mm)","Species"], "Species", model=LogisticRegression(solver='lbfgs', max_iter=10000))


# Pretty good decision regions, just some ambiguous almost overlapping points. It would be hard to improve it beyond this point; slightly altering the decision regions doesn't appear that it would increase the accuracy much.
# 
# We already hit our desired accuracy, so we can just move on to something else. For this classification task we may also consider linear discriminant analysis. We will do some cross-validation with it below.

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the raw data
url = "https://raw.githubusercontent.com/liaochunyang/PIC16/refs/heads/main/PIC16A/data/palmer_penguins.csv"
data = pd.read_csv(url)

# -------------------------------
# 1. Split the raw data first
# -------------------------------
# We split before cleaning to prevent data leakage.
raw_train, raw_test = train_test_split(data, test_size=0.25)

# Define feature columns and target
numeric_features = ['Culmen Length (mm)', 'Culmen Depth (mm)']
categorical_features = ['Island']
cols_to_check = numeric_features + categorical_features + ['Species']

# -------------------------------
# 2. Clean and process the training data
# -------------------------------
# Drop rows with missing values in the critical columns on the training set
train_data = raw_train.dropna(subset=cols_to_check)

# Build the numeric feature matrix
X_train_num = train_data[numeric_features].copy()

# Process the categorical feature using one-hot encoding
X_train_cat = pd.get_dummies(train_data[categorical_features])

# Combine numeric and categorical features
X_train = pd.concat([X_train_num, X_train_cat], axis=1)

# Convert features to a NumPy array with type float32
X_train = X_train.values.astype(np.float32)

# Encode the target variable "Species"
le = LabelEncoder()
y_train = le.fit_transform(train_data['Species'])

# -------------------------------
# 3. Clean and process the test data using training set decisions
# -------------------------------
# Drop rows with missing values in the same columns on the test set
test_data = raw_test.dropna(subset=cols_to_check)

# Build the numeric feature matrix for the test set
X_test_num = test_data[numeric_features].copy()

# One-hot encode the categorical feature for the test set
X_test_cat = pd.get_dummies(test_data[categorical_features])

# Ensure the test set has the same dummy columns as the training set:
X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

# Combine numeric and categorical features for the test set
X_test = pd.concat([X_test_num, X_test_cat], axis=1)
X_test = X_test.values.astype(np.float32)

# Encode the target variable in the test set using the same encoder fitted on the training data
y_test = le.transform(test_data['Species'])

kfold = KFold(n_splits=20, shuffle=True)

# Initialize LDA model
model = LinearDiscriminantAnalysis()
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

cv_results


# That's pretty good, a lot of splits give it 100% accuracy. We'll also try a different function to ensure accuracy.

# In[ ]:


accs, avg = test_clf_model_acc(10, penguins, features_and_label = ['Island','Flipper Length (mm)','Body Mass (g)','Culmen Length (mm)','Species'], model=LinearDiscriminantAnalysis())
print(f'List of accuracies is {accs} with an average accuracy of {avg}.')


# Not exactly the same result, perhaps due to different cross-validation settings, but still very good.

# In[ ]:


accuracies, mean, conf = test_clf_model_acc_by_cross_validation(penguins, k=10, model=LinearDiscriminantAnalysis(), features=['Culmen Length (mm)',"Culmen Depth (mm)", "Island"], label='Species')
print(accuracies)
print(mean)
for i in range(len(conf)):
    print(conf[i])


# We're getting something like 97-98% average accuracy for linear discriminant analysis, which is pretty good. We will consider that moving forward.
# 
# It's pretty accurate, but let's see if we can identify why it is making mistakes by examining its decision regions.

# In[ ]:


plot_decision_regions(penguins, ["Culmen Length (mm)","Culmen Depth (mm)","Species"], "Species", LinearDiscriminantAnalysis())


# Decision regions look pretty good. There are some near-overlapping points though, so it would be hard for the model to distinguish these.
# 
# Let us try something more advanced - perceptron, a neural network.

# ## Why a Perceptron Model?
# 
# A perceptron is one of the fundamental building blocks of neural networks. While a basic perceptron can only solve linearly separable problems, adding hidden layers with non-linear activation functions (creating a multi-layer perceptron) allows us to model complex relationships between features and target classes.
# 
# For our penguin classification task, a neural network approach offers several advantages:
# 
# 1. Non-linear decision boundaries: These penguin species have complex  relationships that aren't linearly separable
# 2. Feature interaction modeling: The model can tell how Culmen Length (mm), Culmen Depth (mm), and Species interact wiht eachother.
# 3. Probabilistic output: The softmax output layer provides confidence scores for each class

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Data Loading and Preprocessing
url = "https://raw.githubusercontent.com/liaochunyang/PIC16/refs/heads/main/PIC16A/data/palmer_penguins.csv"
data = pd.read_csv(url)

# Define only the features you want to use:
numeric_features = ['Culmen Length (mm)', 'Culmen Depth (mm)'] 
categorical_features = ['Island']

# Drop rows with missing values in the selected features and target
cols_to_check = numeric_features + categorical_features + ['Species']
data.dropna(subset=cols_to_check, inplace=True)

# Build numeric part
X_num = data[numeric_features].copy()
# One-hot encode the categorical feature(s)
X_cat = pd.get_dummies(data[categorical_features])
# Combine numeric and categorical features
X = pd.concat([X_num, X_cat], axis=1)

# Encode target variable "Species"
le = LabelEncoder()
y_encoded = le.fit_transform(data['Species'])  # values 0,1,2

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

num_classes = 3
Y = one_hot_encode(y_encoded, num_classes)

# Convert features to a NumPy array of type float32
X = X.values.astype(np.float32)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 2. Scaling Layer: Min–Max Scaling
def min_max_scale(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8), X_min, X_max

X_train_scaled, X_min, X_max = min_max_scale(X_train)
X_test_scaled = (X_test - X_min) / (X_max - X_min + 1e-8)

# Model Architecture Design

#Our neural network uses a simple but effective architecture:

# Input Layer: Accepts our preprocessed features (2 numeric + one-hot encoded islands)

# Hidden Layer: Contains 9 neurons with ReLU activation 
#   - ReLU (Rectified Linear Unit) introduces non-linearity while avoiding vanishing gradient problems
#   - 9 neurons provide sufficient capacity to learn the feature relationships without overfitting

# Output Layer: 3 neurons with softmax activation (one for each penguin species)
#   - Softmax converts raw scores to probabilities that sum to 1

input_dim = X_train_scaled.shape[1]
hidden_dim = 9   # perceptron layer with 9 neurons
output_dim = 3   # 3 classes (species)

W1 = np.random.randn(hidden_dim, input_dim) * 0.05
b1 = np.zeros((hidden_dim, 1))
W2 = np.random.randn(output_dim, hidden_dim) * 0.05
b2 = np.zeros((output_dim, 1))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Loss Function

# We use cross-entropy loss, which is standard for multi-class classification problems. This loss function:
# - Penalizes confident but incorrect predictions heavily
# - Provides appropriate gradients for the backpropagation algorithm
# - Works effectively with softmax outputs


def compute_loss(Y_hat, Y):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
    return loss

# Training Algorithm

# We train the network using mini-batch gradient descent with the following hyperparameters:
# - Learning rate: 0.9 (relatively high to accelerate convergence for this simple problem)
# - Epochs*: 250 (sufficient for convergence without overfitting)

# During each training iteration:
# Forward propagation calculates predictions
# Loss is computed by comparing predictions to actual values
# Backpropagation calculates gradients
# Weights and biases are updated proportionally to their contribution to the error

# The process continues until we complete all epochs or reach convergence.


# Training the Network with Gradient Descent
learning_rate = .9
num_epochs = 250
m_train = X_train_scaled.shape[0]
loss_history = []  # to store loss at each epoch

for epoch in range(num_epochs):
    # Forward pass
    A0 = X_train_scaled.T  # shape: (input_dim, m)
    Z1 = np.dot(W1, A0) + b1  # shape: (hidden_dim, m)
    A1 = relu(Z1)           # shape: (hidden_dim, m)
    Z2 = np.dot(W2, A1) + b2  # shape: (output_dim, m)
    A2 = softmax(Z2)        # shape: (output_dim, m)
    
    loss = compute_loss(A2, Y_train.T)
    loss_history.append(loss)
    
    # Backward pass
    dZ2 = A2 - Y_train.T                      # (output_dim, m)
    dW2 = (1/m_train) * np.dot(dZ2, A1.T)
    db2 = (1/m_train) * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m_train) * np.dot(dZ1, A0.T)
    db1 = (1/m_train) * np.sum(dZ1, axis=1, keepdims=True)
    
    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 10 == 0:  # Changed to print more frequently
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluation on Test Data
A0_test = X_test_scaled.T
Z1_test = np.dot(W1, A0_test) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(W2, A1_test) + b2
A2_test = softmax(Z2_test)
predictions = np.argmax(A2_test, axis=0)
true_labels = np.argmax(Y_test, axis=1)
accuracy = np.mean(predictions == true_labels)
print("Test Accuracy:", accuracy)


# ## Feature Importance Analysis
# 
# Now that we have trained our model, we can analyze which features contribute most significantly to the classification decision. The weights connecting the input layer to the hidden layer provide insights into which features are most important for each neuron's activation.
# 
# Larger absolute weight values indicate stronger influence of that feature on the neuron's output. By examining these weights, we can understand how the model "perceives" the relationship between culmen measurements, island location, and penguin species.
# 
# The following analysis shows the weights for each neuron in the first hidden layer, revealing which features each neuron specializes in detecting.

# In[ ]:


# Analyze feature importance
feature_names = numeric_features + list(X_cat.columns)
print("\nFeature Importance Analysis:")
# For each neuron in the first layer
for i in range(hidden_dim):
    print(f"\nNeuron {i+1} weights:")
    for j, feature in enumerate(feature_names):
        print(f"{feature}: {W1[i,j]:.4f}")


# # Selecting Perceptron Hyperparameters
# The complications for the perceptron model tuning arise from its various hyperparameters and their interconnectedness. For this reason I ran multiple combinations of epochs and learning rate while keeping other hyperparameters within margins found on online forums and websites.
# 
# After changing these hyperparameters I then created a training loss function graph to visualize how different combinations affected the loss curve's concavity and convergence patterns. This visualization approach proved crucial, allowing me to directly observe how parameter adjustments impacted model performance over training time.
# 
# The analysis led to two key conclusions: I wanted to keep the epochs to a minimum as in accordance with diminishing returns, and the nature of the dataset allows a very high training rate to be warranted, very close to 1. These together allowed me to pick the hyperparameters you see above.

# In[ ]:


# Plot Training Loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()


# ## Visualizing Model Performance
# 
# 1. Confusion Matrix: Reveals which species are correctly classified and which are confused with each other, providing insight into the model's strengths and weaknesses.
# 
# 2. Prediction Accuracy Visualization: A direct comparison of true vs. predicted labels for test samples, highlighting any patterns in misclassifications.
# 
# These visualizations help us assess whether the model has learned meaningful patterns or if it's making systematic errors that might indicate a need for model adjustments.

# In[ ]:


# Plot Confusion Matrix for Test Data
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix on Test Data")
plt.show()


# Okay that looks 100% accurate based on the confusion matrix which is as good as it gets. Let's do a double take and visualize the accuracy a second way to make sure.

# In[ ]:


# Create a scatter plot of true vs predicted labels for test samples
plt.figure(figsize=(10, 5))
plt.scatter(range(len(true_labels)), true_labels, 
            color='blue', marker='o', label='True Labels')
plt.scatter(range(len(predictions)), predictions, 
            color='red', marker='x', label='Predicted Labels')
plt.xlabel("Test Sample Index")
plt.ylabel("Class Label")
plt.title("True vs Predicted Class Labels")
plt.legend()
plt.show()


# Yes, that appears 100% accurate, so indeed the additional complexity for the perceptron model outperforms our previous two simpler ones.
# 
# We'll now plot the decision regions to see how it differs from the previous two; based on how perceptron works the regions should not be delineated by lines (as they were in LDA and logistic regression), which should help if it has fit correctly (which, judging by the accuracy, it probably has).

# ## Decision Boundary Visualization
# 
# The most insightful visualization for our perceptron model is the decision boundary plot. Since we're using only two numeric features (Culmen Length and Depth), we can create a 2D visualization showing how the model separates the three penguin species.
# 
# For each island, we generate a separate decision boundary to see how geographical location influences classification decisions. This visualization:
# 
# 1. Shows the non-linear decision boundaries learned by our neural network
# 2. Reveals how data clusters by species and island
# 3. Highlights areas where misclassifications are more likely to occur
# 4. Demonstrates how island location shifts the decision boundaries
# 
# The contour lines represent boundaries between different species classifications, with training data points overlaid to show the actual distribution of measurements.

# In[ ]:


#Plot decision boundaries
if len(numeric_features) == 2:  # We can visualize 2D decision boundaries
    plt.figure(figsize=(10, 8))
    
    # Create a mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # For each island, plot the decision boundary
    island_names = list(X_cat.columns)
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    # Create empty scatter plots for legend
    species_handles = []
    for s, species_name in enumerate(le.classes_):
        species_handles.append(plt.scatter([], [], c=colors[s], marker=markers[s], label=species_name))
    
    island_handles = []
    for i, island in enumerate(island_names):
        island_handles.append(plt.plot([], [], colors[i], linestyle='-', label=f"{island} boundary")[0])
    
    # Plot decision boundaries for each island
    for i, island in enumerate(island_names):
        # Create input with current island one-hot encoded
        island_vector = np.zeros(len(island_names))
        island_vector[i] = 1
        
        # Create input grid
        grid = np.c_[xx.ravel(), yy.ravel(), 
                    np.tile(island_vector, (xx.ravel().shape[0], 1))]
        
        # Scale  grid
        grid_scaled = (grid - X_min) / (X_max - X_min + 1e-8)
        
        # Forward pass
        A0_grid = grid_scaled.T
        Z1_grid = np.dot(W1, A0_grid) + b1
        A1_grid = relu(Z1_grid)
        Z2_grid = np.dot(W2, A1_grid) + b2
        A2_grid = softmax(Z2_grid)
        Z_grid = np.argmax(A2_grid, axis=0)
        
        # Reshape result back to grid
        Z_grid = Z_grid.reshape(xx.shape)
        
        # Plot decision boundary for this island without adding to legend
        plt.contour(xx, yy, Z_grid, colors=colors[i], alpha=0.4, 
                   levels=[0.5, 1.5, 2.5], linestyles=['--', '-', '--'])
    
    # Plot data points for each island and species (without adding to legend)
    for i, island in enumerate(island_names):
        mask = X_train[:, 2 + i] == 1
        
        # For each species (0, 1, 2), plot points on this island
        for species in range(3):
            species_mask = np.argmax(Y_train, axis=1) == species
            combined_mask = mask & species_mask
            plt.scatter(X_train[combined_mask, 0], 
                       X_train[combined_mask, 1],
                       c=colors[species], 
                       marker=markers[species])
    
    plt.xlabel(numeric_features[0])
    plt.ylabel(numeric_features[1])
    plt.title('Decision Boundaries by Island and Species')
    
    # Create two separate legends and position them strategically
    plt.legend(handles=species_handles, title="Species", loc="upper right") 
    plt.legend(handles=island_handles, title="Island Boundaries", loc="lower right")
    
    # This trick creates two separate legends
    first_legend = plt.legend(handles=species_handles, title="Species", loc="upper right")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=island_handles, title="Island Boundaries", loc="lower right")
    
    plt.show()


# ## Discussion

# # Logistic regression analysis
# 
# Overall, the models we chose perform pretty well. Based on our decision region plots, however, there are ambiguous points;
# For example, the logistic regression models the decision regions as specific "arcs" of space, which, while close, does not perfectly capture the real distribution of features for each species. For example, on some of the decision region plots, there are points corresponding to penguins of different species that are right next to each other, indicating they are extremely similar in those features despite not being the same species. The model is not able to differentiate these. Perhaps adding more features to provide some more information to distinguish between these ambiguous points would be helpful. 
# 
# 

# # Perceptron Model Analysis with Culmen and Island Features
# 
# The perceptron is a neural network. We gave it three features, which is a restriction given the data; we were challenged to find a model that could beat the 97-98% accuracy regime of LDA and logistic regression. We thought that implementing a more complicated model might be able to succeed at this even with the feature limitation, and in short, that ended up being correct.
# 
# Here is some additional explanation due to the complexity of this model.
# 
# Data preprocessing steps included:
# 1. Feature selection (Culmen Length, Culmen Depth, Island)
# 2. Removing samples with missing values
# 3. One-hot encoding of the Island feature
# 4. Min-max scaling to normalize numerical features
# 5. Label encoding the target variable (penguin species)
# 
# ## Model Architecture
# 
# I implemented a simple neural network with the following architecture:
# - Input layer: Dimensions match our feature set (2 numerical + one-hot encoded island features)
# - Hidden layer: 9 neurons with ReLU activation
# - Output layer: 3 neurons (one per species) with softmax activation
# 
# This architecture represents an enhanced perceptron model that can learn non-linear relationships, which was probably important in its ability to outcompete the more limited LDA and logistic regression models.
# 
# ## Training Process
# 
# The model was trained using:
# - Cross-entropy loss function
# - Gradient descent optimization with a learning rate of 0.9
# - 250 training epochs
# 
# The loss consistently decreased during training, indicating effective learning of the patterns in the data. The relatively small network architecture prevented overfitting while capturing the essential relationships between features and species.
# 
# ## Results and Performance
# 
# The model achieved approximately 99% accuracy on the test set, which is impressive given limited features.
# 
# ## Feature Importance Analysis
# 
# Analysis of the trained weights reveals that:
# 1. Culmen Length appears to be the most discriminative feature for species classification
# 2. The Biscoe Island feature correlates strongly with Gentoo penguins
# 3. Culmen Depth provides important supplementary information, especially for distinguishing Adelie from Chinstrap penguins
# 
# ## Conclusion
# 
# The perceptron model can account for relationships in the data that some of the other models cannot, giving it increased accuracy.

# # LDA Analysis
# 
# Based on our numeric and categorical features, our LDA model has exceptionally high performance. Averaging 99% accuracy across 10 folds is very promising.
# 
# ## Areas for improvment
# 
# Given the restrictions and limited data given for the assignment, there is little area for improvment. However if we were to further work on making this model suitable for real world use we would do the following:
# 
# - Increase the number of features
# 	- Potentially adding other promising feature sets previously mentioned such as flipper length or body mass
# - Work with a larger data set
# 	- The dataset we are currently working with is rather limited and the model would be more accurate if we could train it on more data
# - Analyze more complex models
# 	- While with this dataset a linear model is sufficient, it is more likely that with a larger dataset, a more complex relationship would necessitate a more complex model. Some considerations are some done in this project like the Perceptron model or a QDA
# 
# 

# # Conclusion
# 
# Overall, it seems that the best features are Culmen Length, Culmen Depth, and Island (given the restriction of one categorical and two numeric features), and the best model is perceptron. This indicates that the classification challenge when restricted to one categorical and two numeric features is easily possible with the tools provided in sklearn; some simpler models such as logistic regression and LDA just meet the 97% threshold and are easy to work with. To achieve a model with much improved performance, some more advanced setup is necessary, but the perceptron neural network can obtain near-100% accuracy even with this limited data.
