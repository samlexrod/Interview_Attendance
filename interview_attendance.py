import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
import math

# DATA EXTRACTION
# ==============================================================

# Extracting data from CSV
df = pd.read_csv('interview.csv', sep=',', header=0)

# Encoding Labels to 1=yes and 0=no
target_name = 'Observed Attendance'
df[target_name] = np.where(df[target_name].str.lower()=='yes', 1, 0)

# Saving target in target data frame
target = df[target_name][:-1]                   # <- Saves the target only without the last row
df = df.drop(target_name, axis=1)               # <- Drops the target from the features
df = df[:-1]                                    # <- Saves the features without the last row
df = df.drop(list(df.columns)[-5:], axis=1)     # <- Drops the extra 5 columns in the CSV file



# DATA CLEANING
# ===============================================================
def show_unique(data_frame, feature_names):
    for feature_name in feature_names:
        print "\nShowing Unique Values for, {}".format(feature_name)
        print np.unique(data_frame[feature_name]), '\n'

def clean_time(data_frame, column):
    this = data_frame[column].str.lower()
    this_dict = this.to_dict()
    for k, v in this_dict.items():
        try:
            if type(int(v[0])) == int and \
                    (v.find('am') > 0 or v.find('pm') > 0):  # <- Is apparent time format
                this_dict[k] = [1, v]
            else:
                this_dict[k] = [0, 'ERROR----------------']  # <- Item not assessed
        except:
            this_dict[k] = [0, v]  # <- Normal item

    is_time = pd.DataFrame.from_dict(this_dict, orient='index')
    data_frame[column] = np.where(is_time[0], 'nan', is_time[1])

    return data_frame

# -> Cleaning Dates
clean = 'Date of Interview'
this = df[clean]
df[clean] = this.str.replace('.', '/')
df[clean] = this.str.replace('-', '/')

# -> Cleaning Location
clean = 'Location'
this = df[clean]
df[clean] = this.str.lower()
df[clean] = this.str.strip('- ')
df[clean] = np.where(this == 'gurgaonr', 'gurgaon', this)

# -> Cleaning Position to be closed
clean = 'Position to be closed'
this = df[clean]
df[clean] = this.str.replace('-', '')

# -> Clean Nature of Skillset
clean = 'Nature of Skillset'
df = clean_time(df, clean)
this = df[clean]
df[clean] = this.str.replace(' ', '')
df[clean] = this.str.strip("-")
df[clean] = this.str.replace('\\xe2\\x80\\x93', '/')
df[clean] = this.str.replace(',', '/')

# -> Clean Interview Type
clean = 'Interview Type'
this = df[clean].str.lower()
this = this.str.replace(' ', '')
df[clean] = np.where(this == 'sceduledwalkin', 'scheduledwalkin', this)

# -> Clean Candidate Current Location
clean = 'Candidate Current Location'
this = df[clean].str.lower()
df[clean] = this.str.strip('- ')

# -> Clean Candidate Job Location
clean = 'Candidate Job Location'
this = df[clean].str.lower()
df[clean] = this.str.strip('- ')

# -> Clean Interview Venue
clean = 'Interview Venue'
this = df[clean].str.lower()
df[clean] = this.str.strip('- ')

# -> Clean Candidate Native location
clean = 'Candidate Native location'
this = df[clean].str.lower()
this = this.str.replace('/ncr', '')
df[clean] = this.str.strip('- ')

# -> Clean Expected Attendance
clean = 'Expected Attendance'
df = clean_time(df, clean)

this = df[clean].str.lower()



# NAN ANALYSIS
# ===========================================================
def nan_table(data_frame, groupby_columns, agg_column):
    for num, column in enumerate(groupby_columns):
        data_frame[column] = df[column].astype(str)
        print "\nTable {}".format(num+1)
        output = data_frame.groupby(column)[agg_column].count()
        print output.sort_values(ascending=False)

def replace_with(data_frame, replace_this_list, with_this, on_columns):
    for replace in replace_this_list:
        for column in on_columns:
            this = data_frame[column]
            data_frame[column] = np.where(this == replace, with_this, this)
    return data_frame

def keep_else(data_frame, keep_this_list, else_this, on_columns):
    for column in on_columns:
        for keep in keep_this_list:
            this = data_frame[column]
            data_frame[column] = np.where(this == keep, this, else_this)
    return data_frame

# Columns with many missing values
columns = ['Have you obtained the necessary permission to start at the required time',
           'Can I Call you three hours before the interview and follow up on your attendance for the interview',
           'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much',
           'Have you taken a printout of your updated resume. Have you read the JD and understood the same',
           'Are you clear with the venue details and the landmark.',
           'Has the call letter been shared',
           'Hope there will be no unscheduled meetings']

# Changing data type
for column in columns:
    df[column] = df[column].str.lower()
    df[column] = df[column].astype(str)

# Replacing unwanted labels
df = replace_with(df, ['not sure', 'need to check', 'yet to check', 'havent checked'], 'uncertain', columns)

# Keeping wanted labels, replacing rest
for column in columns:
    this = df[column]
    df[column] = np.where(this == 'yes', this,
                          np.where(this == 'no', this,
                                   np.where(this == 'nan', this,
                                            np.where(this == 'uncertain', this,
                                                     'no'))))

# Showing count tables
nan_table(df, columns, 'Client name')
nan_table(df, ['Expected Attendance'], 'Client name')


# DROPPING IRRELEVANT FEATURES
# ===========================================================
# Goo practice cleaning, but this feature is useless since data does not clearly represent metadata
df = df.drop('Nature of Skillset', axis=1)

# The candidate ID will not provide predictive value
df = df.drop('Name(Cand ID)', axis=1)

# The dates are too messy and do not provide predictive value in a qualitative analysis
df = df.drop('Date of Interview', axis=1)

# CREATING NEW FEATURE
# ============================================================
interview = 'Interview Venue'
job = 'Candidate Job Location'
df['Interview in Different Location'] = np.where(df[job] == df[interview], 'same', 'different')


# HASHING QUALITATIVE VARIABLES
# ============================================================
def auto_encode(data_frame):
    le = LabelEncoder()

    for column in list(data_frame.columns):
        le.fit(np.unique(data_frame[column]))
        data_frame[column] = le.transform(data_frame[column].values)

    return data_frame


df = auto_encode(df)

# FEATURE DELETION
# ==============================================================
df = df.drop('Gender', axis=1)

# SEPARATING TARGET FROM FEATURES
# ==============================================================

# Handling extra columns and rows
features = df.values
feature_names = list(df.columns)

# Analyze Unique Values
show_unique(df, feature_names)

# print the feature_names
pprint(feature_names)
features = features
target = target

print "\nFeatures value preview:"
pprint(features[0])
print "\tRows: {}\n" \
      "\tColumns: {}".format(len(features), len(features[0]))

print "\nTarget labels preview:"
pprint(target[:10])
print "\tRows: {}\n" \
      "\tColumns: {}".format(len(target), 1)


# AUTO TUNING PARAMETERS
# ==============================================================



# CROSS VALIDATION
# ==============================================================
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, target, test_size=0.3, random_state=42)

print "\nSize of Train:\n" \
      "\tFeatures: {}\n" \
      "\tLabels: {}\n" \
      "Size of Test:\n" \
      "\tFeatures: {}\n" \
      "\tLabels: {}".format(len(features_train),
                            len(labels_train),
                            len(features_test),
                            len(labels_test))

features_train = features_train.astype('int')

dt = DecisionTreeClassifier()

dt.fit(features_train, labels_train)

accuracy = dt.score(features_test, labels_test)

pred = dt.predict(features_test)

print pred
print "Actual Attendance: {}".format(sum(labels_test))
print "Actual Non-attendance: {}".format(len(labels_test) - sum(labels_test))

print "Accuracy: {}".format(accuracy)
print classification_report(labels_test, pred)

from sklearn.model_selection import cross_val_score
print cross_val_score(dt, features_train, labels_train, cv=5)
print

# RANDOM FOREST
# ================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'criterion': ['gini', 'entropy'], 'min_samples_split': range(2, 50, 2),
                 'max_depth': [None, 1, 100],
                 'min_samples_leaf': range(1, 60, 2)}

rf = RandomForestClassifier()

rf = GridSearchCV(rf, parameters)

rf.fit(features_train, labels_train)

print rf.best_estimator_

pred = rf.predict(features_test)

accuracy = rf.score(features_test, labels_test)
print "Accuracy: {}".format(accuracy)
print(classification_report(labels_test, pred))
print cross_val_score(rf, features_train, labels_train, cv=5)













