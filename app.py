import seaborn as sns
import streamlit as st
import missingno as msno
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas_profiling as pp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from streamlit_pandas_profiling import st_profile_report
from sklearn import tree
from statsmodels.formula.api import ols

st.set_page_config(layout="wide")
st.title('Appointment No-Shows')


@st.cache
def load_data():
    data = pd.read_csv('dr_appointment_no_shows.csv')
    return data


@st.cache
def load_profile(data):
    st_profile_report(data.profile_report())
    return


data_load_state = st.text('Loading data...')
df = load_data()
data = df.copy().copy()
data_load_state.text("Data Loaded")
st_profile_report(data.profile_report())
st.text('Data before cleaning')
st.dataframe(data)
gender = pd.get_dummies(data['Gender'], drop_first=True)
no_show = pd.get_dummies(data['No-show'], drop_first=True)

data.drop(['Gender', 'No-show'], axis=1, inplace=True)
data = pd.concat([data, gender, no_show], axis=1)


st.text('Data after cleaning')
st.dataframe(data)

st.subheader('Correlations')
st.dataframe(data.corr())
fig, ax = plt.subplots()
sns.heatmap(data.corr(), ax=ax)
st.write(fig)
columns = data.columns


if np.count_nonzero(data.isnull().values) > 0:
    fig, ax = plt.subplots(figsize=(10, 10))
    msno.heatmap(data, ax=ax)
    st.pyplot(fig)
    st.text('Imputing missing data')
    imputer = IterativeImputer(random_state=42, skip_complete=True)


st.text(f'Missing data: {np.count_nonzero(data.isnull().values)}')

st.subheader('Split')
df_missed = data[data['Yes'] == 1]
df_attended = data[data['Yes'] == 0]
st.write('attended appointment')
st.dataframe(df_attended.describe())
st.write('missed appointment')
st.dataframe(df_missed.describe())

st.subheader('Model Fitting')
data = data.dropna()
x = data.drop(['Yes', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood', 'PatientId', 'AppointmentID'], axis=1)
y = data['Yes']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


@st.cache(allow_output_mutation=True)
def model_stuff(x_train, y_train):
    lr = LogisticRegression(random_state=42)
    model = lr.fit(x_train, y_train)
    clf = LogisticRegressionCV(cv=5, random_state=42, n_jobs=-1).fit(x_train, y_train)
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }

    CV_clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    CV_clf.fit(x_train, y_train)
    clf_best = CV_clf.best_estimator_
    return model, clf, clf_best


model, clf, clf_best = model_stuff(x_train, y_train)

st.subheader('Logistic Regressions')
st.text(f'Logistic regression accuracy score: {round(accuracy_score(y_test, model.predict(x_test)), 2)}')
st.text(f'CV Logistic regression accuracy score: {round(accuracy_score(y_test, clf.predict(x_test)), 2)}')

st.text('Importance of coefficients from Logistic Regression model')
importance = model.coef_[0]
cm = confusion_matrix(y_test, clf.predict(x_test))
st.write('Confusion matrix for CV Logistic Regression: ', cm)
coef_importance_bar = pd.DataFrame({'Features': x.columns, 'coefficients': importance})
coef_importance_bar.set_index('Features', inplace=True)
coef_importance_bar.plot.bar()

st.pyplot(fig=plt)
plt.clf()

st.subheader('Random Forest Classifier')
y_rf_pred = clf_best.predict(x_test)
st.text(f'Random Forest Classification accuracy score: {round(accuracy_score(y_test, y_rf_pred), 2)}')
cm = confusion_matrix(y_test, y_rf_pred)
st.write('Confusion matrix for Random Forest: ', cm)
st.text('Feature importance from tree ensamble')
importances = clf_best.feature_importances_
fig, ax = plt.subplots()
forest_importances = pd.Series(importances, index=x.columns)
std = np.std([tree.feature_importances_ for tree in clf_best.estimators_], axis=0)
forest_importances.plot.bar(yerr=std, ax=ax)
st.pyplot(fig=plt)
plt.clf()

st.text('Example tree plot from Random Forest Classifier')
fig, axes = plt.subplots(nrows=1, ncols=1)

tree.plot_tree(clf_best.estimators_[0],
               feature_names=x.columns,
               class_names='treatment',
               filled=True)

st.pyplot(fig=plt)
plt.clf()

dat_age_sms = data[['Age', 'SMS_received', 'Yes']]

rf = RandomForestClassifier(random_state=42)
x_train, x_test, y_train, y_test = train_test_split(dat_age_sms[['Age', 'SMS_received']], dat_age_sms['Yes'], test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

CV_clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
CV_clf.fit(x_train, y_train)
clf_best = CV_clf.best_estimator_

y_rf_pred = clf_best.predict(x_test)
st.text(f'Random Forest Classification accuracy score: {round(accuracy_score(y_test, y_rf_pred), 2)}')
cm = confusion_matrix(y_test, y_rf_pred)
st.write('Confusion matrix for Random Forest: ', cm)