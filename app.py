import seaborn as sns
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas_profiling as pp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import missingno as msno
import matplotlib.pyplot as plt
from streamlit_pandas_profiling import st_profile_report
from sklearn import tree


st.title('Mental Health in the Workplace')


@st.cache
def load_data():
    data = pd.read_csv('mental_health.csv')
    return data


data_load_state = st.text('Loading data...')
df = load_data()
data = df.copy().copy()
data_load_state.text("Data Loaded")
st_profile_report(data.profile_report())
st.text('Data before cleaning')
st.dataframe(data)
data['gender'] = data['gender'].replace({'F': 'f',
                                         'Female': 'f',
                                         'female': 'f',
                                         'M': 'm',
                                         'Male': 'm',
                                         'male': 'm',
                                         'Androgyne': 'o',
                                         'Genderqueer': 'o',
                                         'Trans woman': 'o',
                                         'Agender': 'o'
                                         })
data['work_interfere'] = data['work_interfere'].replace(np.nan, 'unknown')
gender = pd.get_dummies(data['gender'])
family_history = pd.get_dummies(data['family_history'], drop_first=True)
treatment = pd.get_dummies(data['treatment'], drop_first=True)
work_interfere = pd.get_dummies(data['work_interfere'])
care_options = pd.get_dummies(data['care_options'])
wellness_program = pd.get_dummies(data['wellness_program'])
seek_help = pd.get_dummies(data['seek_help'])
leave = pd.get_dummies(data['leave'])
mental_vs_physical = pd.get_dummies(data['mental_vs_physical'])

care_options.columns = ['care_options_no', 'care_options_not_sure', 'care_options_yes']
family_history.columns = ['family_history']
treatment.columns = ['treatment']
leave.columns = ['leave_dont_know', 'leave_somewhat_difficult', 'leave_somewhat_easy', 'leave_very_difficult',
                 'leave_very_easy']
mental_vs_physical.columns = ['mental_vs_physical_dont_know', 'mental_vs_physical_no', 'mental_vs_physical_yes']
seek_help.columns = ['seek_help_dont_know', 'seek_help_no', 'seek_help_yes']
wellness_program.columns = ['wellness_program_dont_know', 'wellness_program_no', 'wellness_program_yes']
work_interfere.columns = ['work_interfere_never', 'work_interfere_often', 'work_interfere_rarely',
                          'work_interfere_sometimes', 'work_interfere_unknown']

data['age'][data['age'] < 0] = np.nan
data = pd.concat([data['age'], gender, family_history, treatment, work_interfere, care_options, wellness_program,
                  seek_help, leave, mental_vs_physical], axis=1)

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

st.subheader('Split between treatment and not treatment')
df_treatment = data[data['treatment'] == 1]
df_no_treatment = data[data['treatment'] == 0]
st.text('People who sought treatment')
st.dataframe(df_treatment.describe())
st.text('People who did not seek treatment')
st.dataframe(df_no_treatment.describe())

st.subheader('Model Fitting')
data = data.dropna()
x = data.drop(['treatment'], axis=1)
y = data['treatment']
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
coef_importance_bar = pd.DataFrame({'Features': x.columns, 'coefficients': importance})
coef_importance_bar.set_index('Features', inplace=True)
coef_importance_bar.plot.bar()

st.pyplot(fig=plt)
plt.clf()

st.subheader('Random Forest Classifier')
y_rf_pred = clf_best.predict(x_test)
st.text(f'Random Forest Classification accuracy score: {round(accuracy_score(y_test, y_rf_pred), 2)}')

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
