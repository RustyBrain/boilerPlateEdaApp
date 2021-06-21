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



st.title('Uber pickups in NYC')

DATE_COLUMNS = ['scheduledday', 'appointmentday']
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data():
    data = pd.read_csv('dr_appointment_no_shows.csv')
    return data


data_load_state = st.text('Loading data...')
df = load_data()
data = df.copy().copy()
data_load_state.text("Done! (using st.cache)")
st_profile_report(data.profile_report())

st.dataframe(data)
gender = pd.get_dummies(data['Gender'], drop_first=True)
no_show = pd.get_dummies(data['No-show'], drop_first=True)

data.drop(['Gender', 'No-show'], axis=1, inplace=True)
data = pd.concat([data, gender, no_show], axis=1)
# st.dataframe(data.shape)
# st.dataframe(data.info())
# st.dataframe(data.describe())
# data.drop_duplicates(inplace=True)

st.dataframe(data.corr())
columns = data.columns
data1 = data.copy()
# data = data.mask(np.random.random(data.shape) < .1)


if np.count_nonzero(data.isnull().values) > 0:
    fig, ax = plt.subplots(figsize=(10, 10))
    msno.heatmap(data, ax=ax)
    st.pyplot(fig)
    st.text('Imputing missing data')
    imputer = IterativeImputer(random_state=42, skip_complete=True)


# numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
# std_scale = StandardScaler()
# data[numeric_cols] = std_scale.fit_transform(data[numeric_cols])
# st.dataframe(data)
# enc = OneHotEncoder()
# st.text(data.columns[~data.columns.isin(numeric_cols)].to_list())
#
# data = pd.get_dummies(data, columns=['base'])
# st.dataframe(data)
st.text(f'Missing data: {np.count_nonzero(data.isnull().values)}')

# st.subheader('Overview of the data:')
#
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)
# #
# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
#
# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)

st.subheader('Split')
df_missed = data[data['Yes'] == 1]
df_attended = data[data['Yes'] == 0]
st.dataframe(df_attended.describe())
st.dataframe(df_missed.describe())



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

st.text(f'Logistic regression accuracy score: {round(accuracy_score(y_test, model.predict(x_test)) * 100, 2)}')
st.text(f'CV Logistic regression accuracy score: {round(accuracy_score(y_test, clf.predict(x_test)), 2)}')

importance = model.coef_[0]
coef_importance_bar = pd.DataFrame({'Features': x.columns, 'coefficients': importance})
coef_importance_bar.set_index('Features', inplace=True)
coef_importance_bar.plot.bar()

st.pyplot(fig=plt)
plt.clf()

y_rf_pred = clf_best.predict(x_test)
st.text(f'Random Forest Classification accuracy score: {round(accuracy_score(y_test, y_rf_pred), 2)}')


importances = clf_best.feature_importances_
fig, ax = plt.subplots()
forest_importances = pd.Series(importances, index=x.columns)
std = np.std([tree.feature_importances_ for tree in clf_best.estimators_], axis=0)
forest_importances.plot.bar(yerr=std, ax=ax)
st.pyplot(fig=plt)
plt.clf()
