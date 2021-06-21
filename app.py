import seaborn as sns
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas_profiling as pp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

st.dataframe(data)
gender = pd.get_dummies(data['Gender'], drop_first=True)
no_show = pd.get_dummies(data['No-show'], drop_first=True)

data.drop(['Gender', 'No-show'], axis=1, inplace=True)
data = pd.concat([data, gender, no_show], axis=1)
st_profile_report(data.profile_report())
st.dataframe(data.shape)
st.dataframe(data.info())
st.dataframe(data.describe())
data.drop_duplicates(inplace=True)

st.dataframe(data.corr())
columns = data.columns
data1 = data.copy()
data = data.mask(np.random.random(data.shape) < .1)


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

st.subheader('Overview of the data:')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
#
# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
#
# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)

data = data.dropna()
x = data.drop(['Yes', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1)
y = data['Yes']

lr = LogisticRegression(random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = lr.fit(x_train, y_train)

st.text(round(accuracy_score(y_train, model.predict(x_train))*100, 2))
