import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the model
#lr->Linear Regression 
#rf->Random Forest

pipe_lr = pickle.load(open('pipe_lr.pkl', 'rb'))
pipe_rf = pickle.load(open('pipe_rf.pkl', 'rb'))
df = pickle.load(open('df_lr.pkl', 'rb'))
mydata = pd.read_csv("laptop_data.csv")
# for col in mydata.columns:
#     st.text(col)
# st.text(mydata['Unnamed: 0'])
mydata = mydata.drop('Unnamed: 0', axis = 1)

# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a shipping method",
#         ("Standard (5-15 days)", "Express (2-5 days)")
#     )
st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())


# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])
# st.line_chart(chart_data)

# sns.barplot(x=df['Company'],y=df['Price'])
# plt.xticks(rotation='vertical')
# plt.show()

# fig = plt.figure.Figure(figsize=(10, 4))
# # sns.lineplot(x = "Company", y = "Price", data = df)
# st.text(mydata['Company'])
# # fig = sns.pairplot(mydata, hue="Company")
# # fig = sns.countplot(x = "Company", data = mydata)
# fig = sns.barplot(x=mydata.Company,y=dups_cmpny)
# st.pyplot(fig)

dups_cmpny = df.pivot_table(columns=['Company'], aggfunc='size')
dups_cat = df.pivot_table(columns=['TypeName'], aggfunc='size')
# st.text(dups_cmpny)
def linePlot2():
    st.title("Laptops of Similar Configurations Vs Prices - Line Plot")
    # st.text(df)
    newdf = ((df.loc[(df['Cpu brand']==cpu) & (df['Gpu brand']==gpu) & (df['Ram']==ram) & (df['TypeName']==type),
                    ['Company','TypeName','Price']]))
    # st.text(newdf)
    newdf['Products'] = "P" + (newdf.reset_index().index+1).astype(str)
    # st.text(newdf)
    if(newdf.empty):
        st.text("There Are No Products Which Have The Similar Configuration")
    else:        
        fig = plt.figure(figsize=(35, 15))
        sns.lineplot(x = "Products", y = "Price", data = newdf)
        st.pyplot(fig)

def linePlot():
    st.title("Laptops of Selected Company Vs Prices - Line Plot")
    grouped = df.groupby(mydata['Company'])
    # st.text(grouped)
    newdata = grouped.get_group(company) 
    # st.text(newdata)
    newdata[company] = "P" + (newdata.reset_index().index+1).astype(str)
    # st.text(newdata)
    fig = plt.figure(figsize=(40, 12))
    sns.lineplot(x = company, y = "Price", data = newdata)
    st.pyplot(fig)

# def linePlot():
#     st.title("Laptops of Selected Company Vs Prices - Line Plot")
#     grouped = df.groupby(mydata['Company'])
#     newdata = grouped.get_group(company) 
#     # st.text(newdata)
#     newdata[company] = "P" + (newdata.reset_index().index+1).astype(str)
#     # st.text(newdata)
#     fig = plt.figure(figsize=(40, 12))
#     sns.lineplot(x = company, y = "Price", data = newdata)
#     st.pyplot(fig)

def scatter_plot():
    st.title("Laptop Companies Vs Number Of Laptops - Scatter Plot")
    #Create numpy array for the visualisation
    x = np.array(mydata['Company'].unique())   
    y = np.array(dups_cmpny)
    # st.text(x.size) 
    # st.text(y.size) 
    fig = plt.figure(figsize=(20, 10))
    plt.yticks(np.arange(0, 400, 20))
    plt.scatter(x, y)
    st.balloons()
    st.pyplot(fig)

def pieChart():
    st.title("Type Of Laptops Vs Quantity - Pie Chart")
    mylabels = np.array(mydata['TypeName'].unique())   
    x = np.array(dups_cat)

    fig = plt.figure(figsize=(25, 10))
    plt.pie(x, labels = mylabels)

    st.balloons()
    st.pyplot(fig)     

def bar_chart():
    st.title("Laptop Companies Vs Prices - Bar Graph")
    company = np.array(mydata['Company'])
    prices = np.array(mydata['Price'])
    # st.text(company)
    # st.text(prices)
    fig = plt.figure(figsize = (20, 10))
    plt.yticks(np.arange(0, 400000, 40000))
    plt.bar(company, prices)
    plt.xlabel("Laptop Companies")
    plt.ylabel("Laptop Prices")
    plt.title("Laptop Company VS Price")
    st.pyplot(fig)  

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen,
                     ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title("Using Linear Regression")
    st.text("The predicted price of this configuration is " +
             str(int(np.exp(pipe_lr.predict(query)[0]))))

    st.title("Using Random Forest")
    st.text("The predicted price of this configuration is " +
             str(int(np.exp(pipe_rf.predict(query)[0]))))

    predicted_price = int(np.exp(pipe_lr.predict(query)[0]))

    linePlot2()   
    linePlot()
    # scatter_plot()
    pieChart()
    bar_chart()    
