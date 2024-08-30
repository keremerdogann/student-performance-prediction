import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import StandardScaler



df=pd.read_csv("student_performance_prediction.csv")



df.drop(columns=["Student ID"],inplace=True)

#We dont need this column to predict success of student.

#Ogrencinin basarisini tahmin etmek icin bu sutuna ihtiyacimiz yok

#Eksik verileri ortalama ve diger istatiksel verilerle doldurma


print(df.isnull().sum())


df["Study Hours per Week"]=df["Study Hours per Week"].fillna(df["Study Hours per Week"].mean())

df.dropna(subset=["Passed"],inplace=True)

df["Attendance Rate"]=df["Attendance Rate"].fillna(df["Attendance Rate"].mean())

df["Previous Grades"]=df["Previous Grades"].fillna(df["Previous Grades"].mean())

df["Participation in Extracurricular Activities"]=df["Participation in Extracurricular Activities"].fillna(method="ffill")


label_encoder = LabelEncoder()


# Egitim seviyesini 0,1,2 olacak sekilde ayirma
df["Parent Education Level"]=label_encoder.fit_transform(df["Parent Education Level"])

mean_of_education_level = (int)(df["Parent Education Level"].mean())

df["Parent Education Level"]=df["Parent Education Level"].fillna(mean_of_education_level)


print(df.isnull().sum())

print(df.columns)


#Aktivitelere katilimi Yes ve No 0,1 olacak sekilde ayirma
df["Participation in Extracurricular Activities"]=label_encoder.fit_transform(df["Participation in Extracurricular Activities"])

#Geçti kaldi durumunu 0,1 olacak sekilde ayirma
df["Passed"]=label_encoder.fit_transform(df["Passed"])


y1=df["Passed"]

df.drop(columns=["Passed"],inplace=True)

x1=df

x_train , x_test , y_train , y_test = train_test_split(x1,y1,test_size=0.3,random_state=0)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


rl=LinearRegression()

rl.fit(x_train,y_train)

y_pred = rl.predict(x_test)

print(y_pred)
print(y_test)

results = pd.DataFrame({
    'Gerçek Değerler': y_test,
    'Tahmin Edilen Değerler': y_pred
})


plt.figure(figsize=(14, 7))
results = results.head(20)
results.reset_index(drop=True, inplace=True)
results.plot(kind='bar', figsize=(14, 7), width=0.8, alpha=0.75)
plt.title('Gerçek ve Tahmin Edilen Değerler')
plt.xlabel('Gözlem')
plt.ylabel('Değer')
plt.xticks(ticks=np.arange(len(results)), labels=results.index, rotation=90)
plt.legend(["Gerçek Değerler", "Tahmin Edilen Değerler"])
plt.grid(axis='y')
plt.show()

print("Gerçek Değerler Aralığı:", y_test.min(), y_test.max())
print("Tahmin Edilen Değerler Aralığı:", y_pred.min(), y_pred.max())





