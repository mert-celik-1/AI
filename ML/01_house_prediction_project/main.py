import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

housing = fetch_california_housing()

df = pd.DataFrame(housing.data,columns=housing.feature_names)

df['median_house_value'] = housing.target

print("Veri seti şekli:",df.shape)
print("\nKolon isimleri",df.columns.tolist())
print("\nVeri seti bilgisi",df.info())
print("\nIstatistiksel özet\n",df.describe())
print("\nEksik değerler\n",df.isnull().sum())

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Korelasyon matrisi')
plt.tight_layout()
#plt.show()


# Hedef değişken ile diğer özellikler arasındaki ilişki
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1]):  # Hedef değişken hariç
    plt.subplot(3, 3, i+1)
    plt.scatter(df[column], df['median_house_value'], alpha=0.5)
    plt.title(f'{column} vs Target')
    plt.xlabel(column)
    plt.ylabel('Median House Value')
plt.tight_layout()
#plt.show()


# Yeni anlamlı featureler oluşturma

# Yatak odası oranı (toplam oda sayısına göre)
df['bedroom_ratio'] = df['AveBedrms'] / df['AveRooms']

# Gelir/oda oranı (zengin mahallelerde kişi başına daha fazla oda olabilir)
df['income_per_rooms'] = df['MedInc'] / df['AveRooms']

# Korelasyon matrisinde MedInc'in en etkili faktör olduğunu gördük
# Daha yüksek gelir seviyelerini vurgulamak için karesel bir terim ekleyebiliriz
df['MedInc_squared'] = df['MedInc'] ** 2


# Aykırı değerleri kontrol edelim (özellikle rooms ve bedrooms'da aykırı değerler gördük)
print("Oluşturulan özelliklerde potansiyel sonsuz veya NaN değerleri kontrol ediliyor...")
print(df.isnull().sum())
print("\nSonsuz değerler:")
print(np.isinf(df).sum())

# Eğer sonsuz değerler varsa bunları NaN ile değiştirelim ve sonra makul değerlerle dolduralım
df.replace([np.inf, -np.inf], np.nan, inplace=True)

if df.isnull().sum().sum() > 0 :
    df.fillna(df.mean(),inplace=True)
    print("Nan değerler ortalama ile dolduruldu")


correlation_with_target = df.corr()['median_house_value'].sort_values(ascending=False)
print("\nHedef değişken ile korelasyonlar:")
print(correlation_with_target)


X = df.drop('median_house_value' , axis=1)
y = df['median_house_value']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Eğitim seti boyutu : {X_train.shape}")
print(f"Test seti boyutu : {X_test.shape}")

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled,columns=X_test.columns)

print("\nÖlçeklendirilmiş eğitim verisi istatistikleri:")
print(X_train_scaled_df.describe().round(2))


def evaluate_model(model,X_train,X_test,y_train,y_test):

    # Model eğit
    model.fit(X_train,y_train)

    # Eğitim seti üzerinde tahmin
    y_train_pred = model.predict(X_train)

    # Test seti üzerinde tahmin
    y_test_pred = model.predict(X_test)


    # Metrikler
    train_mse = mean_squared_error(y_train,y_train_pred)
    test_mse = mean_squared_error(y_test,y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_mae = mean_absolute_error(y_train,y_train_pred)
    test_mae = mean_absolute_error(y_test,y_test_pred)

    train_r2 = r2_score(y_train,y_train_pred)
    test_r2 = r2_score(y_test,y_test_pred)

    cv_scores = cross_val_score(model,X_train,y_train,cv=5,scoring='r2')

    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }



# Lineer Regresyon
lr_model = LinearRegression()
lr_results = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_results = evaluate_model(dt_model, X_train_scaled, X_test_scaled, y_train, y_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
rf_results = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test)


models = {
    'Linear Regression' : lr_results,
    'Decision Tree' : dt_results,
    'Random Forest' : rf_results
}

results_df = pd.DataFrame({
    'Linear Regression': lr_results,
    'Decision Tree': dt_results,
    'Random Forest': rf_results
}).T


print("\nModel Performans Karşılaştırması:")
print(results_df.round(4))

# Grafik ile karşılaştırma
metrics = ['test_rmse', 'test_r2', 'cv_r2_mean']
plt.figure(figsize=(12, 8))

for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    plt.bar(results_df.index, results_df[metric])
    plt.title(f'Model Karşılaştırma - {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()






