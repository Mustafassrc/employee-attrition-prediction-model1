import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,classification_report
from sklearn.model_selection import GridSearchCV, cross_validate,KFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

pd.set_option("display.max_columns",None)
pd.set_option("display.width",None)
pd.set_option("display.max_rows",20)
pd.set_option("display.float_format",lambda x: "%.3f" % x)

df = pd.read_csv("Datasets/isciverisi.csv")
df.head()
#print(df)

def kontrol_df(dataframe,head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    numeric_df = dataframe.select_dtypes(include=['number'])
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

#kontrol_df(df)

def yakala_sutun_tipini(dataframe, cat_th=10,car_th=20):
    #Katagorik ve kardinaller sütunlar
    kategorik_sutunlar=[col for col in dataframe.columns if dataframe[col].dtypes=="O"]
    numerik_ama_kategorik=[ col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes!="0"]
    kategorik_ama_kardinal=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes=="O"]
    kategorik_sutunlar=kategorik_sutunlar+numerik_ama_kategorik
    kategorik_sutunlar=[col for col in kategorik_sutunlar if col not in kategorik_ama_kardinal]

    #Numerik sütunlar
    numerik_sutunlar=[col for col in dataframe.columns if dataframe[col].dtypes!="O"]
    numerik_sutunlar=[col for col in numerik_sutunlar if col not in numerik_ama_kategorik]

    print(f"Observations:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f"Kategorik Sütunlar:{len(kategorik_sutunlar)}")
    print(f":Numerik Sütunlar {len(numerik_sutunlar)}")
    print(f'Kategorik ama Kardinal: {len(kategorik_ama_kardinal)}')
    print(f'Numerik ama Kategorik: {len(numerik_ama_kategorik)}')

    return kategorik_sutunlar, numerik_sutunlar, kategorik_ama_kardinal

kategorik_sutun,numerik_sutun,kategorik_ama_kardinal=yakala_sutun_tipini(df)
#print("Kategorik sütun isimleri:",kategorik_sutun)
#print("numerik sütun:",numerik_sutun)
#print("kategorik ama kardinal isimleri:",kategorik_ama_kardinal)
print("########################################")
def kategorik_ozet(dataframe,kolon_ismi,plot=False):
    print(pd.DataFrame({kolon_ismi: dataframe[kolon_ismi].value_counts(),
                        "Ratio":100*dataframe[kolon_ismi].value_counts()/len(dataframe)}))
    print("############################################")
    if plot:
        sns.countplot(x=dataframe[kolon_ismi],data=dataframe)
        plt.show()
#kategorik_ozet(df,"BusinessTravel")


def numerik_ozet(dataframe,kolon_ismi,plot=False):
    quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[kolon_ismi].describe(quantiles).T)

    if plot:
        dataframe[kolon_ismi].hist(bins=20)
        plt.xlabel(kolon_ismi)
        plt.title(kolon_ismi)
        plt.show()

#for col in numerik_sutun:
    #numerik_ozet(df,col)

def hedef_degisken_icin_ozet_numeriklerle(dataframe,target,kolon_ismi):
    print(dataframe.groupby(target).agg({kolon_ismi:"mean"}),end="\n\n\n")
#for col in numerik_sutun:
    hedef_degisken_icin_ozet_numeriklerle(df,"TotalWorkingYears",col)

#Koralasyon Isı Haritası

le=LabelEncoder()
for col in kategorik_sutun:
    df[col]=le.fit_transform(df[col])

f,ax=plt.subplots(figsize=(18,13))
sns.heatmap(df.corr(),annot=True,fmt=".2f",ax=ax,cmap="magma")
ax.set_title("Correlation Matrix",fontsize=20)
#plt.show()

#Basit Model Kurulumu
y=df["Attrition"]
X=df.drop("Attrition",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


rf_model = RandomForestClassifier()
rf_model=rf_model.fit(X_train,y_train)
y_pred=rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

def onem_dereceleri(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()

#onem_dereceleri(rf_model, X)

#ÖZELLİK MÜHENDİSLİĞİ

#->Eksik Değer Analizi
zero_columns=[col for col in df.columns if(df[col].min()==0and col not in ["Attrition"])]
print(zero_columns)

for col in zero_columns:
    df[col]=np.where(df[col]==0,np.nan,df[col])
print(df.isnull().sum())

def eksik_degerler_tablosu(dataframe,na_name=False):
    na_sutunlar=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_eksikler=dataframe[na_sutunlar].isnull().sum().sort_values(ascending=False)
    oran=(dataframe[na_sutunlar].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    kayip_df=pd.concat([n_eksikler, np.round(oran,2)],axis=1,keys=["n_eksikler","oran"])
    print(kayip_df.T,end="\n")
    if na_name:
        return na_sutunlar

na_sutunlar=eksik_degerler_tablosu(df,na_name=True)

def cok_eksik_varsa_sil(dataframe,th=0.5):
    missing_ratio=df.isnull().mean()
    cols_to_drop=missing_ratio[missing_ratio>th].index.tolist()
    return df.drop(columns=cols_to_drop)

df=cok_eksik_varsa_sil(df)

print(df.head())
#Aşağıdaki fonksiyonu çok beğendim !!
def eksik_hedef_deg_analizi(dataframe,target,na_columns):
    temp_df=dataframe.copy()
    for col in na_columns:
        temp_df[col+'_NA_FLAG']=np.where(temp_df[col].isnull(),1,0)
    na_flags=temp_df.loc[:,temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                           "Count":temp_df.groupby(col)[target].count()}),end="\n\n\n")

#eksik_hedef_deg_analizi(df,"Attrition",na_sutunlar)

for col in df.columns:  # Sadece mevcut sütunlar için işlemi uygula
    if df[col].isnull().sum() > 0:
        df.loc[df[col].isnull(), col] = df[col].median()


df.isnull().sum()

#Aykırı Değer Analizi
def outlier_thresholds(dataframe,kolon_ismi,q1=0.05,q3=0.95):
    quartile1=dataframe[kolon_ismi].quantile(q1)
    quartile3 = dataframe[kolon_ismi].quantile(q3)
    interquantile_range=quartile3-quartile1
    uplimit=quartile3+1.5*interquantile_range
    lowlimit=quartile3-1.5*interquantile_range
    return lowlimit,uplimit
def check_outlier(dataframe,kolon_ismi):
    lowlimit,uplimit=outlier_thresholds(dataframe,kolon_ismi)
    if dataframe[(dataframe[kolon_ismi]>uplimit)|(dataframe[kolon_ismi]<lowlimit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe,variable,q1=0.05,q3=0.95):
    lowlimit,uplimit=outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable]<lowlimit),variable]=lowlimit
    dataframe.loc[(dataframe[variable]>uplimit),variable]=uplimit

#for col in df.columns:
    print(col,check_outlier(df,col))
    if check_outlier(df,col):
        replace_with_thresholds(df,col)

#for col in df.columns:
    print(col,check_outlier(df,col))


# ÖZELLİK ÇIKARIMI KISMINI ATLIYORUM!!!!!!!


#ENCODİNG

#Değişkenlerin tiplerine göre ayrılması işlemi

kategorik_sutun,numerik_sutun,kategorik_ama_kardinal=yakala_sutun_tipini(df)

def label_encoder(dataframe,binary_col):
    labelencoder=LabelEncoder()
    dataframe[binary_col]=labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols=[col for col in df.columns if df[col].dtypes =="O" and df[col].nunique()==2]

for col in binary_cols:
    df=label_encoder(df,col)

#One-Hot Encoding işlemi
#Kategorik Sütünların Güncellenmesi

kategorik_sutun=[col for col in kategorik_sutun if col not in binary_cols and col not in["Attrition"]]
print(kategorik_sutun)
def one_hot_encoder(dataframe,categorical_cols, drop_first=False):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
df=one_hot_encoder(df,kategorik_sutun,drop_first=True)





#Standartlaştırma

scaler=StandardScaler()
df[numerik_sutun]=scaler.fit_transform(df[numerik_sutun])
print(df.head())
print(df.shape)



y = df["Attrition"]
X = df.drop("Attrition", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=10)

rf_model = RandomForestClassifier(random_state=10).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")




model = RandomForestClassifier(random_state=10)
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation Scores: ", scores)
print("Mean Accuracy: ", scores.mean())

param_grid = {
    'n_estimators': [200],
    'max_depth': [70],
    'min_samples_split': [4]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-validation Score: ", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)

y_pred_best = best_rf_model.predict(X_test)

print(f"Best Model Accuracy: {round(accuracy_score(y_pred_best, y_test), 2)}")
print(f"Best Model Recall: {round(recall_score(y_pred_best, y_test), 3)}")
print(f"Best Model Precision: {round(precision_score(y_pred_best, y_test), 2)}")
print(f"Best Model F1: {round(f1_score(y_pred_best, y_test), 2)}")
print(f"Best Model AUC: {round(roc_auc_score(y_pred_best, y_test), 2)}")

print("Classification Report:\n", classification_report(y_test, y_pred_best))

# FEATURE IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
#plot_importance(rf_model, X)
#print(kategorik_ama_kardinal)

