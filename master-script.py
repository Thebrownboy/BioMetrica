import pickle 
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report


def cleaning_data(df):
    copy_df = df.copy()
    # converting the categorical data to numeric data . 
    # in H_cal_consump we don't need one hot vector it just one or zero
    copy_df["H_Cal_Consump"].replace(["yes","no"],[1,0], inplace=True)
    copy_df["Gender"].replace(['Male',"Female"],[1,0], inplace=True)
    copy_df["Alcohol_Consump"].replace(["no","Sometimes","Frequently","Always"],[0,1,2,3],inplace=True)
    copy_df["Smoking"].replace(["yes","no"],[1,0], inplace=True)
    copy_df=pd.concat([copy_df,pd.get_dummies(copy_df['Food_Between_Meals'],prefix="FBM_")],axis=1)
    if "FBM__Always" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"FBM__Always":[0]*copy_df.shape[0]})],axis=1)
    if "FBM__Frequently" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"FBM__Frequently":[0]*copy_df.shape[0]})],axis=1)
        
    if "FBM__Sometimes" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"FBM__Sometimes":[0]*copy_df.shape[0]})],axis=1)
    if "FBM__no" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"FBM__no":[0]*copy_df.shape[0]})],axis=1)
        
    copy_df.drop(['Food_Between_Meals'],axis=1,inplace=True)
    copy_df['Fam_Hist'].replace(['yes','no'],[1,0],inplace=True)
    copy_df['H_Cal_Burn'].replace(['yes','no'],[1,0],inplace=True)
    copy_df = pd.concat([copy_df,pd.get_dummies(copy_df["Transport"])],axis=1)
    
    if "Automobile" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"Automobile":[0]*copy_df.shape[0]})],axis=1)
    
    if "Bike" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"Bike":[0]*copy_df.shape[0]})],axis=1)
    
    if "Motorbike" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"Motorbike":[0]*copy_df.shape[0]})],axis=1)
        
    if "Public_Transportation" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"Public_Transportation":[0]*copy_df.shape[0]})],axis=1)
        
    if "Walking" not in copy_df.columns:
        copy_df=pd.concat([copy_df,pd.DataFrame({"Walking":[0]*copy_df.shape[0]})],axis=1)
        
        
    copy_df.drop(['Transport'],axis=1,inplace=True)
    copy_df["Body_Level"].replace(["Body Level 1","Body Level 2","Body Level 3","Body Level 4"],
                              [0,1,2,3],inplace=True)
    
    cols=['Gender', 'Age', 'Height', 'Weight', 'H_Cal_Consump', 'Veg_Consump',
       'Water_Consump', 'Alcohol_Consump', 'Smoking', 'Meal_Count', 'Fam_Hist',
       'H_Cal_Burn', 'Phys_Act', 'Time_E_Dev', 'Body_Level', 'FBM__Always',
       'FBM__Frequently', 'FBM__Sometimes', 'FBM__no', 'Automobile', 'Bike',
       'Motorbike', 'Public_Transportation', 'Walking']
    copy_df=copy_df[cols]
    return copy_df

dict_class= {0:"Body Level 1",1:"Body Level 2",2:"Body Level 3",3:"Body Level 4"}

def test_model(model,X_test,Y_test):
    Y_test_pred=model.predict(X_test)
    Y_test_pred=np.argmax(Y_test_pred,axis=1)
    with open("pred.txt","w") as file  :
        for i in Y_test_pred : 
            file.write(dict_class[i])
            file.write("\n")

    accurate_accuracy= np.sum(Y_test_pred==Y_test)/Y_test.shape[0]
    classify_repo = classification_report(Y_test, Y_test_pred)
    
    return accurate_accuracy , classify_repo



scaler = None 

with open("scaler",'rb')as fb : 
    scaler= pickle.load(fb)


test = pd.read_csv("test.csv")
test= cleaning_data(test)

Y_test=test["Body_Level"].to_numpy()
X_test=test.drop(["Body_Level"],axis=1).to_numpy()
X_test = scaler.transform(X_test)
print(X_test.shape,Y_test.shape)

model= None 

with open("before_samplingmaster.h5","rb") as fb : 
    model = pickle.load(fb)


accuracy , repo =  test_model(model, X_test, Y_test)

