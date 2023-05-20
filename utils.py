import pandas as pd 
from sklearn.utils import resample 
from tensorflow import keras 
from sklearn import preprocessing
import copy 
import tensorflow_addons as tfa
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from focal_loss import SparseCategoricalFocalLoss
import numpy as np 
from sklearn.metrics import classification_report
import pickle 

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
    
    
    
    
def data_spliting(df_original,ratio=1/3):
    class_0=df_original[df_original["Body_Level"]==0]
    class_1=df_original[df_original["Body_Level"]==1]
    class_2=df_original[df_original["Body_Level"]==2]
    class_3=df_original[df_original["Body_Level"]==3]

    class_0_test =class_0.iloc[0:int(class_0.shape[0] * ratio)]
    class_0_train=class_0.iloc[int(class_0.shape[0]*ratio):]

    class_1_test =class_1.iloc[0:int(class_1.shape[0] * ratio)]
    class_1_train=class_1.iloc[int(class_1.shape[0]*ratio):]


    class_2_test =class_2.iloc[0:int(class_2.shape[0] * ratio)]
    class_2_train=class_2.iloc[int(class_2.shape[0]*ratio):]

    class_3_test =class_3.iloc[0:int(class_3.shape[0] * ratio)]
    class_3_train=class_3.iloc[int(class_3.shape[0]*ratio):]

    all_class_test=pd.concat([class_0_test,class_1_test,class_2_test,class_3_test],axis=0)
    all_class_train=pd.concat([class_0_train,class_1_train,class_2_train,class_3_train],axis=0)

    return all_class_train.copy(),all_class_test.copy() 


def over_sampling(data,sampling_ratio=0.5):
    df_ = data[data["Body_Level"]==3]
    for i in range(3):
        df_minority = data[data["Body_Level"]==i]
        df_minority_upsample = resample(df_minority , replace =True ,
                                        n_samples=int(0.4*(454-df_minority.shape[0])),
                                        random_state=42)
        

        df_ = pd.concat([df_,df_minority_upsample,df_minority], axis=0,)

    oversampled=df_
    return oversampled.copy()





class CustomCallback(keras.callbacks.Callback):
    def __init__(self,model_path="Iam_model.h5"):
        self.max_val_acc=0
        self.max_tra_acc=0
        self.model_path=model_path
        
        # old one if the new one give dump results 
#          if  (val_acc > self.max_val_acc) and \
#             (tra_acc>val_acc or (val_acc>tra_acc and val_acc-tra_acc<=0.01) ):
    def on_epoch_end(self, epoch, logs=None):
        val_acc=logs.get("val_accuracy")
        tra_acc= logs.get("accuracy")
        if  (val_acc >= self.max_val_acc or \
             ( tra_acc > self.max_tra_acc  and val_acc >= self.max_val_acc) ) and \
            (tra_acc>val_acc or (val_acc>tra_acc and val_acc-tra_acc<=0.01) ):
            print(f"\nYes You are here {tra_acc} {val_acc}")
            self.max_val_acc=val_acc
            if tra_acc > self.max_tra_acc : 
                self.max_tra_acc = tra_acc 
                
#             with open(self.model_path,"wb") as fb: 
#                 pickle.dump(self.model,fb)
            self.model.save(self.model_path)



            
def build_model(gamma=2, learning_rate=0.01):
    model_NN=keras.Sequential([keras.Input(shape=(23)),
                        layers.Dense(4,activation='softmax')])
    
    model_NN.compile(loss=SparseCategoricalFocalLoss(from_logits=False,gamma=gamma),
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=["accuracy"])
    return model_NN
    
def data_scaling(data):
    pre_scaled_data=copy.deepcopy(data)
    scaler =preprocessing.StandardScaler().fit(pre_scaled_data)
    scaled_data=scaler.transform(pre_scaled_data)
    return scaled_data,scaler



def test_model(model,X_test,Y_test,axis=True):
    Y_test_pred=model.predict(X_test)
    if axis : 
        Y_test_pred=np.argmax(Y_test_pred,axis=1)
    accurate_accuracy= np.sum(Y_test_pred==Y_test)/Y_test.shape[0]
    classify_repo = classification_report(Y_test, Y_test_pred)
    
    return accurate_accuracy , classify_repo
    


def build_one_layer_model(gamma=2, learning_rate=0.01):
    model_one_layer  = keras.Sequential([keras.Input(shape=(23)),
                           layers.Dense(16,activation='relu'),
                           layers.Dropout(0.4),
                        layers.Dense(4,activation='softmax')])
    model_one_layer.compile(loss=SparseCategoricalFocalLoss(from_logits=False,gamma=gamma),
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=["accuracy"])
    return model_one_layer

    
    
    
    
    
    
    
    
    
    
    