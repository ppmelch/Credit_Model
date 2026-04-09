from sklearn.model_selection import train_test_split


class Data_preparation:
    def __init__(self , data):
        self.data = data.copy() 
    
    def prepare_data(self):
        
        # target
        y = self.data['loan_status']
        
        # Features 
        X = self.data.drop(columns=['loan_status', 'interest_rate'])
        
        # Encoding 
        X = X.get_dummies(X , drop_first=True)
        
        return X , y

