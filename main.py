import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

class RecommenderSystem:
    def __init__(self, college_data):
        self.college_data = college_data
        self.scalers = StandardScaler()
        self.encoders = {}
        self.models = {}
        self.features = ['strand', 'gwa', 'verbal', 'numerical', 'abstract', 
                             'spelling', 'usage', 'program', 
                             'soft_skills', 'hard_skills']
        
        
    def loading_data(self):
        print("-----Loading Data-----")
        try:
            self.data = pd.read_csv(self.college_data)
        except FileNotFoundError:
            print("Error: File not found")
            return
        
        # initial_count = len(self.data)
        # self.data = self.data.dropna() 
        # print(f"Data Cleaning: Dropped {initial_count - len(self.data)} rows with missing values.")
        
        categorical = ["strand", "program", "soft_skills", "hard_skills"]

        for col in categorical:
            le = LabelEncoder()
            if col in self.data.columns:
                self.data[col] = le.fit_transform(self.data[col])
                self.encoders[col] = le
            else:
                print(f"Warning:  Column {col} is not found.")

        target = 'acad_status'
        if target not in self.data.columns:
            print(f"Error: Target column '{target}' not found.")
            return
        
        self.data['target'] = self.data[target].astype(str).str.lower()
        binary_map = {'regular':1, 'irregular':0}

        #create a new column on the file with binary values
        self.data['target_col'] = self.data['target'].map(binary_map)

        #data catch for typo in csv file
        self.data = self.data.dropna(subset=['target_col'])

        #feature validation
        valid_features = [c for c in self.features if c in self.data.columns]
        X = self.data[valid_features]
        y = self.data["target_col"]
        
        #data splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        #converting pandas to numpy
        self.y_train = self.y_train.values
        self.y_test = self.y_test.values

        #scaling
        self.X_train = self.scalers.fit_transform(self.X_train)
        self.X_test = self.scalers.transform(self.X_test)

        print("Data is successfully loaded and preprocessed")

    #out-of-fold training for rf+nb
    def oofprediction(self, model, X, y):

        oof_prediction = np.zeros((X.shape[0], 1))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #reminder to add in rrl 

        for train_index, val_index in skf.split(X, y):
            X_fold_train, X_fold_val = X[train_index], X[val_index]
            y_fold_train = y[train_index]

            from sklearn.base import clone
            clf  =clone(model)
            clf.fit(X_fold_train, y_fold_train)

            preds = clf.predict_proba(X_fold_val)[:, 1]
            oof_prediction[val_index] = preds.reshape(-1, 1)
        
        return oof_prediction



    def train_stacking(self):
        print("\n ----- Level 0 Stacking Model Training -----")

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )
        nb = GaussianNB()

        #generating input for ffnn from rf+nb stacking
        print("Generating RF prediction")
        rf_oof = self.oofprediction(rf, self.X_train, self.y_train)

        print("Generating NB prediction")
        nb_oof = self.oofprediction(nb, self.X_train, self.y_train)

        self.X_train_meta = np.hstack([rf_oof, nb_oof])

        rf.fit(self.X_train, self.y_train)
        nb.fit(self.X_train, self.y_train)
        self.models['rf'] = rf
        self.models['nb'] = nb

        rf_test = rf.predict_proba(self.X_test)[:, 1].reshape(-1,1)
        nb_test = nb.predict_proba(self.X_test)[:, 1].reshape(-1,1)
        self.X_test_meta = np.hstack([rf_test, nb_test])

        print("\n ----- Level 1 Meta-Learner Model Training -----")

        meta_dim = self.X_train_meta.shape[1]


        ffnn = Sequential([
            Input(shape=(meta_dim,)),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        #ffnn compilation
        ffnn.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        #early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        ffnn.fit(self.X_train_meta, self.y_train,
                 validation_data=(self.X_test_meta, self.y_test),
                 epochs=100, batch_size=16, verbose=1, callbacks=[early_stop])


        self.models['meta'] = ffnn

    def evaluation(self):
        print("\n ----- Final Performance Evaluation -----")

        final_probs = self.models['meta'].predict(self.X_test_meta)

        #random forest evaluation
        rf_preds = self.models['rf'].predict(self.X_test)
        rf_acc = accuracy_score(self.y_test, rf_preds)

        #naive bayes evaluation
        nb_preds = self.models['nb'].predict(self.X_test)
        nb_acc = accuracy_score(self.y_test, nb_preds)

        #meta learner (ffnn) evaluation
        meta_preds = (final_probs >= 0.5).astype(int)
        meta_acc = accuracy_score(self.y_test, meta_preds)

        #comparison table
        print("\n--- Model Accuracy Comparison ---")
        print(f"Random Forest Only:    {rf_acc:.2%}")
        print(f"Naive Bayes Only:      {nb_acc:.2%}")
        print("-" * 30)
        print(f"Meta Learner Accuracy: {meta_acc:.2%}")

        #result context
        best_base = max(rf_acc, nb_acc)
        if meta_acc > best_base:
            print(f"\nStacking improved accuracy by {meta_acc - best_base:.2%}")
        elif meta_acc == best_base:
            print(f"\nStacking matched the best base model.")
        else:
            print(f"\nStacking slightly underperformed (Overfitting risk).")

        
        #confusion matrix
        print("\n--- Confusion Matrix (Stacked Model) ---")

        conf = confusion_matrix(self.y_test, meta_preds)
        print(f"Confusion Matrix: \n{conf}")

        print("\nClassification Report (1=Regular, 0=Irregular):")
        print(classification_report(self.y_test, meta_preds))

    def recommend_program(self, student_dict, threshold=0.70):
        print("\n ----- Program Analysis and Recommnedation -------")

        current_program = student_dict.get('program', 'Unknown')

        def get_prediction(prog_name):
            try:
                temp_df = pd.DataFrame([student_dict])
                temp_df['program'] = prog_name

                row_vals = []

                for col in self.features:
                    if col not in temp_df.columns: continue
                    val = temp_df.iloc[0][col]

                    if col in self.encoders:
                        try: 
                            val = self.encoders[col].transform([val])[0]
                        except:
                            val = 0
                    row_vals.append(val)

                X_input_df = pd.DataFrame([row_vals], columns=self.features)
                X_input = self.scalers.transform(X_input_df)

                p_rf = self.models['rf'].predict_proba(X_input)[:, 1].reshape(-1, 1)
                p_nb = self.models['nb'].predict_proba(X_input)[:, 1].reshape(-1, 1)
                meta_input = np.hstack([p_rf, p_nb])
            
                return self.models['meta'].predict(meta_input, verbose=0)[0][0]
            except Exception as e:
                print(f"Skipping {prog_name} due to data error: {e}")
                return 0.0
        
        #program evaluation if success rate is high
        current_score = get_prediction(current_program)
        print(f"\nCurrent Choice: {current_program}")
        print(f"Predicted Success Rate: {current_score:.2%}")

        if current_score >= threshold:
            print(f"Prediction shows high confidence. The {current_program} program fits the student")
            return
        
        else:
            print(f"Prediction shows significant risk. Program chosen falls below {threshold:.2%}")
            print(f"Searching for alternative programs with highher sucess rates... ")

            available_programs = self.encoders['program'].classes_
            recommendations = []

            for prog in available_programs:
                if prog == current_program: continue

                score = get_prediction(prog)

                if score >= threshold:
                    recommendations.append((prog, score))
                
            recommendations.sort(key=lambda x: x[1], reverse=True)

            if not recommendations:
                print("\n No other program exceeds the 70% threshold based in your profile")
                print("However, these are your best options:")

                all_prog = []
                for prog in available_programs:
                    if prog ==  current_program: continue
                    all_prog.append((prog, get_prediction(prog)))
                all_prog.sort(key=lambda x: x[1], reverse=True)
                recommendations = all_prog[:3]

                print(f"\n Recommended Alternatives:")
                for i, (prog, score) in enumerate(recommendations[:3], 1):
                    print(f"{i}, {prog} ({score:.2%} chance of regular)")

if __name__ == "__main__":

    #initializing the class
    stacker = RecommenderSystem(r"C:\Users\Rex Gatchalian\Desktop\Recommendation_Model\college_data.csv")

    #Loading and training the models
    stacker.loading_data()
    stacker.train_stacking()
    stacker.evaluation()

    print("\n" + "="*50)
    print("   TESTING RECOMMENDER FUNCTIONALITY   ")
    print("="*50)


    # #debugger
    # program_0 = stacker.encoders['program'].inverse_transform([0])[0]
    # skill_0 = stacker.encoders['hard_skills'].inverse_transform([0])[0]
    # print(f"\n[debug] ID 0 maps to -> Program: '{program_0}', Skill: '{skill_0}'")

    test_student_1= {#mismatch student, should prompt recommendations
        'student_name': 'Student A',
        'strand': 'HUMSS',               
        'gwa': 80, 
        'verbal': 2, 
        'numerical': 1, 
        'abstract': 2, 
        'spelling': 1, 
        'usage': 2, 
        'program': 'Medical Biology',   
        'soft_skills': 'Critical Thinking', 
        'hard_skills': 'IT Fundamentals'  
    }
    print(f"\n--- Analyzing: {test_student_1['student_name']} ---")
    stacker.recommend_program(test_student_1)
    
    # print(f"\n--- Testing {test_student_1['student_name']} ---")
    

    # Debug checker: Manually check encoding before running the full function
    # try:
    #     prog_id = stacker.encoders['program'].transform([test_student_debug['program']])[0]
    #     print(f"Program '{test_student_debug['program']}' found! ID: {prog_id}")
    # except:
    #     print(f"ERROR: Program '{test_student_debug['program']}' NOT FOUND. Defaults to 0.")

    # try:
    #     skill_id = stacker.encoders['hard_skills'].transform([test_student_debug['hard_skills']])[0]
    #     print(f"Skill '{test_student_debug['hard_skills']}' found! ID: {skill_id}")
    # except:
    #     print(f"ERROR: Skill '{test_student_debug['hard_skills']}' NOT FOUND. Defaults to 0.")

    test_student_2= {  #match student
        'student_name': 'Student B',
        'strand': 'STEM',               
        'gwa': 96, 
        'verbal': 3, 
        'numerical': 3, 
        'abstract': 2, 
        'spelling': 2, 
        'usage': 3, 
        'program': 'Computer Science',   
        'soft_skills': 'Critical Thinking', 
        'hard_skills': 'Programming'  
    }
    print(f"\n--- Analyzing: {test_student_2['student_name']} ---")
    stacker.recommend_program(test_student_2)

    
    