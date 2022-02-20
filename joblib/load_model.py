import joblib
import warnings
warnings.filterwarnings('ignore')

model = joblib.load('diabities.pkl')

output = model.predict([[1,2,3,4,5,6,7,8]])

print(output)

if output[0] == 1:
    print('diabatic')
else:
    print('not diabatic')