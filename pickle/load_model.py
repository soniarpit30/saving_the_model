import pickle
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('diabities.pkl' , 'rb'))

output = model.predict([[1,2,3,4,5,6,7,8]])

print(output)

if output[0] == 1:
    print('diabatic')
else:
    print('not diabatic')