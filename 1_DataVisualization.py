import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"data\airfoil_self_noise.dat", 
                 sep="\t+",  
                 usecols=[0, 1, 2, 3, 4, 5], 
                 names=['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness', 'SealedSoundPressureLevel'],
                 engine = 'python')

dataset.to_csv(r"data\airfoil_self_noise.csv", index = False)

#print(df.head)
#print(df.columns)
print(dataset.isna().sum())

print(dataset.describe())

print(dataset.corr())

for feat in ['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness']:
    outliers = []
    data = dataset[feat]
    mean = np.mean(data)
    std =np.std(data)
    
    for y in data:
        z_score= (y - mean)/std 
        if np.abs(z_score) > 3:
            outliers.append(y)
    print('\nOutlier caps for {}:'.format(feat))
    print('  --95p: {:.1f} / {} values exceed that'.format(data.quantile(.95), 
          len([i for i in data if i > data.quantile(.95)])))
    print('  --3sd: {:.1f} / {} values exceed that'.format(mean + 3*(std), len(outliers)))
    print('  --99p: {:.1f} / {} values exceed that'.format(data.quantile(.99),
          len([i for i in data if i > data.quantile(.99)])))
    
for feature in ['Frequency', 'AngleOfAttack', 'ChordLength', 'FreeStreamVelocity', 'SuctionSideDisplacementThickness']:
    sns.distplot(dataset[feature], kde=False)
    plt.title('Histogram for {}'.format(feature))
    plt.show()
    
