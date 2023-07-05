import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Read the data from CSV in the working directory
data = pd.read_csv('agriculture_data.csv')

# Perform the orthogonal design analysis
model = ols('Crop_Yield ~ C(Fertilizer) + C(Irrigation) + C(Fertilizer):C(Irrigation)', data).fit()
anova_table = anova_lm(model)

# Visualize the results using box plots
sns.boxplot(x='Fertilizer', y='Crop_Yield', hue='Irrigation', data=data)
plt.title('Interaction plot of fertilizer type and irrigation frequency')
plt.xlabel('Fertilizer type')
plt.ylabel('Crop yield')
plt.legend(title='Irrigation frequency')
plt.show()
