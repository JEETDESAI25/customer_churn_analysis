import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#agegroup vs churn plot
sns.set(style="white",font_scale = 1.5)
sns.countplot(x='ageGroup', hue = 'Churn',data = churn_df, palette="Accent_r")
plt.show()

#customer status vs churn plot
sns.set(style="white",font_scale = 1.5)
sns.countplot(x='status', hue = 'Churn',data = churn_df, palette="Accent_r")
plt.show()

sns.set(style="white",font_scale = 2,rc={'figure.figsize':(20,8.27)})
sns.countplot(x='call_failure', hue = 'complains', data = df, palette="Accent_r")
plt.show()

sns.set(style="white",font_scale = 1.5)
sns.countplot(x='charge_amount', hue = 'ageGroup',data = df, palette="Accent_r")
plt.show()

plt.pie(df['ageGroup'].value_counts(), labels=df['ageGroup'].value_counts().index.tolist(), autopct='%1.0f%%',shadow = True)
plt.title('Distribution of Age', fontsize = 20)
plt.show()

churn_df['call_failure'].hist(bins=30,figsize=(10,10),color=(0,0.4,0.4))
plt.xlabel("Call Failure",fontsize=17)
plt.ylabel("Count value", fontsize=17)
plt.show()

churn_df['subs_len'].hist(bins=30,figsize=(10,10),color=(0,0.4,0.4))
plt.xlabel("Subscription  Length",fontsize=17)
plt.ylabel("Count value", fontsize=17)
plt.show()

churn_df['seconds_of_use'].hist(bins=30,figsize=(10,10),color=(0,0.4,0.4))
plt.xlabel("Seconds of use",fontsize=17)
plt.ylabel("Count value", fontsize=17)
plt.show()
