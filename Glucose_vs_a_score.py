import matplotlib.pyplot as plt
import random


anomaly_score = [round(random.uniform(0.01, 0.05),3 )for i in range(20)]
glucose_value = [round(random.uniform(5.01, 8.05),3 )for j in range(20)]

plt.scatter(glucose_value,anomaly_score)
plt.ylabel("Anomaly Score")
plt.xlabel("Glucose")
plt.title('Glucose vs Anomaly')
plt.show()