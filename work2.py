import scipy.io as sio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import pydot
import os
os.environ['PATH'] = os.environ['PATH'] + (';c:\\Program Files (x86)\\Graphviz2.38\\bin\\')
data=sio.loadmat('data_label.mat')
df_data=pd.DataFrame(data['data'])
df_label=pd.DataFrame(data['label'])

x_train,x_test,y_train,y_test=train_test_split(df_data,df_label,test_size=0.3,random_state=4)

tree=DecisionTreeClassifier(random_state=4)
tree.fit(x_train,y_train)

print('Train score:{:.3f}'.format(tree.score(x_train,y_train)))
print('Test score:{:.3f}'.format(tree.score(x_test,y_test)))

#生成可视化图
export_graphviz(tree,out_file="tree.dot",class_names=['0','1','2','3','4'],impurity=False,filled=True)
#展示可视化图
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')