import matplotlib.pyplot as plt
import webbrowser
import pandas as pd
from sklearn.tree import plot_tree
import shap
# explainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold
base_folder=r'C:\\Users\\user\\git\\github\\py2401_rf_shap_pls\\'
FileName=base_folder+'regression_pls.csv'
columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','Target']
df=pd.read_csv(FileName, encoding='utf-8', engine='python', usecols=columns)
features=[c for c in df.columns if c !='Target']
train, test=train_test_split(df, test_size=0.2, random_state=115)
X_train=train[features]
y_train=train['Target'].values
X_test=test[features]
y_test=test['Target'].values
rf=RandomForestRegressor(n_estimators=775, max_depth=62, max_features=6, random_state=444)
rf.fit(X_train, y_train)
explainer=shap.TreeExplainer(
    model=rf,
    data=X_test,
    feature_perturbation='interventional',
    check_additivity=False
)
PdfFile_ShapSummary=base_folder+'pdf\\rf_shap.pdf'
shap_values=explainer(X_test)
shap_values_ar=explainer.shap_values(X_test)
print('shap_values: ', type(shap_values))
print('shap_values_ar: ', type(shap_values_ar))
# 
# 1-1 サマリBeeswarm
# 
shap.summary_plot(shap_values, X_test, max_display=10, show=False)
plt.savefig(PdfFile_ShapSummary, dpi=700)
plt.show()
webbrowser.open_new(PdfFile_ShapSummary)
# 
# 1-2 Beeswarm
# 
PdfFile_Beeswarm=base_folder+'pdf\\rf_shap_Beeswarm_pls.pdf'
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.savefig(PdfFile_Beeswarm, dpi=700)
plt.show()
webbrowser.open_new(PdfFile_Beeswarm)
# 
# 2 Waterfall
# 
PdfFile_Waterfall=base_folder+'pdf\\rf_shap_Waterfall_pls.pdf'
fig, ax=plt.subplots(figsize=(12,6))
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.savefig(PdfFile_Waterfall, bbox_inches='tight')
plt.show()
webbrowser.open_new(PdfFile_Waterfall)
# 
# 3-1 特徴量の影響（1レコード）
# 
PdfFile_Decision=base_folder+'pdf\\rf_shap_Decision_pls.pdf'
shap.decision_plot(explainer.expected_value, shap_values_ar[0], X_train.iloc[[0], :], show=False)
plt.savefig(PdfFile_Decision, bbox_inches='tight')
plt.show()
webbrowser.open_new(PdfFile_Decision)
# 
# 3-2 特徴量の影響（10レコード）
# 
PdfFile_Decision10=base_folder+'pdf\\rf_shap_Decision10_pls.pdf'
shap.decision_plot(explainer.expected_value, shap_values_ar[0:10], X_train.iloc[0:10, :], show=False)
plt.savefig(PdfFile_Decision10, bbox_inches='tight')
plt.show()
webbrowser.open_new(PdfFile_Decision10)
# 
# 3-3 特徴量の影響（全レコード）
# 
PdfFile_DecisionAll=base_folder+'pdf\\rf_shap_DecisionAll_pls.pdf'
shap.decision_plot(explainer.expected_value, shap_values_ar, X_train, show=False)
plt.savefig(PdfFile_DecisionAll, bbox_inches='tight')
plt.show()
webbrowser.open_new(PdfFile_DecisionAll)
# 
# 3-4 特徴量のSHAP値（1レコード目）
# 
PdfFile_1rec=base_folder+'pdf\\rf_shap_1rec_pls.pdf'
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values_ar[0], X_train.iloc[0, :], show=False, matplotlib=True)
plt.savefig(PdfFile_1rec, bbox_inches='tight')
plt.show()
webbrowser.open_new(PdfFile_1rec)
# 
# 4 特徴量の寄与度（SHAP値の絶対値の平均）
# 
PdfFile_MeanShap=base_folder+'pdf\\rf_shap_MeanShap_pls.pdf'
shap.plots.bar(shap_values, max_display=20, show=False)
plt.savefig(PdfFile_MeanShap, bbox_inches='tight')
plt.show()
webbrowser.open_new(PdfFile_MeanShap)
# 
# 5 特徴量の寄与度（SHAP値の絶対値の平均）
# 
PdfFile_X1=base_folder+'pdf\\rf_shap_X18_pls.pdf'
shap.plots.scatter(shap_values[:,'x18'], color=shap_values, show=False)
plt.savefig(PdfFile_X1, bbox_inches='tight')
plt.show()
webbrowser.open_new(PdfFile_X1)