import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("seaborn-whitegrid")
#########################################################################################
### page 설정
st.sidebar.title('palmer penguins')
option=st.sidebar.selectbox('palmer penguins data',('데이터EDA','모델링'))

#########################################################################################
### data table
if option=='데이터EDA':
    st.title('palmer penguins 데이터 분석')
    @st.cache(allow_output_mutation=True)

    def load_penguins():
        return sns.load_dataset("penguins")

    df = load_penguins()
    st.subheader("데이터 설명")
    st.write(
        """
        - Species: penguin species (Chinstrap, Adélie, or Gentoo)      
        - Island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
        - bill_length_mm: bill length (mm)
        - bill_depth_mm: bill depth (mm)
        - flipper_length_mm: flipper length (mm) 
        - body_mass_g: body mass (g)
        - Sex: penguin sex
        """
    )
    st.subheader( "palmer penguins 데이터")
    st.write(df)
############################################################################################
### distplot
    penguins = df.copy()
    X = penguins["flipper_length_mm"]
    Y = penguins["body_mass_g"]

    st.subheader("data plot")
    col1, col2 = st.beta_columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax = df['species'].value_counts().plot(kind='barh')
        plt.title('Species value')
        st.pyplot(fig)
    with col2:
        title = "Flipper length(mm) vs Body mass(g)"
        fig, ax = plt.subplots()
        ax.scatter(X, Y)
        plt.title(title)
        plt.xlim(160, 240)
        plt.ylim(2500, 6500)
        st.pyplot(fig)
############################################################################################
### Bill Depth Distribution
    df.drop(df[df['body_mass_g'].isnull()].index, axis=0, inplace=True)
    df['sex'] = df['sex'].fillna('Male')
    df.drop(df[df['sex'] == '.'].index, inplace=True)

    st.subheader("Bill Depth Distribution")
    col1, col2 = st.beta_columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=df, x="species", y="bill_length_mm", size=8)
        #plt.title('Species value')
        st.pyplot(fig)
    with col2:
        title = "Flipper length(mm) vs Body mass(g)"
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=df, x="species", y="bill_length_mm")
        #plt.title(title)
        st.pyplot(fig)
############################################################################################
### PairGrid
    st.subheader("PairGrid")
    penguins = sns.load_dataset("penguins")
    penguins["body_mass_100g"] = penguins["body_mass_g"] / 100
    data = penguins.drop('body_mass_g', axis=1)
    labels = ["Bill Length (mm)", "Bill Depth (mm)", "Flipper Length (mm)", "Body Mass (100g)"]
    label_fontdict = {"fontsize": 14, "fontweight": "bold", "color": "gray"}
    labelpad = 12
    tick_labelsize = "large"
    g = sns.PairGrid(data, hue="species", diag_sharey=False)
    g.map_diag(sns.kdeplot, fill=True)
    g.map_lower(sns.scatterplot)
    g.map_lower(sns.regplot, scatter=False, truncate=False, ci=False)
    g.map_upper(sns.kdeplot, alpha=0.3)
    g.add_legend()
    for i in range(4):
        g.axes[3, i].set_xlabel(labels[i], fontdict=label_fontdict)
        g.axes[i, 0].set_ylabel(labels[i], fontdict=label_fontdict)
    g.fig.align_ylabels(g.axes[:, 0])
    st.pyplot(g)
#########################################################################################
### 모델링
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold,train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, precision_score, recall_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import export_graphviz
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if option=='모델링':
    def load_penguins():
        return sns.load_dataset("penguins")
    df1 = load_penguins()
    df1.drop(df1[df1['body_mass_g'].isnull()].index, axis=0, inplace=True)
    df1['sex'] = df1['sex'].fillna('Male')
    df1.drop(df1[df1['sex'] == '.'].index, inplace=True)

    st.title('palmer penguins 모델링')
    target = 'sex'
    encode = ['species', 'island']
    target_mapper = {'Male': 0, 'Female': 1}
    df1['sex'] = df1['sex'].map(target_mapper)

    target_mapper = {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}
    df1['species'] = df1['species'].map(target_mapper)

    target_mapper = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    df1['island'] = df1['island'].map(target_mapper)

    y = df1['sex']
    X = df1.drop('sex', axis=1)

    kfold = StratifiedKFold(n_splits=5)
    random_state = 1
    clf = []

    clf.append(XGBClassifier(random_state=random_state))
    clf.append(LGBMClassifier(random_state=random_state))
    clf.append(KNeighborsClassifier())
    clf.append(RandomForestClassifier(random_state=random_state))
    clf.append(GradientBoostingClassifier(random_state=random_state))
    clf.append(DecisionTreeClassifier(random_state=random_state))
    clf.append(LogisticRegression(random_state=random_state))
    clf.append(SVC(random_state=random_state))

    clf_results = []
    for classifier in clf:
        clf_results.append(cross_val_score(classifier, X, y=y, scoring="f1_weighted", cv=kfold, n_jobs=4))
    clf_means = []
    clf_std = []
    for clf_result in clf_results:
        clf_means.append(clf_result.mean())
        clf_std.append(clf_result.std())
    clf_re = pd.DataFrame({"CrossValMeans": clf_means, "CrossValerrors": clf_std},
                          index=['XGB', 'LGB', 'KNeighbors', 'RF', 'GBC', 'DT', 'Logist', 'SVC'])
    st.subheader("kfold 결과")
    st.write(clf_re)

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(clf_re['CrossValerrors'], color='r')

    ax2 = ax1.twinx()
    ax2.bar(clf_re.index, clf_re['CrossValMeans'], alpha=0.3, color='teal')
    st.subheader("모델 비교")
    st.pyplot(fig)
#########################################################################################
### LGB
    st.title('LightGBM')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True)
    def cnf_matrix_model(model):
        cnf_matrix_gbc = confusion_matrix(y_val, model)
        g = sns.heatmap(pd.DataFrame(cnf_matrix_gbc), annot=True, cmap="BuGn", fmt='g')
        buttom, top = g.get_ylim()
        g.set_ylim(buttom + 0.5, top - 0.5)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        return g


    def four_f1(model_f1, model):
        print("F1 Cross_validate", model_f1)
        print("F1 Macro:", f1_score(y_val, model, average='macro'))
        print("F1 Micro:", f1_score(y_val, model, average='micro'))
        print("F1 Weighted:", f1_score(y_val, model, average='weighted'))
        print("\nMatrix of confusion")
        return confusion_matrix(y_val, model)


    model_lgbc = LGBMClassifier(random_state=16)
    model_lgbc.fit(X_train, y_train)
    Predicted_lgbc = model_lgbc.predict(X_val)
    lgbc_f1 = cross_val_score(model_lgbc, X_train, y_train, cv=kfold, scoring='f1_weighted').mean()

    st.subheader('LGB_confusion_matrix')
    fig, ax = plt.subplots()
    ax = cnf_matrix_model(Predicted_lgbc)
    st.pyplot(fig)

    st.subheader('LGB_feature_importance')
    feature_importance = model_lgbc.feature_importances_
    a = pd.Series(feature_importance, X_train.columns).sort_values(ascending=True)
    b = pd.DataFrame(a, columns=['feature_importance']).tail(15)
    st.bar_chart(b)

    st.subheader('LGB_tree')
    a= lgb.create_tree_digraph(model_lgbc, orientation='vertical')
    st.write(a)

#########################################################################################
### RF
    st.title('RandomForest')
    model_rf = RandomForestClassifier(random_state=16)
    model_rf.fit(X_train, y_train)
    Predicted_rf = model_rf.predict(X_val)
    rf_f1 = cross_val_score(model_rf, X_train, y_train, cv=kfold, scoring='f1_weighted').mean()

    st.subheader('RF_confusion_matrix')
    fig, ax = plt.subplots()
    ax = cnf_matrix_model(Predicted_rf)
    st.pyplot(fig)

    st.subheader('RF_feature_importance')
    feature_importance = model_rf.feature_importances_
    a = pd.Series(feature_importance, X_train.columns).sort_values(ascending=True)
    b = pd.DataFrame(a, columns=['feature_importance']).tail(15)
    st.bar_chart(b)

    st.subheader('RF_tree')
    estimator = model_rf.estimators_[3]
    g = export_graphviz(estimator, out_file='tree.dot',
                    feature_names=X_train.columns,
                    class_names=['1', '2', '3', '4'],
                    max_depth=2,  # 표현하고 싶은 최대 depth
                    precision=2,  # 소수점 표기 자릿수
                    filled=True,  # class별 color 채우기
                    rounded=True,  # 박스의 모양을 둥글게
                    )
    with open("tree.dot") as f:
        dot_graph = f.read()
    st.graphviz_chart(dot_graph)
#########################################################################################
### XGB
    st.title('XGboost')
    model_xgb = xgb.XGBClassifier(random_state=16)
    model_xgb.fit(X_train, y_train)
    Predicted_xgb = model_xgb.predict(X_val)
    xgb_f1 = cross_val_score(model_xgb, X_train, y_train, cv=kfold, scoring='f1_weighted').mean()

    st.subheader('XGB_confusion_matrix')
    fig, ax = plt.subplots()
    ax = cnf_matrix_model(Predicted_xgb)
    st.pyplot(fig)

    st.subheader('XGB_feature_importance')
    feature_importance = model_xgb.feature_importances_
    a = pd.Series(feature_importance, X_train.columns).sort_values(ascending=True)
    b = pd.DataFrame(a, columns=['feature_importance']).tail(15)
    st.bar_chart(b)

    st.subheader('XGB_tree')
    node_params = {'shape': 'box',
                   'style': 'filled, rounded',
                   'fillcolor': 'LightBlue'}
    leaf_params = {'shape': 'box',
                   'style': 'filled',
                   'fillcolor': 'GreenYellow'}
    diagraph = xgb.to_graphviz(model_xgb, num_trees=0, size='5,10',
               condition_node_params=node_params,
               leaf_node_params=leaf_params)
    diagraph.format = 'png'
    st.image(diagraph.view())
#########################################################################################
### Clustering
    st.title('Clustering')
    col1, col2 = st.beta_columns(2)
    X = df1[['bill_length_mm', 'bill_depth_mm']]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X.loc[:, 'bill_length_mm'], X.loc[:, 'bill_depth_mm'], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    with col1:
        fig, ax = plt.subplots()
        plt.scatter(X.loc[:, 'bill_length_mm'], X.loc[:, 'bill_depth_mm'], c=y_kmeans, s=50, cmap='viridis')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.title('CLUSTERING ON Bill LENGTH AND Bill DEPTH')
        st.pyplot(fig)

    X = df1[['flipper_length_mm', 'body_mass_g']]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    with col2:
        title = "CLUSTERING ON FLIPPER LENGTH AND BODY MASS"
        fig, ax = plt.subplots()
        plt.scatter(X.loc[:, 'flipper_length_mm'], X.loc[:, 'body_mass_g'], c=y_kmeans, s=50, cmap='viridis')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.title(title)
        st.pyplot(fig)
