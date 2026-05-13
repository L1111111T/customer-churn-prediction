# Customer Churn Prediction

> 电信用户流失预测 · [Kaggle 比赛页面](https://www.kaggle.com/competitions/udel-customer-churn-prediction)
> 公榜得分：0.79772
> 比赛排名：143 / 525  (前27%)
---

## 项目背景

电信行业用户流失率直接影响企业营收，提前识别高风险用户是精准营销的核心问题。本项目基于用户合同、消费行为、服务订阅等19个特征，构建二分类模型预测用户是否会在下一周期内流失，为业务团队提供早期预警依据。

数据集包含训练集 5,634 条、测试集 1,409 条记录，目标变量 `Churn` 存在约 **26% 正样本**的类别不均衡问题。

---

## 方法流程

### 1. 数据加载与清洗
- 使用 `pandas` 读取 CSV，`df.info()` 检查字段类型与缺失情况
- `TotalCharges` 字段含非数值字符，用 `pd.to_numeric(errors='coerce')` 强制转换后删除缺失行
- 分离数值特征（4个）与类别特征（15个）分别处理

### 2. 特征工程
使用 `sklearn.pipeline.Pipeline` + `ColumnTransformer` 封装完整预处理流程，
防止训练集信息泄露到测试集：

| 特征类型 | 处理步骤 |
|---------|---------|
| 数值特征（4个） | `SimpleImputer(mean)` → `StandardScaler` |
| 类别特征（15个） | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

### 3. 建模与对比

逐步从简单到复杂，共训练 **4 个模型**：

| 模型 | 核心思路 | 调参方式 |
|------|---------|---------|
| DummyClassifier（基线） | 全部预测多数类，提供下界参考 | 无 |
| 逻辑回归 | 梯度下降优化对数损失，线性决策边界 | 5折交叉验证 |
| 决策树 | 递归分裂特征空间，熵/基尼系数选择分裂点 | GridSearchCV（10折） |
| 随机森林 | Bagging 集成多棵决策树，随机特征子集投票 | GridSearchCV（5折，F1评分） |

### 4. 超参数搜索范围

**决策树 GridSearchCV：**
```python
param_grid = {
    'preprocessor__num_pipeline__num_imputer__strategy': ['mean', 'median'],
    'preprocessor__cat_pipeline__cat_imputer__strategy': ['most_frequent'],
    'dt_clf__criterion': ['gini', 'entropy'],
    'dt_clf__max_depth': [2,3,5,7,10]
}
# cv=10, scoring='accuracy'
# 最优：criterion='gini', max_depth=3
```

**随机森林 GridSearchCV：**
```python
rf_param_grid = {
    'rf_clf__n_estimators': [50, 100],
    'rf_clf__max_depth':    [3, 5, 7],
    'rf_clf__criterion':    ['gini', 'entropy']
}
# cv=5, scoring='f1'
```

---

## 核心结果

| 模型 | 测试集准确率 | F1（流失类） | AUC-ROC |
|------|------------|------------|---------|
| 基线（DummyClassifier） | 0.7365 | 0.00 | 0.50 |
| 逻辑回归 | 0.7959 | 0.56 | — |
| 决策树（调参后）| **0.7799** | 0.48 | — |
| 随机森林（调参后）| — | — | — |

> **最终选用决策树模型**提交 Kaggle 公榜  
> **得分：0.79772 · 排名：143 / 525（前 27%）**

**决策树最终评估（测试集）：**
```
              precision    recall  f1-score
No  （未流失）   0.79      0.95      0.86
Yes （流失）    0.73      0.36      0.48
accuracy                           0.78
```

混淆矩阵：
```
预测 No  预测 Yes
实际 No  [ 765    43 ]
实际 Yes [ 205   114 ]
```

---

## 特征重要性（随机森林）

随机森林识别出的 Top 10 最重要特征：

| 排名 | 特征 | 业务含义 |
|-----|------|---------|
| 1 | tenure | 在网时长越短，流失风险越高 |
| 2 | Contract_Month-to-month | 月付合同用户流失率显著更高 |
| 3 | OnlineSecurity_No | 未开通网络安全服务的用户更易流失 |
| 4 | InternetService_Fiber optic | 光纤用户流失率高于DSL用户 |
| 5 | TechSupport_No | 未使用技术支持服务的用户流失风险更高 |
| 6 | TotalCharges | 累计消费金额与在网时长高度相关 |
| 7 | PaymentMethod_Electronic check | 电子支票付款用户流失率明显偏高 |
| 8 | Contract_Two year | 两年合同用户忠诚度显著更高 |
| 9 | StreamingMovies_No internet service | 无网络服务用户的特征标识 |
| 10 | OnlineSecurity_Yes | 已开通网络安全服务对留存有正向作用 |

---

## 数据获取

本项目数据来自 Kaggle 竞赛官网：
1. 访问 [比赛页面](https://www.kaggle.com/competitions/udel-customer-churn-prediction)
2. 在 **Data** 中下载并解压 `udel-churn-train.csv` 和 `udel-churn-test.csv` 放在与 Notebook 相同的目录下

---

## 环境配置

```bash
pip install -r requirements.txt
```

在 Jupyter Notebook 中打开 `CustomerChurn_code.ipynb` 并按顺序运行所有 cell。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `CustomerChurn_code.ipynb` | 完整分析流程：EDA → 特征工程 → 建模 → 评估 → 提交 |
| `requirements.txt` | Python 依赖包版本 |
| `.gitignore` | 排除数据文件和临时文件 |

---

## 技术栈

`Python 3.12` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `seaborn`

**核心方法：** Pipeline · ColumnTransformer · GridSearchCV · Cross-Validation  
**算法：** Logistic Regression · Decision Tree · Random Forest (Bagging Ensemble)
