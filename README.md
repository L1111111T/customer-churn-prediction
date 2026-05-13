# Customer Churn Prediction

> 电信用户流失预测 · Kaggle 竞赛项目  
> **比赛排名：143 / 525 **
> [Kaggle 比赛页面](https://www.kaggle.com/competitions/customer-churn-prediction-udel)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 项目背景

电信行业用户流失率直接影响企业营收。本项目基于用户合同、消费行为、服务订阅等
19个特征，构建分类模型预测用户是否会在下一周期内流失，为业务团队提供早期预警依据。

数据集包含训练集 5,634 条、测试集 1,409 条记录，目标变量 `Churn` 存在约 **26% 正样本**的类别不均衡问题。

---

## 方法流程

### 1. 数据清洗
- `TotalCharges` 字段含非数值字符，使用 `pd.to_numeric(errors='coerce')` 转换后删除缺失行
- 分离数值特征与类别特征，分别进入不同的处理管道

### 2. 特征工程（Pipeline封装）
- **数值特征**：`SimpleImputer(mean)` 填补缺失值 → `StandardScaler` 标准化
- **类别特征**：`SimpleImputer(most_frequent)` 填补 → `OneHotEncoder` 编码
- 使用 `ColumnTransformer` 整合为统一预处理器，封装进 `Pipeline` 防止数据泄露

### 3. 建模与对比

| 模型 | 说明 | 调参方式 |
|------|------|---------|
| DummyClassifier（基线） | 全部预测多数类，作为下界参考 | 无 |
| 逻辑回归 | 梯度下降优化，可解释性强 | 默认参数 + 5折CV |
| 决策树 | 非线性分类，控制深度防过拟合 | GridSearchCV（10折）|
| 随机森林 | Bagging集成，多棵树投票降低方差 | GridSearchCV（5折，F1评分）|

### 4. 评估指标
因存在类别不均衡，以 **F1-score** 和 **AUC-ROC** 为主要评估指标，而非单纯准确率。

---

## 核心结果

| 模型 | Accuracy | F1（流失类） | AUC-ROC |
|------|----------|------------|---------|
| 基线（DummyClassifier） | 0.73 | 0.00 | 0.50 |
| 逻辑回归 | 0.XX | 0.XX | 0.XX |
| 决策树（调参后） | 0.XX | 0.XX | 0.XX |
| 随机森林（调参后） | 0.XX | 0.XX | 0.XX |

> 最终选用**决策树**提交 Kaggle，公榜排名 **483/525**

---

## 可视化结果

### 特征重要性 Top 10
![特征重要性](outputs/feature_importance.png)

### 决策树混淆矩阵
![混淆矩阵](outputs/confusion_matrix.png)

### 用户流失分布
![流失分布](outputs/churn_distribution.png)

---

## 关键结论

1. **tenure（在网时长）** 和 **Contract（合同类型）** 是最重要的流失预测特征，月付合同用户流失率显著高于年付用户
2. **MonthlyCharges（月费）** 越高，流失风险越大，与业务直觉一致
3. 随机森林在 AUC 指标上优于单棵决策树，体现了集成学习降低方差的优势
4. 类别不均衡问题导致模型对流失用户（少数类）的召回率偏低，实际业务部署时建议调整决策阈值

---

## 项目结构

```
customer-churn-prediction/
├── CustomerChurn_code.ipynb   # 完整分析流程
├── outputs/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── churn_distribution.png
├── requirements.txt
└── README.md
```

---

## 环境配置

```bash
pip install -r requirements.txt
```

```
numpy>=1.26
pandas>=2.2
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
```

---

## 考核知识点覆盖

| 考核方向 | 本项目对应内容 |
|---------|--------------|
| 基础数据结构（列表、字典） | `tolist()` 管理特征名、`DataFrame` 存储结果对比 |
| 控制流与函数 | 数据清洗逻辑、Pipeline各步骤封装 |
| NumPy / Pandas | 数据读取、类型转换、描述统计、特征筛选 |
| 可视化 | 流失分布图、特征重要性图、混淆矩阵热力图 |
| 经典机器学习（分类） | 逻辑回归、决策树、随机森林全流程实现 |
| 面向对象 | Pipeline / ColumnTransformer 封装预处理与建模 |
| 算法复杂度 | 决策树 O(n·m·log n)，随机森林 O(T·n·m·log n) |
