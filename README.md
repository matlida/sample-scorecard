### Folder a-b ：信用风险评分卡之A\b卡，申请评分卡 行为评分卡
#### code : a_card    -----      data: LoanStats_2018Q3
####    code : b_card  -----      data: testData & trainData


建模时注意的问题。

1、数据类型是否有decimal ，最好先统一小数点后几位、因为后面的小数点可能会影响模型的最终得份

2、groupby 时，注意dropna = False

3、做特征衍生时、不仅仅可以用常见的方式，决策树也可以用来做特征衍生
