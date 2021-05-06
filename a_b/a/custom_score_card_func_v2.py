import pandas as pd
import warnings;warnings.filterwarnings('ignore')
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
from scipy.stats import norm#作distplot的fit参数取值，拟合正态分布
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

'''
数据集缺失情况概览：仅return有缺失的变量信息
'NullOrNot'：是否缺失，True有缺失，False无缺失；'CntNotNull'：非缺失样本数；
'CntNull'：缺失样本数；'PntNull'：缺失率；
'''
def view_null(df,cols=None):
    if cols==None:cols=list(df.columns)
    df0=df[cols];m=df0.shape[0]
    TFs=[];cntNotNulls=[];cntNulls=[];pntNulls=[]
    dict0={}
    for var in cols:
        TF=df0[var].isnull().any();TFs.append(TF)
        cntNotNull=df0[var].count();cntNotNulls.append(cntNotNull)
        cntNull=(df0[var].isnull()).sum();cntNulls.append(cntNull)
        pntNull=cntNull/m;pntNulls.append(pntNull)
    df_null=pd.DataFrame(np.array([TFs,cntNotNulls,cntNulls,pntNulls]).T,index=cols,
                        columns=['NullOrNot','CntNotNull','CntNull','PntNull'])
    df_null['NullOrNot']=df_null['NullOrNot'].map({1:True,0:False})
    return df_null[df_null.PntNull>0].sort_values(by='PntNull',ascending=False)

    
#图示离散变量的分布，无返回值；
#df：DataFrame，数据集；cols：list，离散变量名；col_target：str，目标变量，y；
def plot_discrete(data,cols,col_target):
    cnt=len(cols)
    fig1152,ax1152=plt.subplots(cnt,2,figsize=(2*6,cnt*4))
    #变量取值分布
    for i in range(len(cols)):
        df1152=data[cols[i]].value_counts(dropna=False)
        df1152_pnt=100*df1152/df1152.sum()
        df1152.plot(kind='barh',ax=ax1152[i,0])
        ax1152set=ax1152[i,0].set(title=cols[i])
        #图注释，例如：975—3.09%，表示某变量取值下样本数975，占所有样本总数和的比例3.09
        for j in range(len(df1152)):
            ax1152text=ax1152[i,0].text(df1152.iloc[j],j,'%s—%s%%'%(df1152.iloc[j],round(df1152_pnt.iloc[j],2)))
    #双变量分布：目标变量按单个变量分组后，组内不同类别样本占比
        df1503=pd.crosstab(data[cols[i]],data[col_target],normalize='index',dropna=False)
        df1503.plot(kind='barh',ax=ax1152[i,1],stacked=True)
        ax1503set=ax1152[i,1].set(title='%s vs loan_status'%cols[i])
        #图注释，例如：80%，20%，表示变量某取值下，样本分为两个不同类别(1和0)，分别占比80,20%，
        for k in range(len(df1503)):
            ax1503set=ax1152[i,1].text(df1503.iloc[k,0],k,'%s%%'%round(100*df1503.iloc[k,0],2),ha='right')
            ax1503set=ax1152[i,1].text(1,k,'%s%%'%round(100*df1503.iloc[k,1],2),ha='center')
    #自动调整子图布局
    plt.tight_layout()
    
#图示连续变量的分布：箱线图、密度曲线、拟合正态分布曲线，无返回值；
#df：DataFrame，数据集；cols：list，要分析的连续变量；col_target：str，目标变量名；
def plot_continue(data,cols,col_target):
    cnt=len(cols)
    fig1733,ax1733=plt.subplots(cnt+1,4,figsize=(4*3,4*(cnt+1)))#cnt=1时，子图布局一维，ax[1]调用，ax[1,0]会报错
    for i in range(cnt):
        #连续变量绘制箱线图
        id_notnull=data[cols[i]].notnull()
        data[cols[i]][id_notnull].plot(kind='box',ax=ax1733[i,0])
        #连续变量绘制分组箱线图
        data[[cols[i],col_target]][id_notnull].boxplot(by=col_target,ax=ax1733[i,1])
        box_set2=ax1733[i,1].set(xlabel='%s grouped by %s'%(cols[i],col_target),title='')
        #连续变量绘制密度曲线，拟合正态分布曲线
        sns.distplot(data[cols[i]][id_notnull],hist=True,fit=norm,ax=ax1733[i,2])
        #连续变量按目标变量取值分组后，分别绘制密度图，fit=norm需要from scipy.stats import norm
        sns.distplot(data[cols[i]][id_notnull][data[col_target]==1],hist=True,fit=norm,ax=ax1733[i,3],label='1')
        sns.distplot(data[cols[i]][id_notnull][data[col_target]==0],hist=True,fit=norm,ax=ax1733[i,3],label='0')
        ax1733[i,3].legend(loc='best')
    plt.suptitle('')#取消居中标题
    plt.tight_layout()#自动子图调整布局


#统计变量不同取值下的样本总数、坏样本数、坏样本率，返回DataFrame统计结果；
#df:DataFrame,数据集；var：str，变量名；label：str，目标变量名;
def bin_badrate(df,var,label):
    total=df.groupby(var)[label].count()
    bad=df.groupby(var)[label].sum()
    regroup=pd.DataFrame({'total':total,'bad':bad})
    regroup['badrate']=regroup.apply(lambda x:x.bad/x.total,axis=1)
    return regroup

'''
离散变量分箱时，更新原始取值与最新箱标记的对应关系，返回dict，形如{变量原始取值：最新箱标记}；
dict_early：dict，更早的分箱规则；dict_late：dict，最新分箱规则；前者的值是后者的键；
例如：变量home_ownership，传入dict_early={'OWN':0,'NONE':1}，dict_late={0:0,1:0}，则返回{'OWN':0,'NONE':0}
'''
def refresh_val_2bin(dict_early,dict_late):
    dict_return={}
    for k,v in dict_early.items():
        if v not in dict_late:dict_return[k]=dict_early[k]
        else:dict_return[k]=dict_late[v]
    return dict_return


'''
计算并返回两个相邻组的卡方值；
regroup_sub：DataFrame，两个相邻组，形如函数bin_badrate的返回结果，举个栗子：
df1530=bin_badrate(train_data,'home_ownership','y')[['total','bad']]
df2151=df1530.iloc[1:3]，只选取df1530的相邻两组赋值参数；
当相邻两组组坏样本都为0（或1）时，over_badrate为0（或1），期望坏样本数为0（或1），实际与期望一样，卡方值为0；
连续型变量多状态取值合并至少状态取值时，每次都会比较所有相邻组的卡方值，使用list而非DataFrame，大大加快速度。
'''
def chi(regroup_sub):
    total=list(regroup_sub.total)
    bad=list(regroup_sub.bad);over_badrate=sum(bad)/sum(total)
    bad_except=[i*over_badrate for i in total]
    if over_badrate==0:chi_bad=[0,0]
    else:chi_bad=[(a-e)**2/e for a,e in zip(bad,bad_except)]
    
    good=[t-b for t,b in zip(total,bad)];over_goodrate=sum(good)/sum(total)
    good_except=[i*over_goodrate for i in total]
    if over_goodrate==0:chi_good=[0,0]
    else:chi_good=[(a-e)**2/e for a,e in zip(good,good_except)]
    return sum(chi_bad+chi_good)

'''
给定准确索引或索引范围，计算出最小卡方值的相邻两组的索引为(id_min,id_min+1)，返回id_min+1；
regroup：DataFrame，bin_badrate的结果，并按照期望方式（坏样本率或索引）排序过，本函数内不排序；
field='all'：默认比较所有相邻组，可选取值范围[ 0,regroup.shape[0]-1 ]；
示例：
field='all'：具有最小卡方值的相组索引为(id_min,id_min+1)，则返回更大索引id_min+1；
field=0，只能与field=1合并，则返回更大索引1；
field=regroup.shape[0]-1，只能与field=regroup.shape[0]-2合并，则返回更大索引regroup.shape[0]-1；
field=k，k属于[ 1,regroup.shape[0]-2 ]，若与field=k-1的卡方值更小，则返回更大索引k，
若与field=k+1的卡方值之和更小，则返回更大索引k+1；
'''
def chisum_min_id(regroup,field='all'):
    cnt=regroup.shape[0]
    #第一组只能与第二组合并，无需比较卡方值
    if field==0:id_min=1
    #最后一组只能与倒数第二组合并，无需比较卡方值
    elif field==cnt-1:id_min=cnt-1
    #最小卡方值对应应相邻组索引为(k,k+1)，则id_min=k+1
    elif field=='all':
        chis=[]
        for i in range(cnt-1):
            chi_val=chi(regroup.iloc[i:i+2]);chis.append(chi_val)
        id_min=chis.index(min(chis))+1
    #chi(k-1,k)<=chi(k,k+1)，则id_min=k，否则id_min=k+1
    else:
        chi_pre=chi(regroup.iloc[field-1:field+1])
        chi_later=chi(regroup.iloc[field:field+2])
        if chi_pre<=chi_later:id_min=field
        else:id_min=field+1
    return id_min

'''
函数设计背景：变量分箱过程中，无论何种原因检测到某两组需要合并，需要遵守一些规则：
1、离散无序取值种类少变量，合并的相邻组其坏样本率应相邻；
2、离散有序变量，合并的相邻组不能破坏其有序性，例如学历变量，检测到初中组需要合并，那么存在小学组、高中组的情况下，
  它应当与博士组合并；
3、连续型变量，合并的相邻组取值大小应当相近（本函数内不考虑连续型变量分组与合并）；

函数设计思路：
1、传入regroup0,m行(也就是m组)，离散无序变量取值种类少以badrate升排序，离散有序变量以索引升排序；
2、检测到需要合并的组索引序值为k；
3、将排序后的regroup0和k传入函数chisum_min_id，进一步确定需要合并的相邻两组索引序值，例如(k,k+1)；
4、返回排序后的regroup0，k+1；(返回值会作为合并相邻组函数的参数使用)

注：索引序值，索引升排序后表示索引排列位置的整数值；索引不一定是整数，但索引序值一定是整数，0为起始序值；

参数说明：
regroup0:DataFrame，bin_badrate的结果；
var_order=False：bool，变量取值是否有序，默认无序；（主函数bin_discrete中该参数是一个dict，原因是那里还需要
  知道有序细节，而这里只需知道有序否，故用bool）
check_object：str,检测对象，默认'badrarte'(针对badrate为0或1的情况)，可选取值：
  'min_pnt'，针对某组样本总数占所有组样本总数的比例是否大于等于期望最小比例的情况；
  'chisum_min'：针对需要找到卡方值最小相邻组的情况；
  var_order=False：变量取值是否有序，默认False无序
'''
def order_regroup(regroup0,var_order=False,check_object='badrate'):
    regroup=regroup0.copy()
    #离散无序：badrate升排序
    if var_order==False:
        regroup.sort_values(by='badrate',ascending=True,inplace=True)
    #离散有序：索引升排序
    if var_order==True:
        regroup.sort_index(ascending=True,inplace=True) 
    #针对check_object不同取值的处理
    if check_object=='badrate':
        id_badrate_min=list(regroup.index).index(regroup.badrate.idxmin())
        id_badrate0=chisum_min_id(regroup,field=id_badrate_min)

        id_badrate_max=list(regroup.index).index(regroup.badrate.idxmax())
        id_badrate1=chisum_min_id(regroup,field=id_badrate_max)
        return regroup,id_badrate0,id_badrate1

    if check_object=='min_pnt':
        id_min_pnt=list(regroup.index).index(regroup.total_pnt.idxmin())#total_pnt在函数外计算
        id_pnt=chisum_min_id(regroup,field=id_min_pnt)
        return regroup,id_pnt

    if check_object=='chisum_min':
        id_merge=chisum_min_id(regroup,field='all')
        return regroup,id_merge

'''
合并相邻组；
regroup，id_target，来自函数order_regroup的返回结果；
dict_map：dict，分组规则，形如{变量原始取值：最新箱标记}；

注：数值而非字符串，如'bin'+数值，作箱标记的原因：离散有序变量以索引(箱标记)排序，若以'bin'+数值形式，
    则会出现'bin1','bin11','bin2'，这是字符串和数值升排序的各自规则决定的，然而升排序时'bin11'应排在
    'bin2'之后的，故直接采用纯数值作箱标记。
'''
def merge_neighbor(regroup,id_target,dict_map):
    #首先重置索引再refresh_val_2bin以确保一一对应；也便于发起合并组为中间某组时修改组标记
    regroup=regroup.reset_index()
    dict_reindex={v:k for k,v in regroup[regroup.columns[0]].to_dict().items()}
    #重置索引后，合并后分组规则dict键与dict_map不再一致，故这里需要先更新
    return_dict=refresh_val_2bin(dict_map,dict_reindex)
    id_all=list(regroup.index)
    m=regroup.shape[0];dict_merge={}
    #发起合并组为第一组，则与第二组合并
    if id_target==1:
        dict_merge[0]=0;dict_merge[1]=0
        for i in range(2,m):dict_merge[i]=i-1
    #发起合并组为最后一组，则与倒数第二组合并（注意合并后箱标记维持索引原序）
    elif id_target==m-1:
        dict_merge[m-1]=m-2;dict_merge[m-2]=m-2
        for i in range(m-2):dict_merge[i]=i
    #发起合并组为中间某组，在其前后组中选择合并后卡方值更小的那一组进行合并
    #特别注意：离散有序变量检查badrate为0或1时，只能在发起合并组前或后一组选择一组进行合并
    else:
        lab_bin=id_all[:id_target]
        lab_bin.extend([i-1 for i in id_all[id_target:]])
        dict_merge={i:lab_bin[i] for i in range(len(lab_bin))}
    #合并后更新变量取值分组规则
    return_dict=refresh_val_2bin(return_dict,dict_merge)
    return return_dict

'''
作用：
判断组间坏样本率期望单调性（不严格，允许相邻组badrate相等）与实际单调性是否相符；不符时是否放宽
至U形（严格，不允许相邻组badrate相等），正U，开口向上，
倒U，开口向下；U形时，极值索引不一定是索引中位数，可能存在偏移；符合期望单调或U型，则返回True，否则返回False；

参数：
regroup：函数bin_badrate执行结果；
shape='mono'：期望单调性，默认单调增或减，可选'mono_up'单调升，'mono_down'单调减；
u=False：是否允许U形，默认False不允许，可选'u'正U或倒U、u_up'正U、'u_down'倒U；
'''
def monotone_badrate(regroup,shape='mono',u=False):
    regroup=regroup.sort_index(ascending=True)
    badrate=list(regroup.badrate)
    cnt=len(badrate)
    if cnt<=1:print('%s组无所谓单调性'%cnt);return True
    #单调，计算：up=True不严格单调增，down=True不严格单调减；
    up=all([badrate[i] <= badrate[i+1] for i in range(cnt-1)])
    down=all([badrate[i] >= badrate[i+1] for i in range(cnt-1)])
    #期望单调性与实际单调性相符则mono_return=True，否则mono_return=False
    if shape=='mono_up':mono_return=up
    elif shape=='mono_down':mono_return=down
    else:mono_return=any([up,down])
    ###########################################################################
    #已判断为非单调，但允许U形，则进一步判断是否U形
    if mono_return==False and u!=False:
        #bool_strict=False，表示存在至少一对相邻数值相等，则这组数值即便成U形，但不会严格。
        bool_strict=1-any([badrate[i]==badrate[i+1] for i in range(cnt-1)])
        if bool_strict==False:return mono_return
        id_min=badrate.index(min(badrate));id_max=badrate.index(max(badrate))
        #倒U形：极大值索引id_max不在首或尾，且id_max左边单调增，右边单调减
        if id_max not in [0,cnt-1]:
            up_left=[badrate[i]<=badrate[i+1] for i in range(id_max)]
            down_right=[badrate[i]>=badrate[i+1] for i in range(id_max,cnt-1)]
            U_down=all(up_left+down_right)
        #不构成倒U形：极大值索引在首或尾
        else:U_down=False
        #正U形：极小值索引id_min不在首或尾，且id_min左边单调减，右边单调增
        if id_min not in [0,cnt-1]:
            down_left=[badrate[i]>=badrate[i+1] for i in range(id_min)]
            up_right=[badrate[i]<=badrate[i+1] for i in range(id_min,cnt-1)]
            U_up=all(down_left+up_right)
        #不构成正U形：极小值索引在首或尾
        else:U_up=False
        #期望单调性与实际单调性相符则True，否则False
        if u=='u_up':mono_return=U_up
        elif u=='u_down':mono_return=U_down
        else:mono_return=any([U_up,U_down])
    return mono_return


'''
卡方分箱过程中，有时可能希望某些取值不参与分组与合并，而是始终自成一组，但这些有可能坏样本率为0或1，
组样本数比例(total_pnt)低于期望最低比例(min_pnt)，此时需要调整，然后与正常分组的取值跨行拼接再计算woe，iv。
调整方式（这只是为了程序能正常运行，不清楚真实业务场景应如何处理）：badrate为0或1时组内调整；
total_pnt<min_pnt时不调整但提示；

#regroup_nor：DataFrame，参与分组的变量取值执行函数bin_badrate的结果；
#regroup_special:DataFrame，不参与分组的变量取值执行函数bin_badrate的结果；
#min_pnt=0.05：组样本数占所有组样本总数的期望最小值，默认0.05；
'''
def regroup_special_merge(regroup_nor,regroup_special,min_pnt=0.05):
    #badrate=0或1时，组内调整至非0或1
    while regroup_special.badrate.min()==0:
        id0=regroup_special.badrate.idxmin()
        regroup_special.loc[id0,'bad']=1
        regroup_special['badrate']=regroup_special.bad/regroup_special.total
    while regroup_special.badrate.max()==1:
        id1=regroup_special.badrate.idxmax()
        regroup_special.loc[id1,'bad'] -= 1
        regroup_special['badrate']=regroup_special.bad/regroup_special.total
    if min(regroup_special['total_pnt'])<min_pnt:
        print('\n变量：%s，不参与分组的取值单独自组，存在组样本数占比小于期望最低比例 %s ，未调整。\n%s'%(
            regroup_special.index.name,min_pnt,regroup_special))
    regroup_nor.index.name=regroup_special.index.name
    regroup_turn=pd.concat([regroup_nor,regroup_special],axis=0).drop(columns='total_pnt')
    return regroup_turn


'''
计算并返回变量iv，变量取值分组下woe；
参数：regroup:DataFrame，函数bin_badrate结果
'''
def calc_woe_iv(regroup):
    regroup['good']=regroup.total-regroup.bad
    regroup['good_pnt']=regroup.good/regroup.good.sum()
    regroup['bad_pnt']=regroup.bad/regroup.bad.sum()
    regroup['woe']=np.log(regroup.bad_pnt/regroup.good_pnt)
    regroup['iv']=(regroup.bad_pnt-regroup.good_pnt)*regroup.woe
    dict_woe=regroup.woe.to_dict();iv_sum=regroup.iv.sum()
    return dict_woe,iv_sum

'''
主函数：
针对离散变量取值种类少、离散有序变量的卡方分箱主函数bin_discrete，变量取值分组后，可实现：
1、任意一组badrate不为0或1；
2、任意一组total_pnt不低于期望最低比例min_pnt（min_pnt可由人为指定，不参与分组的取值单独成组的情况，
   其total_pnt可能小于min_pnt）；
3、实现期望组间badrate单调，期望U形；

badrate：组内坏样本数量占组内样本总数的比例；
total_pnt：组内样本总数占所有组样本总数的比例；

参数：
df0：DataFrame，训练集；
var_list：list，[变量名1,...,变量名n]；离散有序变量需要根据{变量取值：序值}关系，构建新变量，且新变量名
    需要带'_order'后缀，；
y='y'：str，目标变量名，默认'y';
special_val={}：dict，不参与分组的特殊取值，
    形如{ 变量名1:[特殊值1,...,特殊值n] ,..., 变量名k:[特殊值1,...,特殊值n] }；
intervals_max=5：int，期望最大分组数（含），默认5；
min_pnt=0.05：float，任意一组样本数占所有组样本总数比例的期望最小值（含），默认0.05；
discrete_order={}：dict，默认空表变量离散无序；非空dict表离散有序，
    形如{ 变量名1:{变量值1:序值1,...,变量值n:序值n} ,..., 变量名k:{变量值1:序值1,...,变量值n:序值n} }；
mono_except={}：dict，默认空表变量离散无序，无需检查单调性；非空dict表离散有序，需要检查badrate单调性，参数赋值
形如{ 变量名1:{'shape':期望单调性,'u':不单调时，是否允许U形} ,..., 变量名k:{'shape':期望单调性,'u':不单调时，是否允许U形}}，
    'shape'可选择参数：'mono'期望单调增或减，'mono_up'期望单调增，'mono_down'期望单调减；
    'u'可选参数：'u'正U（开口向上）或倒U（开口向下），'u_up'正U，'u_down'倒U。
print_process=False：bool，是否展示卡方分箱过程，默认False不展示；


不同特殊值组间、特殊值组与正常值组间不考虑单调性的问题；
若出现坏样本率为0或1时，样本数调增或调减1再重新计算坏样本率，理由：woe编码计算式为woe=ln(badrate/goodrate)），
分子分母都不能为0；badrate=0或1，表示的是组内好坏样本数量的差异达到了极限，为了使woe能正常计算，做最低程度调整，
以维持组内好坏样本数量的差异极大这一事实，具体为：
bad_cnt + good_cnt = total_cnt，
badrate=bad_cnt/total_cnt=0时，即bad_cnt=0，先调整bad_cnt=1，再计算badrate；
goodrate=good_cnt/total_cnt=0时，即bad_cnt=total_cnt，bad_cnt=bad_cnt-1，再计算badrate_rate；
组样本数占所有组样本数比例低于，期望最小比例时，仍然单独自成一组，同时print提示；
'''
def bin_discrete(df0,var_list,y='y',special_val={},intervals_max=8,min_pnt=0.05,discrete_order={},mono_except={},print_process=False):
    #当前变量不参与分组的取值无论是否存在，检查组样本数占比时，都应当是基于完整数据集样本总数；然而，当存在不参与
    #分组的取值时，这一环检查的数据集是完整数据集的子集，样本总数被减少了。故这里用samples_cnt先固定下来。
    samples_cnt=df0.shape[0]
    discrete_iv={};var_woe=[];discrete_woe={}#返回变量的iv、编码后变量名、各取值下的woe编码
    for var in var_list:
        df=df0[[var,y]]
        max_intervals=intervals_max
        var_val_2bin={i:i for i in set(df[var])}#变量取值及其分箱规则初始化
        #######################################################################################
        if var in special_val and len(special_val[var])>0:
            max_intervals=intervals_max-len(special_val[var])
            df_special=df[df[var].isin(special_val[var])]#先于参与分组的样本df执行
            df=df[~df[var].isin(special_val[var])]
        #######################################################################################
        df[var+'_init']=df[var].copy()
        regroup_d_init=bin_badrate(df,var+'_init',y)
        if var in discrete_order:
            while regroup_d_init.shape[0]>max_intervals:
                regroup_d_init,id_merge_init=order_regroup(
                    regroup_d_init,var_order=True,check_object='chisum_min')
                #*****************************************************************************
                if print_process==True:
                    print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
                    show_process(regroup_d_init,check_object='init')
                #*****************************************************************************
                var_val_2bin=merge_neighbor(regroup_d_init,id_merge_init,var_val_2bin)
                df[var+'_init']=df[var].map(var_val_2bin)
                regroup_d_init=bin_badrate(df,var+'_init',y)
            #*****************************************************************************
            if print_process==True:
                print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
                show_process(regroup_d_init,check_object='init')
            #*****************************************************************************
        ######################################################################
        #检查badrate=0或1，同时存在badrate=0或1时，先处理badrate=1
        df[var+'_badrate']=df[var+'_init'].copy()
        regroup_d_badrate=bin_badrate(df,var+'_badrate',y)
        while regroup_d_badrate['badrate'].min()==0 or regroup_d_badrate['badrate'].max()==1:
            #离散有序：
            if var in discrete_order:
                regroup_d_badrate,id_merge_badrate0,id_merge_badrate1=order_regroup(
                    regroup_d_badrate,var_order=True,check_object='badrate')
            #离散无序:
            else:regroup_d_badrate,id_merge_badrate0,id_merge_badrate1=order_regroup(
                    regroup_d_badrate,var_order=False,check_object='badrate')
            if regroup_d_badrate['badrate'].min()==0:id_need_merge=id_merge_badrate0
            if regroup_d_badrate['badrate'].max()==1:id_need_merge=id_merge_badrate1
            #*****************************************************************************
            if print_process==True:
                print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
                show_process(regroup_d_badrate,check_object='badrate')
            #*****************************************************************************
            #确定regroup_d_badrate排序规则和发起合并组索引后，更新分组规则，更新分组结果
            var_val_2bin=merge_neighbor(regroup_d_badrate,id_need_merge,var_val_2bin)
            df[var+'_badrate']=df[var].map(var_val_2bin)
            regroup_d_badrate=bin_badrate(df,var+'_badrate',y)
        #*****************************************************************************
        if print_process==True:
            print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
            show_process(regroup_d_badrate,check_object='badrate')
        #*****************************************************************************
        ######################################################################
        ######################################################################
        #检查min_pnt：使用badrate为0或1检查后的结果
        df[var+'_min_pnt']=df[var+'_badrate'].copy()
        regroup_d_min_pnt=bin_badrate(df,var+'_min_pnt',y)
        regroup_d_min_pnt['total_pnt']=regroup_d_min_pnt.total/samples_cnt
        while regroup_d_min_pnt['total_pnt'].min()<min_pnt:
            #离散有序：
            if var in discrete_order:
                regroup_d_min_pnt,id_merge_pnt=order_regroup(
                    regroup_d_min_pnt,var_order=True,check_object='min_pnt')
            #离散无序:
            else:regroup_d_min_pnt,id_merge_pnt=order_regroup(
                    regroup_d_min_pnt,var_order=False,check_object='min_pnt')
            #*****************************************************************************
            if print_process==True:
                print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
                show_process(regroup_d_min_pnt,check_object='min_pnt')
            #*****************************************************************************
            var_val_2bin=merge_neighbor(regroup_d_min_pnt,id_merge_pnt,var_val_2bin)
            df[var+'_min_pnt']=df[var].map(var_val_2bin)
            regroup_d_min_pnt=bin_badrate(df,var+'_min_pnt',y)
            regroup_d_min_pnt['total_pnt']=regroup_d_min_pnt.total/samples_cnt
        #计算woe，iv使用的regroup：无单调性检查regroup_d_min_pnt；有单调性检查regroup_d_mono
        #故这里copy一个副本，兼顾可能的两种情况
        regroup_woe_iv=regroup_d_min_pnt.copy()
        #*****************************************************************************
        if print_process==True:
            print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
            show_process(regroup_d_min_pnt,check_object='min_pnt')
        #*****************************************************************************
        ################################################################################
        ################################################################################
        #仅对需要单调性检查的离散有序变量执行
        if var in mono_except:
            df[var+'_mono']=df[var+'_min_pnt'].copy()
            regroup_d_mono=bin_badrate(df,var+'_mono',y)
            mono=monotone_badrate(regroup_d_mono,shape=mono_except[var]['shape'],
                                  u=mono_except[var]['u'])
            while not mono:
                regroup_d_mono,id_merge_mono=order_regroup(
                    regroup_d_mono,var_order=True,check_object='chisum_min')
                #*****************************************************************************
                if print_process==True:
                    print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
                    show_process(regroup_d_mono,check_object='mono')
                #*****************************************************************************
                var_val_2bin=merge_neighbor(regroup_d_mono,id_merge_mono,var_val_2bin)
                df[var+'_mono']=df[var].map(var_val_2bin)
                regroup_d_mono=bin_badrate(df,var+'_mono',y)
                mono=monotone_badrate(regroup_d_mono,shape=mono_except[var]['shape'],
                                      u=mono_except[var]['u'])
            regroup_woe_iv=regroup_d_mono.copy()
            #*****************************************************************************
            if print_process==True:
                print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
                show_process(regroup_d_mono,check_object='mono')
            #*****************************************************************************
        ################################################################################
        #当前变量若存在不参与分组的取值，则计算woe，iv前，regroup_woe_iv需要先把未参与分组的组加回来
        if var in special_val and len(special_val[var])>0:
            regroup_special=bin_badrate(df_special,var,y)
            regroup_special['total_pnt']=regroup_special.total/samples_cnt
            regroup_woe_iv=regroup_special_merge(regroup_woe_iv,regroup_special,min_pnt=min_pnt)
            if print_process==True:
                print('\n{原始取值：最新箱标记}\n%s\n'%var_val_2bin)
                show_process(regroup_woe_iv,check_object='special')
        ################################################################################
        #计算woe、iv并将其与变量取值对应起来
        dict_woe,iv=calc_woe_iv(regroup_woe_iv)#{分组标记：woe}
        var_val_2bin=refresh_val_2bin(var_val_2bin,dict_woe)#{变量取值：woe}不再是{变量取值：分组标记}
        #*****************************************************************************
        if print_process==True:
            print('\n{箱标记：woe}：\n%s'%dict_woe)
            print('\n{原始取值(或离散有序变量序值)：woe}：\n%s'%var_val_2bin)
            print('***************************************************************************************\n')
        #*****************************************************************************
        #离散有序变量var_val_2bin更新为{原变量取值:箱标记}
        if var[-6:]=='_order':
            if print_process==True:print('\n离散有序变量，{变量取值：序值}：\n%s'%discrete_order[var])
            var_val_2bin=refresh_val_2bin(discrete_order[var],var_val_2bin)
            var=var[:-6]
            if print_process==True:
                print('\n离散有序变量，最后{变量取值：woe}：\n%s'%var_val_2bin)
                print('***************************************************************************************\n')
        df0[var+'_woe']=df0[var].map(var_val_2bin)#df0而非df
        discrete_iv[var+'_woe']=iv;var_woe.append(var+'_woe');discrete_woe[var+'_woe']=var_val_2bin
    return discrete_woe,discrete_iv,var_woe


'''
连续变量取值种类数理论上无限，当前取值种类数大于100的先初始切分为100组。
将连续变量取值升排序；等频分为100组；取每组上边界（含）做分组标记；从最小组上边界开始，小于上边界值的映射为上边界值；
df：DataFrame，数据集；var：str，变量名：int，初始分组数，默认init_bins=100；
返回list，里面是不重复的组上边界值
'''
def init_split(df,var,init_bins=100):
    m=df.shape[0]
    cnt_per=math.ceil(m/init_bins)
    init_bins=math.ceil(m/cnt_per)
    vals=sorted(list(df[var]))
    id_thresh=[i*cnt_per-1 for i in range(1,init_bins)]
    thresh=sorted(list(set([round(vals[i],4) for i in id_thresh])))
    return [round(i,4) for i in thresh]
'''
小于组上边界的值映射为上边界值；
x：float或int，连续变量取值；thresh：list，组上边界值，无重复值。
将x与thresh内的值从小到大比较，找到thresh内第一个大于等于x的值，然后将x映射为这个值
'''
def refresh_val_2thresh(x,thresh):
    if x<=thresh[0]:return thresh[0]
    elif x>thresh[-1]:return 10e10
    else:
        for i in range(len(thresh)-1):
            if thresh[i]<x<=thresh[i+1]:return thresh[i+1]

#展示连续变量取值的分组过程
def show_process(regroup,check_object='badrate'):
    chis=[]
    for i in range(regroup.shape[0]-1):
        chi_val=chi(regroup.iloc[i:i+2])
        chis.append(chi_val)
    if check_object=='init':txt='当前组数大于期望最大分箱数max_intervals'
    elif check_object=='min_pnt':txt='组样本总数'
    elif check_object=='mono':txt='组间期望单调性'
    elif check_object=='special':txt='因存在取值未参与分组的调整'
    else:txt='组内坏样本率为0或1'
    print('变量：%s；当前环节：%s检查'%(regroup.index.name,txt))
    print('\nregroup：\n%s'%regroup)
    print('\n相邻组卡方值：\n%s'%(chis))
    print('***************************************************************************************\n')

#连续变量分箱主函数，函数逻辑和参数与bin_discrete总体一致
def bin_continue(df0,var_list,y='y',special_val={},intervals_max=8,min_pnt=0.05,mono_except={},print_process=False):
    samples_cnt=df0.shape[0]
    contin_iv={};var_woe=[];contin_woe={}
    for var in var_list:
        df=df0[[var,y]]
        max_intervals=intervals_max
        ###############################################################################################
        #若存在不参与分组的取值
        if var in special_val and len(special_val[var])>0:
            max_intervals=intervals_max-len(special_val[var])
            df_special=df[df[var].isin(special_val[var])]
            df=df[~df[var].isin(special_val[var])]
        ###############################################################################################
        #连续变量取值种类数大于100，先等频初始分割至100组以内
        df[var+'_init']=df[var].copy()
        if len(set(df[var+'_init']))>100:
            thresh_init=init_split(df,var+'_init',init_bins=100)
            df[var+'_init']=df[var+'_init'].map(lambda x:refresh_val_2thresh(x,thresh_init))
        ###############################################################################################
        #合并组数至max_intervals以内
        df[var+'_merge']=df[var+'_init'].copy()
        regroup_c_merge=bin_badrate(df,var+'_merge',y)
        while regroup_c_merge.shape[0]>max_intervals:
            regroup_c_merge,id_del_merge=order_regroup(regroup_c_merge,var_order=True,check_object='chisum_min')
            thresh_merge=list(regroup_c_merge.index)
            del thresh_merge[id_del_merge-1]
            df[var+'_merge']=df[var+'_merge'].map(lambda x:refresh_val_2thresh(x,thresh_merge))
            regroup_c_merge=bin_badrate(df,var+'_merge',y)
        ###############################################################################################
        #检查badrate=0或1，同时存在badrate=0或1时，先处理badrate=1
        df[var+'_badrate']=df[var+'_merge'].copy()
        regroup_c_badrate=bin_badrate(df,var+'_badrate',y)
        while regroup_c_badrate.badrate.min()==0 or regroup_c_badrate.badrate.max()==1:
            regroup_c_badrate,id_merge_badrate0,id_merge_badrate1=order_regroup(
                    regroup_c_badrate,var_order=True,check_object='badrate')
            thresh_badrate=list(regroup_c_badrate.index)
            if regroup_c_badrate.badrate.min()==0:id_del_badrate=id_merge_badrate0
            if regroup_c_badrate.badrate.max()==1:id_del_badrate=id_merge_badrate1
            #***************************************************************************************
            if print_process==True:show_process(regroup_c_badrate,check_object='badrate')
            #***************************************************************************************
            del thresh_badrate[id_del_badrate-1]
            df[var+'_badrate']=df[var+'_badrate'].map(lambda x:refresh_val_2thresh(x,thresh_badrate))
            regroup_c_badrate=bin_badrate(df,var+'_badrate',y)
        #***************************************************************************************
        if print_process==True:show_process(regroup_c_badrate,check_object='badrate')
        #***************************************************************************************
        ###############################################################################################
        #检查组total_pnt：使用badrate为0或1检查后的结果
        df[var+'_min_pnt']=df[var+'_badrate'].copy()
        regroup_c_minpnt=bin_badrate(df,var+'_min_pnt',y)
        regroup_c_minpnt['total_pnt']=regroup_c_minpnt.total/samples_cnt
        while regroup_c_minpnt['total_pnt'].min()<min_pnt:
            if regroup_c_minpnt.shape[0]==2:break#余两组，将合并为一组，不合并，min_pnt提示，且不计算woe
            regroup_c_minpnt,id_del_pnt=order_regroup(regroup_c_minpnt,var_order=True,check_object='min_pnt')
            thresh_minpnt=list(regroup_c_minpnt.index)
            #***************************************************************************************
            if print_process==True:show_process(regroup_c_minpnt,check_object='min_pnt')
            #***************************************************************************************
            del thresh_minpnt[id_del_pnt-1]
            df[var+'_min_pnt']=df[var+'_min_pnt'].map(lambda x:refresh_val_2thresh(x,thresh_minpnt))
            regroup_c_minpnt=bin_badrate(df,var+'_min_pnt',y)
            regroup_c_minpnt['total_pnt']=regroup_c_minpnt.total/samples_cnt
        #***************************************************************************************
        if print_process==True:show_process(regroup_c_minpnt,check_object='min_pnt')
        #***************************************************************************************
        ############################################################################################### 
        #单调性检查
        df[var+'_mono']=df[var+'_min_pnt'].copy()
        regroup_c_mono=bin_badrate(df,var+'_mono',y)
        mono=monotone_badrate(regroup_c_mono,shape=mono_except[var]['shape'],u=mono_except[var]['u'])
        while not mono:
            regroup_c_mono,id_del_mono=order_regroup(regroup_c_mono,var_order=True,check_object='chisum_min')
            thresh_mono=list(regroup_c_mono.index)
            #***************************************************************************************
            if print_process==True:show_process(regroup_c_mono,check_object='mono')
            #***************************************************************************************
            del thresh_mono[id_del_mono-1]
            df[var+'_mono']=df[var+'_mono'].map(lambda x:refresh_val_2thresh(x,thresh_mono))
            regroup_c_mono=bin_badrate(df,var+'_mono',y)
            mono=monotone_badrate(regroup_c_mono,shape=mono_except[var]['shape'],u=mono_except[var]['u'])
        #***************************************************************************************
        if print_process==True:show_process(regroup_c_mono,check_object='mono')
        #***************************************************************************************
        regroup_woe_iv=regroup_c_mono.copy()
        ############################################################################################### 
        #当前变量若存在不参与分组的取值，则计算woe，iv前，regroup_woe_iv需要先把未参与分组的组加回来
        if var in special_val and len(special_val[var])>0:
            regroup_special=bin_badrate(df_special,var,y)
            regroup_special['total_pnt']=regroup_special.total/samples_cnt
            regroup_woe_iv=regroup_special_merge(regroup_c_mono,regroup_special,min_pnt=min_pnt)
            #***************************************************************************************
            if print_process==True:show_process(regroup_woe_iv,check_object='special')
            #***************************************************************************************
        ###############################################################################################
        #计算woe，iv
        dict_woe,iv=calc_woe_iv(regroup_woe_iv)
        thresh_end=sorted(list(regroup_woe_iv.index))
        #***************************************************************************************
        if print_process==True:
            print('\n变量：%s，{最终分组上界：woe}：\n%s'%(var,dict_woe))
            print('***************************************************************************************\n')
        #***************************************************************************************
        df0[var+'_woe']=df0[var].map(lambda x:refresh_val_2thresh(x,thresh_end)).map(dict_woe)
        contin_iv[var+'_woe']=iv;var_woe.append(var+'_woe');contin_woe[var+'_woe']=dict_woe
    return contin_woe,contin_iv,var_woe


'''
同时存在1个及以上变量有多重共线性，逐一剔除，以排除多重共线性（需要逐二剔除的情况仅print提示）
df：DataFrame，数据集；
feat：list，变量两两间相关系数小于0.7，且按iv降排序；
vif_thresh=10：int或float，VIF阈值，当VIF>=vif_thresh，认定存在多重共线性，默认取值10；
print_process=False：bool，是否print展示过程，默认False，不展示;
'''
def check_vif(df,feat,vif_thresh=10,print_process=False):
    feat_desc_iv=feat.copy()#可能会从feat内删除变量，故copy副本，保持feat参数不变
    cnt_feat2244=len(feat_desc_iv)
    x_vif=np.array(df[feat_desc_iv])
    vif_all=[variance_inflation_factor(x_vif,i) for i in range(cnt_feat2244)]
    
    #检测出存在多重共线性的变量索引，并索引升排序
    id_high_vif=sorted([i for i in range(cnt_feat2244) if vif_all[i]>=vif_thresh])
    #无多重共线性，返回原输入变量
    if len(id_high_vif)==0:print('无多重共线性');return feat_desc_iv.copy()
    
    ############################以下存在多重共线性###################################################
    if print_process==True:print('\n下面是存在多重共线性的变量索引,首先处理iv最低变量,即变量以iv降排序下索引最大：\n%s,'%id_high_vif)
    while len(id_high_vif)>0:
        #按iv降排序下，vif大于阈值的变量的索引中，最大索引的iv最小，最先针对其排除多重共线性
        id_check=id_high_vif[-1]
        #逐一剔除的变量的id，首先从iv最小变量开始，即iv降排序下最后一个变量
        id_del=cnt_feat2244-1
        while id_del>=0:
            var_x=[feat_desc_iv[i] for i in range(cnt_feat2244) if i != id_del]#不含要剔除的变量
            x_vif2244=np.array(df[var_x])
            vif_new=variance_inflation_factor(x_vif2244,id_check)
            
            #剔除iv值低的变量后，变量id_check的vif仍大于等于阈值，则剔除iv更大的变量，即id_del-1
            if vif_new>=vif_thresh:
                if print_process==True:print('因变量(%s，%s)，剔除自变量(%s，%s)后，VIF=%s>=%s，继续'%(
                        id_check,feat_desc_iv[id_check],id_del,feat_desc_iv[id_del],vif_new,vif_thresh))
                id_del-=1
                if id_del==id_check:id_del-=1
                continue
                
            #剔除iv值低的变量后，变量id_check的vif小于阈值，则剔除iv更小的变量，min(iv(id_check),iv(id_del))
            else:
                if iv_high[feat_desc_iv[id_del]]<=iv_high[feat_desc_iv[id_check]]:id_del_end=id_del
                else:id_del_end=id_check
                if print_process==True:print('因变量(%s，%s)，剔除自变量(%s，%s)后，VIF=%s<%s;\n剔除变量(%s，%s)\n'%(
                                            id_check,feat_desc_iv[id_check],id_del,feat_desc_iv[id_del],
                                            vif_new,vif_thresh,id_del_end,feat_desc_iv[id_del_end]))
                break
            
        #变量id_check，逐一剔除所有其他自变量,不能排除多重共线性，接下来尝试其他VIF大于等于阈值的变量
        if id_del<0:
            if print_process==True:print('因变量(%s，%s),逐一剔除所有其他自变量,不能排除多重共线性'%(
                                            id_check,feat_desc_iv[id_check]))
            del id_high_vif[-1]
            continue
        #剔除变量id_del_end
        del feat_desc_iv[id_del_end]
        
        #每剔除一个变量后，重新检测所有变量的多重共线性
        cnt_feat2244=len(feat_desc_iv)
        x_vif=np.array(df[feat_desc_iv])
        vif_all=[variance_inflation_factor(x_vif,i) for i in range(cnt_feat2244)]
    
        #检测出存在多重共线性的变量索引
        id_high_vif=[i for i in range(cnt_feat2244) if vif_all[i]>=vif_thresh]
        if print_process==True:print('\n下面是存在多重共线性的变量索引,首先处理iv最低变量,即变量以iv降排序下索引最大：\n%s,'%id_high_vif)
    #返回结果  
    if vif_new>=vif_thresh:print('逐一剔除不能排除多重共线性，需要逐二剔除')#尚未遭遇，故无处理代码
    else:print('\n已排除多重共线性')
    return feat_desc_iv.copy()




'''
模型区分度评价指标ks=max(cum_pnt_bad - cum_pnt_good)或ks=max(tpr - fpr)
样本为正例（当前项目正例为坏样本）的概率降排序，计算cum_pnt_bad（累计坏样本数占坏样本总数的比例）；
cum_pnt_good（累计好样本数占好样本总数的比例；
lab_true：一维list、array、series，样本真实标签y；lab_pred：一维list、array、series，样本为正例的概率；
plot_ks=False：bool，是否绘图，默认不绘制。
返回float的ks值
'''
def distin_ks(lab_true,lab_pred,plot_ks=False):
    cnt_all=len(lab_true)
    cnt_stats=lab_true.value_counts().to_dict()
    cnt_bad,cnt_good=cnt_stats[1],cnt_stats[0]
    df_ks=pd.DataFrame({'lab':lab_true,'proba':lab_pred})
    df_ks=df_ks.sort_values(by='proba',ascending=False)
    df_ks['cum_pnt_all']=[i/cnt_all for i in list(range(1,cnt_all+1))]
    df_ks['cum_pnt_bad']=df_ks.lab.cumsum()/cnt_bad
    df_ks['cum_pnt_good']=(df_ks.lab==0).cumsum()/cnt_good
    df_ks['diff']=df_ks['cum_pnt_bad']-df_ks['cum_pnt_good']
    ks=max(df_ks['diff'])
    if plot_ks==True:
        fig1947,ax1947=plt.subplots()
        ax1947.plot(df_ks['cum_pnt_all'],df_ks['cum_pnt_bad'],label='cum_pnt_bad')
        ax1947.plot(df_ks['cum_pnt_all'],df_ks['cum_pnt_good'],label='cum_pnt_good')
        ax1947.plot(df_ks['cum_pnt_all'],df_ks['diff'],label='diff')
        ax1947.set(title='ks=%s'%round(ks,4),xlabel='cum_pnt_all',ylabel='cum_pnt')
        ax1947.legend(loc='best')
    return ks

#ks=max(tpr - fpr)，参数与istin_ks一致
def distin_ks_roc(lab,proba,plot_roc=False):
    fpr,tpr,thresh=roc_curve(lab,proba)
    diff=[t-f for f,t in zip(fpr,tpr)]
    ks_roc=max(diff)
    id_max_diff=diff.index(ks_roc)
    thresh_best=round(thresh[id_max_diff],4)
    if plot_roc==True:
        fig2207,ax2207=plt.subplots()
        ax2207.plot(thresh,fpr,label='fpr')
        ax2207.plot(thresh,tpr,label='tpr')
        ax2207.plot(thresh,diff,label='diff')
        ax2207.set(title='ks=%s,thresh_best=%s'%(round(ks_roc,4),thresh_best),xlabel='thresh',ylabel='fpr or tpr')
    return ks_roc

'''
模型区分度divergence=(u_bad - u_good)**2 / (0.5*(var_bad + var_good))，综合考虑两个集合的中心距离与半径；
u_bad：坏样本被预测为正例的概率均值；u_good：好样本被预测为正例的概率均值；
var_bad：坏样本被预测为正例的概率方差；var_good：好样本被预测为正例的概率方差；
返回float的区分度divergence值
'''
def distin_divergence(lab,proba):
    df_diver=pd.DataFrame({'lab':lab,'proba':proba})
    mean_dict=df_diver.groupby('lab').mean().to_dict()['proba']
    mean0,mean1=mean_dict[0],mean_dict[1]
    var_dict=df_diver.groupby('lab').var().to_dict()['proba']
    var0,var1=var_dict[0],var_dict[1]
    diver=(mean1-mean0)**2/(0.5*(var0+var1))
    return diver

'''
模型区分度gini=sum( ni/N * sum( Pk(1-Pk) ) )
将样本被预测为正例的概率分段，然后计算gini不纯度，越小越纯，区分度越好，但较小的gini值可能是分组细造成，
故使用不同分段数。区分度好的模型，gini值受分段数影响小。
n_splits：list，分段数；其他参数含义参考函数distin_ks；
返回float的gini值
'''
def distin_gini(lab,proba,n_splits=range(3,10),plot_gini=False):
    df_gini=pd.DataFrame({'lab':lab,'proba':proba})
    cnt_all=df_gini.shape[0]
    gini_all=[]
    for n in n_splits:
        thresh=[i*1/n for i in range(1,n)]
        df_gini['splits']=df_gini.proba.map(lambda x:refresh_val_2thresh(x,thresh))
        regroup=bin_badrate(df_gini,'splits','lab')
        regroup['gini']=regroup.apply(lambda x:x.total/cnt_all*2*x.badrate*(1-x.badrate),axis=1)
        gini=regroup.gini.sum()
        gini_all.append(gini)
    if plot_gini==True:
        fig1134,ax1134=plt.subplots()
        ax1134.plot(list(n_splits),gini_all,'ko',label='mean_gini:%s'%round(np.mean(gini_all),3))
        ax1134set=ax1134.set(xlabel='n_splits',ylabel='gini',title='Gini in different n_splits',ylim=[0,1])
        ax1134.legend(loc='best')
        for x,y in zip(list(n_splits),gini_all):
            ax1134text=ax1134.text(x,y,round(y,3),ha='center',va='bottom')
    return np.mean(gini_all)


#对新数据集进行woe编码。若只是考虑批量处理测试集数据，没必要编写该函数；之所以写，主要是针对
#新样本的单个编码处理，以及根据其为正例概率计算出的分数拆解到变量取值。
#df0：DataFrame格式数据，可以是一个或多个样本，传入单个样本时，不能是Series；
#feat_model：模型使用的woe编码后的特征，名称有'_woe'后缀；
#var_discrete_less：分箱前确定的离散无序变量；
#var_order：分箱前确立的离散有序变量；
#woe：训练集确立的woe编码规则；
#train_data_woe：训练集数据，只含woe编码变量及其对应的未编码变量；
#encode_discrete：离散无序取值种类多变量badrate编码规则；
#var_continue：编码前确立的连续型变量
#返回处理好的DataFrame
#在本函数中，除了想要处理的数据df0需要自行传入，其他参数取值在之前程序已经处理好。
def encode_new(df0,feat_model,var_discrete_less,var_order,woe,train_data_woe,encode_discrete,var_continue,y='y'):
    df=df0.copy()
    var_model=[var.replace('_woe','') for var in feat_model]
    for var in  var_model:
        #离散取值种类少变量、离散有序变量
        if var in var_discrete_less or var in var_order:
            df[var+'_woe']=df[var].map(woe[var+'_woe'])
            #df0为一个样本或多个样本时，都可判断
            if df[var+'_woe'].isnull().any():
                print('离散取值种类少变量：%s，新数据集存在训练集未见取值，以badrate最大值对应woe填充'%var)
                regroup_fillna=bin_badrate(train_data_woe,var+'_woe',y).sort_values(by='badrate',ascending=False)
                woe_max_badrate=list(regroup_fillna.index)[0]
                df[var+'_woe']=df[var+'_woe'].fillna(woe_max_badrate)
        #离散取值种类多变量
        if var[-3:]=='_br':
            var=var.replace('_br','')
            df[var+'_br']=df[var].map(encode_discrete[var])
            if df[var+'_br'].isnull().any():
                print('离散取值种类多变量：%s，新数据集存在训练集未见取值，以badrate最大值编码'%var)
                df[var+'_br']=df[var+'_br'].fillna(max(encode_discrete[var].values()))
            var=var+'_br'#让var_continue可识别
        #连续变量        
        if var in var_continue:
            thresh=sorted(list(woe[var+'_woe'].keys()))
            df[var][df[var] > max(thresh)]=max(thresh)
            df[var][df[var] < min(thresh)]=min(thresh)
            df[var+'_woe']=df[var].map(lambda x:refresh_val_2thresh(x,thresh))
            df[var+'_woe']=df[var+'_woe'].map(woe[var+'_woe'])
    return df


'''
一、模型稳定性指标：psi=sum( (观察组占比-参照组占比)/ln(观察组占比/参照组占比) )；
二、说明：组...占比是指 分组后，同一概率区间，组样本数 占 所有样本总数 的比例；
    若参照组来自训练集，观察组来自测试集，psi表示的是模型从训练集到测试集的稳定性；
    psi经验值：<0.1,稳定性高；0.1~0.2，一般；>0.2，差，建议修复模型；(但不同人说法不一)
三、参数：
lab_ctrl：参照组样本真实标签；
proba_ctrl：参照组样本为正例（标签被设置为1，当前程序正例是坏样本）的概率；
lab_actual：观察组样本真实标签；
proba_actual：观察组样本为正例的概率;
n_splits=10：将对照组样本划分为n_splits等份，每等份样本数大致相当。
鉴于某个概率区间样本数可能为0；某个概率值下样本数量涵盖两个区间；实际分组数小于等于n_splits，此时各组样本总数相差
会比较大，但这并不影响psi的效能评价，因为psi关注的是同一概率区间下，观察组与参照组样本数占各自样本总数的比例差异
psi计算过程：
1、对照组样本按 样本为正例的概率(proba_expected) 升排序，然后划分为n_splits等份；
2、取每等份proba_expected最大值为阈值，剔除重复阈值后余下n个阈值，n<=n_spits；
3、使用这n个阈值将对照组和观察组分割为n+1组，计算psi；
'''
def psi(lab_ctrl,proba_ctrl,lab_actual,proba_actual,n_splits=10):
    #根据参照组样本为正例的概率确定分组上界
    df_ctrl=pd.DataFrame({'lab_ctrl':lab_ctrl,'proba_ctrl':proba_ctrl})
    df_ctrl=df_ctrl.sort_values(by='proba_ctrl',ascending=True).reset_index(drop=True)
    per_cnt=math.ceil(df_ctrl.shape[0]/n_splits)
    id_thresh=[i*per_cnt-1 for i in range(1,n_splits)]
    thresh=sorted(list(set([df_ctrl.proba_ctrl.loc[i] for i in id_thresh])))#剔除重复值后保持升排序
    #参照组：计算各组样本数占比
    df_ctrl['bin']=df_ctrl.proba_ctrl.map(lambda x:refresh_val_2thresh(x,thresh))
    dict_ctrl=df_ctrl['bin'].value_counts(normalize=True).to_dict()
    dict_ctrl_sorted=sorted(dict_ctrl.items(),key=lambda x:x[0],reverse=False)
    pnt_ctrl=[i[1] for i in dict_ctrl_sorted]
    #观察组：计算各组样本数占比
    df_actual=pd.DataFrame({'lab_actual':lab_actual,'proba_actual':proba_actual})
    df_actual['bin']=df_actual.proba_actual.map(lambda x:refresh_val_2thresh(x,thresh))
    dict_actual=df_actual.bin.value_counts(normalize=True).to_dict()
    dict_actual_sorted=sorted(dict_actual.items(),key=lambda x:x[0],reverse=False)
    pnt_actual=[i[1] for i in dict_actual_sorted]
    #计算psi
    psi_detail=[(actual-ctrl)*np.log(actual/ctrl) for ctrl,actual in zip(pnt_ctrl,pnt_actual)]
    return sum(psi_detail)


#score = base_score - pdo/log(2) * log(p/1-p)
base_score=1000;pdo=200#相关修改，改这里即可
def proba_2score(proba,base_score=base_score,pdo=pdo):
    odds=proba/(1-proba)
    score=base_score-pdo/np.log(2)*np.log(odds)
    return score


#传入一个或多个DataFrame格式的样本，返回总分和得分细节{ 样本索引：{变量名（或'total'）：{变量取值：得分}} }，
#经验证，拆解再汇总得分与根据概率直接计算的得分一致。
#df：DataFrame格式的一个或多个样本，不能是其他格式；所含列名无指定；
#var_continue：list,未分组前分类确定的连续型变量；
#var_val_score：含所有得分明细的dict，形如{原始变量名：{原始变量取值：得分}}；
def score_split(df,var_continue,var_val_score):
    score_detail={}
    for i in range(len(df)):
        dict_sample=list(pd.DataFrame(df.iloc[i]).unstack().unstack().to_dict('index').items())[0]
        id_sample,var_val_sample=dict_sample[0],dict_sample[1]#样本索引；{变量：取值}
        score_detail[id_sample]={}
        for k,v in var_val_score.items():
            #考虑到之后的if...else结构，'intercept'最后在for结束后处理
            if k=='intercept':continue
            #连续变量取值先映射为其分组上界值，再据此获得得分
            if k in var_continue:
                thresh=refresh_val_2thresh(var_val_sample[k],sorted(list(var_val_score[k].keys())))
                score_detail[id_sample][k]={var_val_sample[k]:v[thresh]}
            #离散无序（取值种类无论多少），离散有序
            else:
                score_detail[id_sample][k]={var_val_sample[k]:v[var_val_sample[k]]}
        score_detail[id_sample]['intercept']=var_val_score['intercept']
        score_total=sum([list(i.values())[0] for i in list(score_detail[id_sample].values())])
        score_detail[id_sample]['total']={'score':score_total}
    return score_detail