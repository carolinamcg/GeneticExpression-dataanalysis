
# coding: utf-8


import numpy as np 
from scipy import stats
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

print('\n\n-----------------------  Carregamento e exloraçao dos dados  ----------------------------------\n')

data = pd.read_csv('gds4766.csv',sep=',', index_col = 0)
print ('Dimensões:' , data.shape)
data.head()
print(data.iloc[:5, :])

#metadadis

print(data.dtypes)
meta=pd.read_csv('meta-gds4766.csv', sep=',', index_col=0 )
meta.head()

print('Dimensões:', meta.shape)
print('\n')
print('Colunas:', meta.columns[:len(meta.columns)])
print('\n')
print('Tipos dos atributos:\n', meta.dtypes) 
print('\n')
print(meta.iloc[1:5,1:])

print('Contagem de valores únicos por atributo')
for i in range(1, len(meta.columns)-1):
    print(meta.groupby(meta.columns[i]).size(),"\n")


#1. **disease.state**:  estados da doença; 13 amostras de individuos com cranco da mama que não associado à gravidez e 20 amostras de individuos com cranco da mama que está associado à gravidez.
#2. **cell.type**: classificação das células quanto à sua localização e funcionalidade; 16 amostras em células epiteliais e 17 em células do estroma.
#3. **specimen**: tipo de células envolvidas; 13 amostras em células normais e 20 em células de tumor. 
#4. **genotype/variation**: classifica as células pela sua capacidade de ligação à hormona estrogénio; 15 amostras em células que não têm uma proteína através da qual 
#hormona estrogénio se liga e 18 que possuem essa mesma proteína. As células cancerígenas que são receptoras de estrogénio negativas não precisam de estrogênio para crescer, 
#e geralmente não param de crescer quando tratadas com hormonas que bloqueiam a ligação do estrogênio. Também chamado de ER-.As células cancerigenas ER+ crescem em resposta ao estrogénio.

print('Descrição de cada atributo')
print('\n')
print(meta.iloc[:,1:len(meta.columns)-1].describe())


print('Gráficos para verificar a distribuição dos valores nos atributos')
print('\n')
for atr in range(1, len(meta.columns)-1): 
    print(meta.columns[atr])
    labels = meta.iloc[:,atr].unique() #todas as linhas do meta, apenas a coluna do atributo em questão (atr). unique=2 tipos de valores para esta variável
    plt.pie(meta.iloc[:,atr].value_counts(), labels=labels, autopct='%.0f%%')
    plt.show()









print('\n\n-----------------------  Análise e pré-processamento  ----------------------------------\n')

# verificar a existência de missing values
print('Valores em falta:', np.sum(meta.isnull()).sum()) 
print('Contagem de string " na ":',meta.apply(lambda x : x.astype(str).str.contains(' na ').sum()).sum())

print('Valores em falta:', np.sum(data.isnull()).sum()) 
print('Contagem de string " na ":',data.apply(lambda x : x.astype(str).str.contains(' na ').sum()).sum())


#boxplot para todas as amostras
data.boxplot(figsize=(20, 10), return_type='axes')
plt.show()
#histogram para todas as amostras
data.hist(figsize=(18, 15)) 
plt.show()

# transposta do dataset, colocando os diferentes genes avaliados nas colunas e as amostras nas linhas do dataframe.
dataT= data.transpose()
dataT.describe()


# verificar se os dados de expressão génica seguem uma distribuição normal
for i in range(dataT.shape[1]):
    d=dataT.iloc[:,i]
    w, p_value = stats.shapiro(d) #Perform the Shapiro-Wilk test for normality.
    if p_value < 0.05:
        print("Coluna " + str(i) + " não segue uma distribuição normal")
    else:
        print("Coluna " + str(i) + " segue uma distribuição normal")


# cálculo da média e variância dos dados
variances = dataT.values.var(axis = 0)
media=dataT.values.mean(axis=0)
medvar = variances.mean() #variancia
med=media.mean()

print(media.shape)
print("Media: ", med)
print(variances.shape)
print("Variância: ", medvar)

#visualizar a variancia dos dados de expressão para todos os genes
X_indices = np.arange(variances.shape[0])
#gráfico de barras
plt.title("Variância dos dados de expressão:")
plt.bar(X_indices, variances, width=.4, label='Var') #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Genes') #título do eixo dos xx
plt.ylabel("Variância")
#plt.axis([0,54675,0,6])
plt.show()

n=variances.max()
print(n)
print(np.ndarray.tolist(variances).index(variances.max()))

# visualização limitada a 50 genes, incluindo o gene com maior variância
X_indices = np.arange(23700,23750)
#gráfico de barras
plt.title("Variância dos dados de expressão:")
plt.bar(X_indices, variances[23700:23750]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Genes') #título do eixo dos xx
plt.ylabel("Variância")
plt.show()


print('**Filtros por variabilidade (flat pattern):**')
# 
# * filtra genes cuja expressão varia pouco, ou seja, com a variância inferior a 2*med_var.
# * Tem-se, como objetivo, remover genes com informação irrelevante para a análise, isto é, genes com valores muito constantes.

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold= medvar*2)
input_filt = sel.fit_transform(dataT.values)
input_filt.shape


variance = input_filt.var(axis=0)
print(variance.shape)
x_indices = np.arange(variance.shape[0])
#  visualização limitada a 50 genes, incluindo o gene com maior variância
n=variance.max()
print(n)
print(np.ndarray.tolist(variance).index(variance.max()))
X_indices = np.arange(3100,3150)
plt.title("Variância dos dados de expressão:")
plt.bar(X_indices, variance[3100:3150]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Genes') #título do eixo dos xx
plt.ylabel("Variância")
plt.show()


prin("**Standardização dos dados (média: 0, variância: 1):**")

from sklearn import preprocessing

input_sc = preprocessing.scale(input_filt)
print(input_sc)
print("Media: ", input_sc.mean()) #média 0
print("Desvio padrao: ", input_sc.std()) #variância 1












print('\n\n-----------------------  Análise estatística multivariada  ----------------------------------\n')


print('** PCA (Principal component analysis):**')

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n=20
pca = PCA(n_components=20)
X_r=pca.fit(input_sc).transform(input_sc)

print('Var. explicada: %s'% str(pca.explained_variance_ratio_))
print(X_r.shape)
print('Os', X_r.shape[1],' componentes explicam', pca.explained_variance_ratio_.sum()*100, '% dos dados')

plt.bar(range(n), pca.explained_variance_ratio_*100, width=1)
plt.xticks(range(n), ['PC'+str(i) for i in range (1, n+1)])
plt.title("Variância explicada por PC")
#plt.ylable("Percentagem")
plt.show()


# Representação das amostras de acordo com os resultados do PCA. Daqui para a frente, sempre que as amostras forem ilustradas em gráfico será utilizada a representação determinada pelo PCA.</div>
plt.figure()
plt.clf()
for c, name in zip("rbg", ["normal", "tumor"]):
    plt.scatter(X_r[meta.values[:,3] == name, 0], X_r[meta.values[:,3] == name, 1], c=c, marker='+' , label=name)
plt.legend()
plt.axis([-80,100,-80,80])
plt.title('PCA - breast cancer')
plt.show()

for c, name in zip("cg", ["non-pregnancy-associated breast cancer", "pregnancy-associated breast cancer"]):
    plt.scatter(X_r[meta.values[:,1] == name, 0],X_r[meta.values[:,1] == name, 1], c=c, marker='+' , label=name)
plt.legend()
plt.axis([-80,100,-80,110])
plt.show()







print('** CLUSTERING:**')
 Clustering distintos (KMeans, AgglomerativeClustering e o Clustering Hierárquico), pelos quais se realizou o agrupamento das amostras em dois grupos de acordo com semelhança entre as amostras.</div>

# AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
n_clusters=2
hclust = AgglomerativeClustering(n_clusters, affinity="euclidean",linkage="ward")
#(numero de clusters a encontrar, métrica usada para calcular a ligação, metodo das distancias utilizado)
# O critério de criação dos cluster selecionado é 'ward' sendo que este procura minimizar a variância dos clusters criados
hclust.fit(X_r) #realiza o clustering

output_diseasestate =  meta.values[:,1]
output_celltype = meta.values[:,2]
output_specimen= meta.values[:,3]
output_genotype= meta.values[:,4]

res_hc = hclust.labels_  #atribui estes nomes às labels
cluster=pd.crosstab([output_diseasestate,output_celltype,output_specimen,output_genotype], res_hc, rownames=['Disease state','Cell type','Specimen','Genotype'],colnames=['Cluster']) #organiza os resultados numa tabela de contingência
cluster


# **Visualização dos clusters**
#http://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py
#Adaptou-se o tutorial do link acima para se visualizar o clustering efetuado
import matplotlib.pyplot as plt
from itertools import cycle

labels = hclust.labels_
plt.figure()
plt.clf()
colors = cycle('cryb')
plt.axis([-80,100,-80,80])
for k, col in zip(range(n_clusters), colors): 
    plt.plot(X_r[labels == k, 0], X_r[labels == k, 1], col + 'D')

plt.title('Resultado do Clustering: ')
plt.show()


print('-> Expressão Diferencial I:')
# **análise da variável diseasestate: cancro associado à gravidez vs cancro não associado à gravidez)**

# Em modo de identificar o p-value de cada gene, realiza-se um teste estatístico ANOVA, aplicado à variável diseasestate. </div>
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import preprocessing

selector1 = SelectPercentile(f_classif, percentile=10) #teste ANOVA-F, percentil 10
selector1.fit(input_sc, output_diseasestate)
selector1.pvalues_
X_indices = np.arange(input_sc.shape[-1]) #Retorna uma matriz de valores uniformemente espaçados dentro de um determinado intervalo.
#Os valores são gerados dentro do intervalo semiaberto [start, stop]. Neste caso, só há um argumento que é o stop
#input_sc-shape=(33,6902) ; input_sc.shape[-1]= 6902 (da coluna 0 á linha 6901)
#Para argumentos inteiros, a função é equivalente à função de intervalo interna do Python, mas retorna uma matriz nd em vez de uma lista.

scores = -np.log10(selector1.pvalues_) #Return the base 10 logarithm of the input array, element-wise.
#plt.axis([0,6902,0,5])
plt.bar(X_indices, scores, width=.4, label='Score')
plt.xlabel('Feature number')
plt.show()

n=scores.max()
print(n)
print(np.ndarray.tolist(scores).index(scores.max()))
X_indices = np.arange(6019,6069)
#gráfico de barras
plt.bar(X_indices, scores[6019:6069]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Feature number') 
plt.show()


# Filtragem os dados identificados, para a manter apenas os genes com p-value<0.05, ou seja, os dados que rejeitam a hipótese nula do 
#teste ANOVA (Ho= dados têm comportamentos iguais). Assim sendo, só os genes com comportamentos diferentes e que se mantêm, de modo a podermos 
#inferir melhor as diferenças entre os dois tipos de dados: cancro associado à gravidez e cancro não associado à gravidez.

ind_keep = np.where(selector1.pvalues_ <0.05)  #condição para pvalues<0.05; retorna os valores para os quais a condição se verifica
print(ind_keep[0]) #index das colunas para as quais o gene tem um p.value < 0.05 
input_sc_filt2 = input_sc[:,ind_keep[0]]
print(input_sc_filt2.shape)
#criação de comandos que permitam contabilizar os genes sobre e sub-expressados entre os de genes c/ maior expressão diferencial
lista_indices = ind_keep[0].tolist() #converter esses valores numa lista
lista_exp=input_sc_filt2.tolist()
downreg=0
upreg=0
for a in lista_exp[0]:
    if a<0:
        downreg += 1
    else:
        upreg +=1       
print('downregulated genes:',downreg)
print('upregulated genes:', upreg)


print('** Clustering 1.1:**')
# Para verificar se a nova distribuição das amostras permite tirar suposições em relação à conexão entre as variáveis, 
#após a filtragem dos dados de acordo com a sua expressão diferencial, efetuou-se um segundo processo de clustering
n_clusters=2
hclust2 = AgglomerativeClustering(n_clusters, affinity="euclidean",linkage="ward") #(numero de clusters a encontrar, metodo das distancias utilizado, ...)
hclust2.fit(input_sc_filt2) #realiza o clustering

output_description= meta.values[:,5]
res_hc = hclust2.labels_  #atribui estes nomes às labels
cluster2=pd.crosstab(output_diseasestate, res_hc, rownames=['Disease state'],colnames=['Cluster']) 
cluster2


# K-Means
from sklearn.cluster import KMeans
k=2
kmeans=KMeans(n_clusters=k, max_iter=1000) #algoritmo heuristico com 1000 iterações
kmeans.fit(input_sc_filt2)
labels=kmeans.labels_ #vetor que contem true/false consoante a amostra pertence ou nao ao cluster 0 1  
centroids=kmeans.cluster_centers_
print(pd.crosstab(labels, meta.iloc[:,1], rownames=['clusters']))

#tabela de contingência manualmente
colNames=meta['disease.state'].unique()
table=pd.DataFrame(0, index=range(k), columns=colNames) #criacao de um df/tabela com zeros de acordo
#com o numero de classes de tissue
#faz tabela onde tem quantas amostras de cada classe pertencem a cada cluster

for classe in colNames:
    table[classe]= [sum(meta[labels==i]['disease.state']==classe) for i in range(k)]
table

#representação gráfica
for i in range(k):
    #slect only data observations with cluster lable ==i
    ds=input_sc_filt2[np.where(labels==i)]
    #plot the data observations
    plt.plot(ds[:,0], ds[:,1], 'o', label = colNames[i])
    #plot the centroids
    lines=plt.plot(centroids[i,0], centroids[i,1], 'kx')
    #make the centroids x's bigger
    plt.setp(lines, ms=15)
    plt.setp(lines, mew=2)
plt.legend(loc='best', shadow=False)
plt.show()


#Clustering Hierárquico
from scipy.cluster.hierarchy import dendrogram, linkage

Z=linkage(input_sc_filt2, metric='correlation', method='average')#, metric='cityblock')

#calculate full dendrogram
plt.figure(figsize=(15,10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, 
          labels=list(meta.iloc[:,1]),
           leaf_rotation=90, #rotate the x axis labels
          leaf_font_size=8) #font size for the x axis labels

#Assignment of colors to lables
label_colors={'pregnancy-associated breast cancer':'r', 'non-pregnancy-associated breast cancer':'g'}
ax=plt.gca()
xlabels=ax.get_xmajorticklabels()
for lbl in xlabels:
    lbl.set_color(label_colors[lbl.get_text()])
plt.show()



# ### **Visualização da expressão dos genes filtrados**
plt.xlabel("Genes com maior expressão diferencial")
plt.ylabel("Amostras")
plt.title("Todas as amostras")
plt.figure(1)
plt.imshow(input_sc_filt2[0:34,],cmap='hot',interpolation='nearest',aspect =8)
plt.show()

plt.xlabel("Genes com maior expressão diferencial")
plt.ylabel("Amostras")
plt.title("Amostras de tecido do estroma de PABC")
plt.imshow(input_sc_filt2[0:13,],cmap='hot',interpolation='nearest',aspect =8)
plt.show()

plt.xlabel("Genes com maior expressão diferencial")
plt.ylabel("Amostras")
plt.title("Amostras de tecido epitelial de PABC")
plt.imshow(input_sc_filt2[13:22,],cmap='hot',interpolation='nearest',aspect =9)
plt.show()

plt.xlabel("Genes com maior expressão diferencial")
plt.ylabel("Amostras")
plt.title("Amostras de tecido do estroma de n-PABC")
plt.imshow(input_sc_filt2[22:28,],cmap='hot',interpolation='nearest',aspect =10)
plt.show()


plt.xlabel("Genes com maior expressão diferencial")
plt.ylabel("Amostras")
plt.title("Amostras de tecido epitelial de n-PABC")
plt.imshow(input_sc_filt2[28:33,],cmap='hot',interpolation='nearest',aspect =10)
plt.show()



print('-> Expressão Diferencial II:')
# >**análise da variável specimen: células normais vs células cancerígenas**

# Para identificar os p-values dos genes realizou-se um teste estatístico ANOVA aplicado à variável specimen.
selector2 = SelectPercentile(f_classif, percentile=10) #teste ANOVA-F, percentil 10
selector2.fit(input_sc, output_specimen)
selector2.pvalues_

X_indices = np.arange(input_sc.shape[-1])
scores = -np.log10(selector2.pvalues_)
plt.bar(X_indices, scores, width=.4, label='Score')
plt.xlabel('Feature number')
plt.show()

n=scores.max()
print(n)
print(np.ndarray.tolist(scores).index(scores.max()))
X_indices = np.arange(0,50)
plt.bar(X_indices, scores[0:50]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Feature number') 
plt.show()


# Filtra-se os dados identificados, de forma a manter apenas os genes com p-value<0.05. 
ind_keep = np.where(selector2.pvalues_ <0.05)  #condição para pvalues<0.05
print(ind_keep[0])
input_sc_filt3 = input_sc[:,ind_keep[0]]
print(input_sc_filt3.shape)
#criação de comandos que permitam contabilizar os genes sobre e sub-expressados entre os de genes c/ maior expressão diferencial
lista = input_sc_filt3.tolist()
listaup=[]
listadown=[]
listagenes1=[]
downreg=0
upreg=0
i=0
for a in lista[0]:
    if a<0:
        downreg += 1
        listadown.append(i)
    else:
        upreg +=1
        listaup.append(i)
    i+=1
print('downregulated genes:',downreg)
print('upregulated genes:', upreg)
print('Genes com maior expressão diferencial',input_sc_filt3[1].shape)
listagenes1=listadown+listaup



print('** Clustering 2.1: (variável specimen)**')
n_clusters=2
hclust2 = AgglomerativeClustering(n_clusters, affinity="euclidean",linkage="ward") #(numero de clusters a encontrar, metodo das distancias utilizado, ...)
hclust2.fit(input_sc_filt3) #realiza o clustering

output_description= meta.values[:,5]
res_hc = hclust2.labels_  #atribui estes nomes às labels
cluster2=pd.crosstab(output_specimen, res_hc, rownames=['Specimen'],colnames=['Cluster']) 
cluster2


# **K-Means**
k=2
kmeans=KMeans(n_clusters=k, max_iter=1000) #algoritmo heuristico com 1000 iterações
kmeans.fit(input_sc_filt3)
labels=kmeans.labels_ #vetor que contem true/false consoante a amostra pertence ou nao ao cluster 0 1  
centroids=kmeans.cluster_centers_
print(pd.crosstab(labels, meta.specimen, rownames=['clusters']))

#tabela de contingência manualmente
colNames=meta['specimen'].unique()
table=pd.DataFrame(0, index=range(k), columns=colNames) #criacao de um df/tabela com zeros de acordo
#com o numero de classes de tissue
#faz tabela onde tem quantas amostras de cada classe pertencem a cada cluster

for classe in colNames:
    table[classe]= [sum(meta[labels==i]['specimen']==classe) for i in range(k)]
table

#representação gráfica
for i in range(k):
    #slect only data observations with cluster lable ==i
    ds=input_sc_filt3[np.where(labels==i)]
    #plot the data observations
    plt.plot(ds[:,0], ds[:,1], 'o', label = colNames[i])
    #plot the centroids
    lines=plt.plot(centroids[i,0], centroids[i,1], 'kx')
    #make the centroids x's bigger
    plt.setp(lines, ms=15)
    plt.setp(lines, mew=2)
plt.legend(loc='best', shadow=False)
plt.show()


# **Clustering Hierárquico** 
Z=linkage(input_sc_filt3, metric='correlation', method='average')#, metric='cityblock')
plt.figure(figsize=(15,10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, 
          labels=list(meta.specimen),
           leaf_rotation=90, #rotate the x axis labels
          leaf_font_size=8) #font size for the x axis labels

label_colors={'tumor':'r', 'normal':'g'}
ax=plt.gca()
xlabels=ax.get_xmajorticklabels()
for lbl in xlabels:
    lbl.set_color(label_colors[lbl.get_text()])
plt.show()




print('-> Expressão Diferencial III:')
# >**análise da variável celltype: células epiteliais vs células do estroma** 

#Para identificar os p-values dos genes, realizou-se um teste estatístico ANOVA aplicado à variavel celltype.

selector3 = SelectPercentile(f_classif, percentile=10) #teste ANOVA-F, percentil 10
selector3.fit(input_sc, output_celltype)
selector3.pvalues_
X_indices = np.arange(input_sc.shape[-1])
scores = -np.log10(selector3.pvalues_)
plt.bar(X_indices, scores, width=.4, label='Score')
plt.xlabel('Feature number')
plt.show()

n=scores.max()
print(n)
print(np.ndarray.tolist(scores).index(scores.max()))
X_indices = np.arange(5450,5500)
#gráfico de barras
plt.bar(X_indices, scores[5450:5500]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Feature number') 
plt.show()


# Filtra-se os dados identificados de forma a manter apenas os genes com p-value<0.05.
ind_keep = np.where(selector3.pvalues_ <0.05)  #condição para pvalues<0.05
print(ind_keep[0])
input_sc_filt4= input_sc[:,ind_keep[0]]
print(input_sc_filt4.shape)
#criação de comandos que permitam contabilizar os genes sobre e sub-expressados entre os de genes c/ maior expressão diferencial
lista = input_sc_filt4.tolist()
listaup=[]
listadown=[]
listagenes2=[]
downreg=0
upreg=0
i=0
for a in lista[0]:
    if a<0:
        downreg += 1
        listadown.append(i)
    else:
        upreg +=1
        listaup.append(i)
    i+=1
print('Genes com maior expressão diferencial',input_sc_filt4[1].shape)
print('downregulated genes:',downreg)
print('upregulated genes:', upreg)
listagenes2=listadown+listaup

print('** Clustering 3.1: (variável celltype)**')
#AgglomerativeClustering
n_clusters=2
hclust2 = AgglomerativeClustering(n_clusters, affinity="euclidean",linkage="ward") #(numero de clusters a encontrar, metodo das distancias utilizado, ...)
hclust2.fit(input_sc_filt4) #realiza o clustering

output_description= meta.values[:,5]
res_hc = hclust2.labels_  #atribui estes nomes às labels
cluster2=pd.crosstab(output_celltype, res_hc, rownames=['Cell type'],colnames=['Cluster']) 
cluster2

# **K-Means**
k=2
kmeans=KMeans(n_clusters=k, max_iter=1000) #algoritmo heuristico com 1000 iterações
kmeans.fit(input_sc_filt4)
labels=kmeans.labels_ #vetor que contem true/false consoante a amostra pertence ou nao ao cluster 0 1  
centroids=kmeans.cluster_centers_
print(pd.crosstab(labels, meta.iloc[:,2], rownames=['clusters']))

#tabela de contingência manualmente
colNames=meta['cell.type'].unique()
table=pd.DataFrame(0, index=range(k), columns=colNames) #criacao de um df/tabela com zeros de acordo
#com o numero de classes de tissue
#faz tabela onde tem quantas amostras de cada classe pertencem a cada cluster

for classe in colNames:
    table[classe]= [sum(meta[labels==i]['cell.type']==classe) for i in range(k)]
table

#representação gráfica
for i in range(k):
    #slect only data observations with cluster lable ==i
    ds=input_sc_filt4[np.where(labels==i)]
    #plot the data observations
    plt.plot(ds[:,0], ds[:,1], 'o', label = colNames[i])
    #plot the centroids
    lines=plt.plot(centroids[i,0], centroids[i,1], 'kx')
    #make the centroids x's bigger
    plt.setp(lines, ms=15)
    plt.setp(lines, mew=2)
plt.legend(loc='best', shadow=False)
plt.show()


# **Clustering Hierárquico**
Z=linkage(input_sc_filt4, metric='correlation', method='average')#, metric='cityblock')
plt.figure(figsize=(15,10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, 
          labels=list(meta.iloc[:,2]),
           leaf_rotation=90, #rotate the x axis labels
          leaf_font_size=8) #font size for the x axis labels

label_colors={'epithelial':'r', 'stroma':'g'}
ax=plt.gca()
xlabels=ax.get_xmajorticklabels()
for lbl in xlabels:
    lbl.set_color(label_colors[lbl.get_text()])
plt.show()









print('\n\n-----------------------  Modelos ML  ----------------------------------\n')
#separação dos dados em conjuntos: treino e teste. Utiliza-se apenas os genes filtrados no ponto anterior, para o caso de comparação entre cancro associado e não associado à gravidez, 
#visto que é esta classificação que se pretender prever. Mantém-se um terço das amostras nos dados de teste e as restantes nos de treino, correspondendo, então, a variável de saída à variável diseasestate.
indices = np.random.permutation(len(input_sc_filt2)) #coloca, na matriz, os índices numa ordem aleatória
numtr = int(1/3 * input_sc_filt2.shape[0])    
#treino:
train_in = input_sc_filt2[indices[:-numtr]]
train_out = output_diseasestate[indices[:-numtr]]
#teste:
test_in  = input_sc_filt2[indices[-numtr:]]
test_out = output_diseasestate[indices[-numtr:]]


# **Modelo de Classificação:KNeighbors**
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_in, train_out)
preds = knn.predict(test_in)
print("Valores previstos:", preds)
print("\n")
#através do modelo criado, vê quais são os resultados obtidos para aquelas entradas
print("Valores reais: " , test_out)
print("\n")

print('Matriz de confusão:')
print(confusion_matrix(test_out,knn.predict(test_in)))
print('\n')

#compara com os valores que realmente saíram
print("Percentagem de acertos:", (preds == test_out).sum()/len(preds)*100,'%' )




# **Modelos de classificação: ÁRVORES de Decisão**
from sklearn import tree

tree_model = tree.DecisionTreeClassifier()
tree_model = tree_model.fit(train_in, train_out)
preds_tree = tree_model.predict(test_in)
print("Valores previstos: ", preds_tree)
print("\n")
print("Valores reais: " , test_out)
print("\n")
print('Matriz de confusão:')
print(confusion_matrix(test_out,tree_model.predict(test_in)))
print('\n')

print("Percentagem de acertos: ", (preds_tree == test_out).sum()/len(preds_tree)*100, '%' )


from sklearn.datasets import load_iris
from sklearn import tree
from graphviz import Digraph
from graphviz import Source
from PIL import Image

clf = tree.DecisionTreeClassifier()
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
n=tree.export_graphviz(clf, out_file='tree.dot')

from IPython.display import Image
Image(filename='tree.PNG')




# **Modelos de classificação: REGRESSÃO LOGÍSTICA**
from sklearn import linear_model

logistic=linear_model.LogisticRegression(C=1e5)
logistic=logistic.fit(train_in,train_out)
print(logistic)
prevs=logistic.predict(test_in)
print("\n")
print("Valores previstos: ", prevs)
print("\n")
print("Valores reais: ",test_out)
print("\n")

print('Matriz de confusão:')
print(confusion_matrix(test_out,logistic.predict(test_in)))
print('\n')

print("Percentagem de acertos: ", (prevs == test_out).sum()/len(prevs)*100, '%') 





# **SGDClassifier** (Stochastic Gradient Descent )
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

sgd_clf = SGDClassifier(loss="hinge", penalty="l2", tol=1e-3)
sgd_clf= sgd_clf.fit(train_in, train_out)
prevs=sgd_clf.predict(test_in)
print("\n")
print("Valores previstos: ", prevs)
print("\n")
print("Valores reais: ",test_out)
print("\n")

print('Matriz de confusão:')
print(confusion_matrix(test_out,sgd_clf.predict(test_in)))
print('\n')

scores = cross_val_score(sgd_clf, input_sc_filt2, output_diseasestate, cv=5)
print('Scores com cross validation com função scoring Pecc e com 5 folds:')
print(scores.mean()*100, '%')


# **Neural Networks**

# Útilizou-se o *solver* 'Lbfgs' que é um otimizador da família *quasi_Newton* e o *alpha* 1e-5 que foram os parâmetros que apresentaram melhores resultados (segundo um tentativas previas).
from sklearn.neural_network import MLPClassifier
neural_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# hidden_layer_sizes=number of neurons in the ith hidden layer.
#If int, random_state is the seed used by the random number generator
neural_clf= neural_clf.fit(train_in, train_out)
prevs=neural_clf.predict(test_in)
print("\n")
print("Valores previstos: ", prevs)
print("\n")
print("Valores reais: ",test_out)
print("\n")

print('Matriz de confusão:')
print(confusion_matrix(test_out, neural_clf.predict(test_in)))
print('\n')

scores = cross_val_score(neural_clf, input_sc_filt2, output_diseasestate, cv=5)
print('Scores com cross validation com função scoring Pecc e com 5 folds:')
print(scores.mean()*100, '%')






# **Modelo de Classificação: Validação cruzada**
from sklearn import svm

model_svm = svm.SVC(gamma=0.001, C=100)
model_svm= model_svm.fit(train_in, train_out)
prevs=model_svm.predict(test_in)
print("\n")
print("Valores previstos: ", prevs)
print("\n")
print("Valores reais: ",test_out)
print("\n")

print('Matriz de confusão:')
print(confusion_matrix(test_out, model_svm.predict(test_in)))
print('\n')

scores = cross_val_score(model_svm, input_sc_filt2, output_diseasestate, cv = 5)
print(scores)
print("Percentagem de acertos:", scores.mean()*100, '%') #Scores com cross validation com função scoring Pecc e com 5 folds







print('\n\n-----------------------  Otimização dos Modelos ML  ----------------------------------\n')



# ####  KNeighbors

#otimização apenas um parâmetro (*n_neighbors*).
# try K=1 through K=23 and record testing accuracy (n_samples = 22)
k_range = range(1, 23)
# We can create Python dictionary using [] or dict()
scores = []
# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_in, train_out)
    preds = knn.predict(test_in)
    score= (preds == test_out).sum()/len(preds)*100)
    scores.append(score)
print(scores)

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# + #### SGDClassifier

# procura em grelha dos parâmetros (gridSearch)

#  exporação de valores aceitáveis para *alpha*. Manteve-se 'l2' para o parâmetro *penalty*, que é o termo de regularização padrão para modelos lineares de SVM, 
#e -1, para *n_jobs*, de modo a usar todos os processadores para fazer a computação *OVA* (*One Versus All, for multi-class problems*). 
# Quanto ao parâmetro *loss*, consideraram-se duas funções viáveis: 'hinge' e 'log'.
# 
#O hiper-parâmetro *alpha* serve para um propósito duplo. É, tanto um parâmetro de regularização, 
#como corresponde à taxa de aprendizagem inicial no cronograma padrão. Isso significa que, além de regularizar os coeficientes de regressão logística, 
#a saída do modelo depende de uma interação entre *alpha* e o número de épocas (*n_iter*) que a rotina de ajuste executa. 
#Especificamente, como *alpha* se torna muito pequeno, *n_iter* deve ser aumentado para compensar a baixa taxa de aprendizado. 
#É por isso que é mais seguro (mas mais lento) especificar *n_iter* suficientemente grande, por ex. 1000, quando pesquisando sobre uma ampla gama de *alphas*.</div>

# In[161]:
from time import time
from operator import itemgetter
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
 

X, y = train_in, train_out
# build a classifier
clf = SGDClassifier()

# Utility function to report best scores
def report(grid_scores,n_top=5):
    top_scores =sorted(grid_scores,key=itemgetter(1),reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank:{0}".format(i +1))
        print("Mean validation score: {0:.3f}(std:{1:.3f})".format(score.mean_validation_score,np.std(score.cv_validation_scores)))
        print("Parameters:{0}".format(score.parameters))
        print("")
 
# use a full grid over all parameters
param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
    'n_iter': [1000], # number of epochs
    'loss': ['log', 'hinge'], # logistic regression,
    'penalty': ['l2'],
    'n_jobs': [-1]} #The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation
   
# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(input_sc_filt2, output_diseasestate)
 
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


# In[182]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# evaluation of the model with the selected hyperparameters
sgd_clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=None,
       n_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
       power_t=0.5, random_state=None, shuffle=True, tol=None,
       validation_fraction=0.1, verbose=0, warm_start=False)
sgd_clf= sgd_clf.fit(train_in, train_out)
prevs=sgd_clf.predict(test_in)

scores = cross_val_score(sgd_clf, input_sc_filt2, output_diseasestate, cv=5)
print('Scores com cross validation com função scoring Pecc e com 5 folds:')
print(scores.mean()*100, '%')



# #### Neural Networks 
# 
# *learning_rate*: 'invscaling' e 'adaptative' -> permitem que a taxa de aprendizagem decresça à medida que o treino avança; 'constant' -> induz uma taxa de aprendizagem constante. 
# número de iteraçõe: apenas valores superiores a 1500 eram capazes de produzir resultados. 
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()

parameter_space = {
    'hidden_layer_sizes': [ (5,), (10,), (50,), (100,)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['lbfgs','sgd', 'adam'],
    'max_iter':[1500, 2000, 3000, 4000],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(input_sc_filt2, output_diseasestate)
print(clf.best_estimator_)
print("\n")
# Best paramete set
print('Best parameters found:\n', clf.best_params_)
print("\n")
print(clf.best_score_)



neural_clf = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(5,), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='sgd', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
neural_clf= neural_clf.fit(train_in, train_out)
prevs=neural_clf.predict(test_in)

scores = cross_val_score(neural_clf, input_sc_filt2, output_diseasestate, cv=5)
print('Scores com cross validation com função scoring Pecc e com 5 folds:')
print(scores.mean()*100, '%')




# + ####  Validação Cruzada (SVM)
from sklearn import svm

model=svm.SVC()

Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = ['auto', 0.0001, 0.001, 0.01, 0.1, 1]
param_grid = {'kernel':('linear','rbf'), 'C': Cs, 'gamma' : gammas}

grid_search =GridSearchCV(model, param_grid)
grid_search.fit(input_sc_filt2, output_diseasestate)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)


model_svm = svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
model_svm= model_svm.fit(train_in, train_out)
prevs=model_svm.predict(test_in)

scores = cross_val_score(model_svm, input_sc_filt2, output_diseasestate, cv = 5)
print("Percentagem de acertos:", scores.mean()*100, '%') 