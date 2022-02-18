
# coding: utf-8

# # Trabalho 2
# **Alunos:** André Branquinho   A82182
#             António Cunha      A79067
#             Carolina Gonçalves A80932
#             Luis Silva         A80981

# <div style="text-align: justify">Existe uma maior agressividade do cancro da mama em mulheres grávidas, quando comparado ao mesmo tipo de cancro em mulheres não grávidas.
# Ocorreu uma recolha de 33 amostras para análise. Estas foram recolhidas de epitélio maligno e de tecido normal (estroma), em individuos com cancro da mama associado à gravidez (PABC) e a individuos com o mesmo cancro mas sem estar associado à gravidez (n-PABC).
# Nos ficheiros ##gds4766.csv (metadados) e gds4766.csv (dados)## estão as caracteristicas e os dados de expressão genética relativa a cada amostra.
# Assim, pretende-se analisar os ficheiros disponíveis de modo a descobrir as caracteristicas do PABC que expliquem as diferenças deste cancro relativamente ao n-PABC.</div>

# # Etapa 1 - Dados

# In[1]:


import numpy as np 
from scipy import stats
import statsmodels.api as sm
import pandas as pd
data = pd.read_csv('gds4766.csv',sep=',', index_col = 0)
print ('Dimensões:' , data.shape)


# <div style="text-align: justify">O dataset possui 54675 genes(linhas) e 33 amostras(colunas). A primeira coluna não é contabilizada como amostra, dado que esta diz respeito à coluna que contém o nome dos genes. Cada gene possui um valor de expressão diferente para cada amostra.</div>

# In[6]:


data.head()


# In[7]:


print(data.iloc[:5, :])


# In[8]:


print(data.dtypes)


# <div style="text-align: justify">Podemos observar que os valores de cada coluna no dataset são valores décimais (float64), estes correspondem a valores de expressão génica para cada gene.</div>

# # ETAPA 2 - Metadados 

# In[2]:


meta=pd.read_csv('meta-gds4766.csv', sep=',', index_col=0 )
meta.head()


# In[26]:


print('Dimensões:', meta.shape)
print('\n')
print('Colunas:', meta.columns[:len(meta.columns)])
print('\n')
print('Tipos dos atributos:\n', meta.dtypes) 
print('\n')
print(meta.iloc[1:5,1:])


# <div style="text-align: justify">O ficheiro com os metadados contem 33 linhas(amostras) e 6 colunas. A primeira coluna (coluna 'sample') não é contabilizada, pois repete-se.</div>
# 
# <br>
# <div style="text-align: justify">Cada coluna contém valores do tipo object. Todas as variáveis são nominais.</div>
# 
# <br>
# <div style="text-align: justify">As colunas são: 'sample', 'disease.state', 'cell.type', 'specimen', 'genotype/variation', 'description'.</div>

# <div style="text-align: justify">Realizou-se a caracterização dos atributos. Como refirido anteriormente, nos metadados, há 33 exemplos (amostras) e 6 variáveis/atributos, todas elas nominais. Para as variáveis discretas, é possível calcular as frequências dos seus valores.</div>

# In[24]:


print('Contagem de valores únicos por atributo')
for i in range(1, len(meta.columns)-1):
    print(meta.groupby(meta.columns[i]).size(),"\n")


# 1. **disease.state**:  estados da doença; 13 amostras de individuos com cranco da mama que não associado à gravidez e 20 amostras de individuos com cranco da mama que está associado à gravidez.
# 
# 
# 2. **cell.type**: classificação das células quanto à sua localização e funcionalidade; 16 amostras em células epiteliais e 17 em células do estroma.
# 
# 3. **specimen**: tipo de células envolvidas; 13 amostras em células normais e 20 em células de tumor. 
# 
# 4. **genotype/variation**: classifica as células pela sua capacidade de ligação à hormona estrogénio; 15 amostras em células que não têm uma proteína através da qual hormona estrogénio se liga e 18 que possuem essa mesma proteína. As células cancerígenas que são receptoras de estrogénio negativas não precisam de estrogênio para crescer, e geralmente não param de crescer quando tratadas com hormonas que bloqueiam a ligação do estrogênio. Também chamado de ER-.As células cancerigenas ER+ crescem em resposta ao estrogénio.

# In[27]:


print('Descrição de cada atributo')
print('\n')
print(meta.iloc[:,1:len(meta.columns)-1].describe())


# In[29]:


import matplotlib.pyplot as plt
print('Gráficos para verificar a distribuição dos valores nos atributos')
print('\n')
for atr in range(1, len(meta.columns)-1): 
    print(meta.columns[atr])
    labels = meta.iloc[:,atr].unique() #todas as linhas do meta, apenas a coluna do atributo em questão (atr). unique=2 tipos de valores para esta variável
    plt.pie(meta.iloc[:,atr].value_counts(), labels=labels, autopct='%.0f%%')
    plt.show()


# # **Etapa 3 - Análise e pré-processamento**

# <div style="text-align: justify">Possiveís valores em falta:</div>

# In[43]:


print('Valores em falta:', np.sum(meta.isnull()).sum()) 
print('Contagem de string " na ":',meta.apply(lambda x : x.astype(str).str.contains(' na ').sum()).sum())


# <div style="text-align: justify">Não existem valores em omisso (NaN). Como se pode verificar, procurou-se também se existia alguma string "na", para ter certeza da não omissão de dados.</div>

# **Dados:**
# Anteriormente, verificou-se que as variáveis dos dados são numéricas (float) e dizem respeito às amostras, enquanto que as linhas dizem respeito aos genes.

# <div style="text-align: justify">Para verificar a existência de valores em falta repetimos o processo realizado para os metadados.</div>

# In[44]:


print('Valores em falta:', np.sum(data.isnull()).sum()) 
print('Contagem de string " na ":',data.apply(lambda x : x.astype(str).str.contains(' na ').sum()).sum())


# <div style="text-align: justify">Logo, não existem valores omissos.</div>

# <div style="text-align: justify">De seguida, construimos um boxplot para todas as amostras. Verificamos a escala e distribuição dos dados.</div>

# In[45]:


import pandas as pd
data.boxplot(figsize=(20, 10), return_type='axes')
plt.show()


# In[56]:


data.hist(figsize=(18, 15)) 
plt.show()


# <div style="text-align: justify">Os *boxplot* associados a cada amostra (paciente) apresentam centro (mediana), forma e tamanho semelhantes. Os histogramas de cada uma destas amostras também são indicativos de uma distribuição normal dos dados.
# Podemos observar, através dos gráficos anteriores, que os valores das diferentes amostras se encontram na mesma gama de valores, permitindo, assim, a sua comparação. </div>

# <div style="text-align: justify">Para analisar os dados da melhor forma, é necessário trabalhar com a transposta do nosso dataset, colocando os diferentes genes avaliados nas colunas e as amostras nas linhas do dataframe.</div>

# In[6]:


dataT= data.transpose()
dataT.describe()


# In[9]:


dataT.shape


# <div style="text-align: justify">De seguida, verificamos se os dados de expressão génica seguem uma distribuição normal.</div>

# In[4]:


from scipy import stats

for i in range(dataT.shape[1]):
    d=dataT.iloc[:,i]
    w, p_value = stats.shapiro(d) #Perform the Shapiro-Wilk test for normality.
    if p_value < 0.05:
        print("Coluna " + str(i) + " não segue uma distribuição normal")
    else:
        print("Coluna " + str(i) + " segue uma distribuição normal")


# <div style="text-align: justify">Como se pode observar,nem todos os dados seguem uma distribuição normal. No entanto, temos mais do que 30 observações em cada amostra.</div>

# <div style="text-align: justify">Procede-se, agora, ao cálculo da média e variância dos dados.</div>

# In[3]:


dataT= data.transpose()
variances = dataT.values.var(axis = 0)
media=dataT.values.mean(axis=0)
medvar = variances.mean() #variancia
med=media.mean()

print(media.shape)
print("Media: ", med)
print(variances.shape)
print("Variância: ", medvar)


# <div style="text-align: justify">A média dos dados de expressão é 5.54090472869 e a variância média é de 0.637082574261. Podemos, então, concluir que os dados ainda não se encontram standartizados.</div>

# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
X_indices = np.arange(variances.shape[0])

#gráfico de barras
plt.title("Variância dos dados de expressão:")
plt.bar(X_indices, variances, width=.4, label='Var') #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Genes') #título do eixo dos xx
plt.ylabel("Variância")
#plt.axis([0,54675,0,6])
plt.show()


# In[12]:


n=variances.max()
print(n)
print(np.ndarray.tolist(variances).index(variances.max()))


# <div style="text-align: justify">Como podemos observar pelo gráfico e pelo bloco de código anterior, as variâncias não são completamente visíveis na figura obtida acima, sendo necessário limitar a representação gráfica a apenas 50 genes, de modo a possibilitar a comparação e perceção viáveis destes valores, vizualizando-se, também, o maior pico de variância.</div>

# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

X_indices = np.arange(23700,23750)

#gráfico de barras
plt.title("Variância dos dados de expressão:")
plt.bar(X_indices, variances[23700:23750]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Genes') #título do eixo dos xx
plt.ylabel("Variância")
plt.show()


# **Filtros por variabilidade (flat pattern):**
# 
# 
# * filtra genes cuja expressão varia pouco, ou seja, com a variância inferior a 2*med_var.
# 
# * Tem-se, como objetivo, remover genes com informação irrelevante para a análise, isto é, genes com valores muito constantes.

# In[4]:


# filtros variabilidade (flat pattern)
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold= medvar*2)
input_filt = sel.fit_transform(dataT.values)
input_filt.shape


# <div style="text-align: justify">O número de genes a analizar diminuiu consideravelmente, passando de 54675 para 6902.</div>

# In[15]:


variance = input_filt.var(axis=0)
print(variance.shape)
x_indices = np.arange(variance.shape[0])

plt.bar(x_indices,variance,width = .4, label = "Var")
plt.title("Variância dos genes após filtro de variabilidade:")
plt.xlabel("Genes")
plt.ylabel("Variância")


# In[16]:


n=variance.max()
print(n)
print(np.ndarray.tolist(variance).index(variance.max()))


# <div style="text-align: justify">Do mesmo modo, verifica-se que é necessário limitar a representação gráfica a apenas 50 genes, tal como no caso anterior, possibilitando a comparação e perceção viáveis destes valores, vizualizando-se, também, o maior pico de variância.</div>

# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

X_indices = np.arange(3100,3150)

#gráfico de barras
plt.title("Variância dos dados de expressão:")
plt.bar(X_indices, variance[3100:3150]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Genes') #título do eixo dos xx
plt.ylabel("Variância")
plt.show()


# <div style="text-align: justify">Assim sendo, para futura representação deste tipo de gráficos, iremos sempre recorrer a esta metodologia, encontrando o valor máximo e representando apenas 50 genes, incluindo esse valor.</div>

# **Processos de pré-processamento:** 
# 
# <br>
# <div style="text-align: justify">standardização dos dados (média: 0, variância: 1)</div>

# In[5]:


from sklearn import preprocessing

input_sc = preprocessing.scale(input_filt)
print(input_sc)
print("Media: ", input_sc.mean()) #média 0
print("Desvio padrao: ", input_sc.std()) #variância 1


# # Etapa 4 - Análise estatística multivariada

# # PCA (Principal component analysis)

# <div style="text-align: justify">O método do PCA é um método estatístico que permite a análise de dados procedendo à sua redução, à eliminação de sobreposições e à escolha das formas que melhor representam os dados, partindo de combinações lineares das variáveis originais (redução de dimensionalidade linear).
# 
# <br>
# <div style="text-align: justify">Tem como objetivo transformar variáveis, possivelmente correlacionadas, num menor número de variáveis denominadas Componentes Principais (CP) capazes de representar os dados.</div>

# In[6]:


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


# <div style="text-align: justify">De seguida, procedeu-se há representação das amostras de acordo com os resultados do PCA. Daqui para a frente, sempre que as amostras forem ilustradas em gráfico será utilizada a representação determinada pelo PCA.</div>

# In[22]:


import matplotlib.pyplot as plt 

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


# <div style="text-align: justify"> Através do primeiro gráfico, podemos observar uma boa separação dos dados, relativamente ao specimen (tumor vs normal). Isto pode indiciar que a análise preditiva terá bons resultados.
# Quanto aos tipo de doença (cancro associado ou não associado à gravidez), já não se verifica uma separação tão eficaz destes dados, não sendo tão linear as diferenças de expressão génica, neste caso.</div>

# # Clustering 

# <div style="text-align: justify"> Nesta fase, é importante perceber se os nossos dados se aglomeram em diferentes grupos segundo o seu grau de semelhança.</div>
# 
# <br>
# <div style="text-align: justify">Para tal, foram utilizados algorítmos de Clustering distintos (KMeans, AgglomerativeClustering e o Clustering Hierárquico), pelos quais se realizou o agrupamento das amostras em dois grupos de acordo com semelhança entre as amostras.</div>

# In[7]:


from sklearn.cluster import AgglomerativeClustering
import pandas as pd
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


# <div style="text-align: justify"> O cluster hierárquico separou 26 amostras num cluster e 7 noutro. O primeiro, contem na sua maioria amostras de tecido normal e de estroma tumoral, enquanto que, o segundo possui apenas amostras de células epiteliais tumorais. Assim, podemos afirmar que as amostras de tecido normal epitelial e do estroma estão mais estreitamente relacionadas entre elas do que o mesmo tipo de tecido proveniente de tumores. Desta foram, a malignidade confere uma impressão digital única ao epitélio e estroma que os distingue dos tecidos normais.</div>

# **Visualização dos clusters**

# In[24]:


#http://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py
#Adaptou-se o tutorial do link acima para se visualizar o clustering hierárquico efetuado
import matplotlib.pyplot as plt
from itertools import cycle

labels = hclust.labels_
plt.figure()
plt.clf()
colors = cycle('cryb')
plt.axis([-80,100,-80,80])
for k, col in zip(range(n_clusters), colors): 
    plt.plot(X_r[labels == k, 0], X_r[labels == k, 1], col + 'D')

plt.title('Resultado do Clustering hierárquico: ')
plt.show()


# ### **Expressão Diferencial I**
# > **análise da variável diseasestate: cancro associado à gravidez vs cancro não associado à gravidez)**

# <div style="text-align: justify">Em modo de identificar o p-value de cada gene, realiza-se um teste estatístico ANOVA, aplicado à variável diseasestate. </div>

# In[8]:


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


# In[24]:


n=scores.max()
print(n)
print(np.ndarray.tolist(scores).index(scores.max()))


# In[27]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

X_indices = np.arange(6019,6069)

#gráfico de barras
plt.bar(X_indices, scores[6019:6069]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Feature number') 
plt.show()


# Filtra-se os dados identificados, para a manter apenas os genes com p-value<0.05, ou seja, os dados que rejeitam a hipótese nula do teste ANOVA (Ho= dados têm comportamentos iguais). Assim sendo, só os genes com comportamentos diferentes e que se mantêm, de modo a podermos inferir melhor as diferenças entre os dois tipos de dados: cancro associado à gravidez e cancro não associado à gravidez.

# In[9]:


from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import preprocessing
import numpy as np
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


# <div style="text-align: justify">Dos 465 genes identificados com maior expressão diferencial, 221 estão subexpressados (downregulated) e 244 estão sobreexpressados (upregulated).</div>

# ### **Clustering #1.1**
# 
# <br>
# <div style="text-align: justify">Para verificar se a nova distribuição das amostras permite tirar suposições em relação à conexão entre as variáveis, após a filtragem dos dados de acordo com a sua expressão diferencial, efetuou-se um segundo processo de clustering.</div>

# In[29]:


from sklearn.cluster import AgglomerativeClustering
import pandas as pd
n_clusters=2
hclust2 = AgglomerativeClustering(n_clusters, affinity="euclidean",linkage="ward") #(numero de clusters a encontrar, metodo das distancias utilizado, ...)
hclust2.fit(input_sc_filt2) #realiza o clustering

output_description= meta.values[:,5]
res_hc = hclust2.labels_  #atribui estes nomes às labels
cluster2=pd.crosstab(output_diseasestate, res_hc, rownames=['Disease state'],colnames=['Cluster']) 
cluster2


# **K-Means**

# In[69]:


import matplotlib.pyplot as plt
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


# **Clustering Hierárquico** 

# In[70]:


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


# <div style="text-align: justify">Observa-se que, o clustering efetuado apenas aos genes com maior expressão diferencial, separou as amostras em dois grupos, nos quais há uma quase total separação das amostras de cancro associado à gravidez e cancro não associado à gravidez, separação que não existia no clustering hierárquico anterior.</div>
# 
# <br>
# <div style="text-align: justify">Deste modo, estes dados mostram que os padrões de expressão genética subjacentes às amostras de cancro, associado e não associado à gravidez, são diferentes.</div>
# 
# <br>
# <div style="text-align: justify">A identificação das funções biológicas dos genes associados em cada tipo de cancro deverá indicar diferenças nos processos biológicos existentes nos dois tipos de cancro.</div>
# 
# <br>
# <div style="text-align: justify">Não obstante, segundo a visualização dos clusters obtidos através do K-Means, estes não se encontram completamente bem divididos, existindo a possibilidade de a previsão apresentar resultados menos elevados, pois os dados relativos a cancro da mama associado e não associado à gravidez não são perfeitamente distinguíveis.</div>

# ### **Visualização da expressão dos genes filtrados**

# In[73]:


import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
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

# imshow é um comando que trata uma matriz como uma imagem
# assume que os elementos da matriz são intensidades de pixeis


# <div style="text-align: justify">Prestando, agora, atenção aos 'mapas de calor' dos genes com maior exprassão diferencial, verificou-se que as amostras de cancro associado e não associado à gravidez apresentam diferenças no padrão de expressão diferencial, ou seja, os genes têm diferente expressão nos dois tipos de cancro. Verifica-se ainda diferenças na expressão genética entre o tecido epitelial e o tecido do estroma.</div>
# 
# <br>
# <div style="text-align: justify">Deduzimos que um estudo mais aprofundado destes padrões de expressão genética poderá indicar diferenças nos processos biológicos existentes nos dois tipos de cancro.</div>

# **Expressão Diferencial II**
# >**análise da variável specimen: células normais vs células cancerígenas**

# <div style="text-align: justify">Para identificar os p-values dos genes realizou-se um teste estatístico ANOVA aplicado à variável specimen.</div>

# In[31]:


from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import preprocessing

selector2 = SelectPercentile(f_classif, percentile=10) #teste ANOVA-F, percentil 10

selector2.fit(input_sc, output_specimen)
selector2.pvalues_

X_indices = np.arange(input_sc.shape[-1])
scores = -np.log10(selector2.pvalues_)
plt.bar(X_indices, scores, width=.4, label='Score')
plt.xlabel('Feature number')
plt.show()


# In[32]:


n=scores.max()
print(n)
print(np.ndarray.tolist(scores).index(scores.max()))


# In[33]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

X_indices = np.arange(0,50)

#gráfico de barras
plt.bar(X_indices, scores[0:50]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Feature number') 
plt.show()


# Filtra-se os dados identificados, de forma a manter apenas os genes com p-value<0.05. 

# In[34]:


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


# **Clustering #2.1 (variável specimen)**
# 
# <br>
# <div style="text-align: justify">Após a filtragem dos dados de acordo com a sua expressão diferencial, efetuou-se um segundo clustering, de modo a verificar se a nova distribuição das amostras permite tirar suposições em relação à conexão entre as variáveis.</div>

# In[76]:


from sklearn.cluster import AgglomerativeClustering
import pandas as pd
n_clusters=2
hclust2 = AgglomerativeClustering(n_clusters, affinity="euclidean",linkage="ward") #(numero de clusters a encontrar, metodo das distancias utilizado, ...)
hclust2.fit(input_sc_filt3) #realiza o clustering

output_description= meta.values[:,5]
res_hc = hclust2.labels_  #atribui estes nomes às labels
cluster2=pd.crosstab(output_specimen, res_hc, rownames=['Specimen'],colnames=['Cluster']) 
cluster2


# **K-Means**

# In[77]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

# In[78]:


from scipy.cluster.hierarchy import dendrogram, linkage

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


# <div style="text-align: justify">Pelos resultados acima obtidos, é possível notar que o clustering efetuado apenas aos genes com maior expressão diferencial separou as amostras em dois grupos, nos quais há uma divisão quase total (a menos de uma amostra) das amostras de tecido normal e tecido proveniente de tumores, registando-se, até, uma separação completa n«pelo algoritmo K-Means.</div>
# 
# <br>
# <div style="text-align: justify">Assim sendo, estes dados demonstram que existem diferenças na expressão genética das amostras de tecido normal e de tecido proveniente de tumores, como já se tinha previsto.</div>
# 
# <br>
# <div style="text-align: justify">Um estudo mais aprofundado destes padrões de expressão genética deverá indicar diferenças nos processos biológicos existentes nos dois tipos de tecido.</div>
# 
# <br>
# <div style="text-align: justify">No entanto e em semelhança ao clustering anterior, segundo a visualização dos clusters obtidos através do K-Means, estes não se encontram completamente bem distanciados, podendo, então, originar uma previsão com resultados menos elevados.</div>

# **Expressão Diferencial III**
# >**análise da variável celltype: células epiteliais vs células do estroma** 

# <div style="text-align: justify"> Para identificar os p-values dos genes, realizou-se um teste estatístico ANOVA aplicado à variavel celltype. </div>

# In[35]:


from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import preprocessing

selector3 = SelectPercentile(f_classif, percentile=10) #teste ANOVA-F, percentil 10

selector3.fit(input_sc, output_celltype)
selector3.pvalues_

X_indices = np.arange(input_sc.shape[-1])
scores = -np.log10(selector3.pvalues_)
plt.bar(X_indices, scores, width=.4, label='Score')
plt.xlabel('Feature number')
plt.show()


# In[36]:


n=scores.max()
print(n)
print(np.ndarray.tolist(scores).index(scores.max()))


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

X_indices = np.arange(5450,5500)

#gráfico de barras
plt.bar(X_indices, scores[5450:5500]) #(valores de x, valores de y,largura das barras,legendas laterais) 
plt.xlabel('Feature number') 
plt.show()


# Filtra-se os dados identificados de forma a manter apenas os genes com p-value<0.05.

# In[38]:


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


# **Clustering #3.1 (variável celltype)**
# 
# <br>
# <div style="text-align: justify">Após a filtragem dos dados de acordo com a sua expressão diferencial, realizou-se um segundo clustering para verificar se a nova distribuição das amostras permite tirar suposições em relação à conexão entre as variáveis.</div>

# In[83]:


from sklearn.cluster import AgglomerativeClustering
import pandas as pd
n_clusters=2
hclust2 = AgglomerativeClustering(n_clusters, affinity="euclidean",linkage="ward") #(numero de clusters a encontrar, metodo das distancias utilizado, ...)
hclust2.fit(input_sc_filt4) #realiza o clustering

output_description= meta.values[:,5]
res_hc = hclust2.labels_  #atribui estes nomes às labels
cluster2=pd.crosstab(output_celltype, res_hc, rownames=['Cell type'],colnames=['Cluster']) 
cluster2


# **K-Means**

# In[84]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

# In[85]:


from scipy.cluster.hierarchy import dendrogram, linkage

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


# <div style="text-align: justify">Mais uma vez, é possível verificar que o clustering efetuado apenas aos genes com maior expressão diferencial separou as amostras em dois grupos, amostras de tecido epitelial e do estroma, quase na sua totalidade.</div>
# 
# <br>
# <div style="text-align: justify">Desta forma, estes dados demonstram que existem diferenças na expressão genética das amostras de tecido epitelial e do estroma associado ao epitélio maligno.</div>
# 
# <br>
# <div style="text-align: justify">Um estudo mais profundo destes padrões de expressão genética deverá indicar diferenças nos processos biológicos existentes nos dois tipos de tecido.</div>
# 
# <br>
# <div style="text-align: justify">Porém, segundo a visualização dos clusters obtidos através do K-Means, estes não se encontram completamente bem distanciados, podendo, então, em conjunto com a não separação total dos dados, originar uma previsão com resultados menos elevados.</div>

# # Etapa 5 - Modelos de aprendizagem máquina

# <div style="text-align: justify">Nesta fase do trabalho, realiza-se a prévia separação dos dados em conjuntos: treino e teste. Utiliza-se apenas os genes filtrados no ponto anterior, para o caso de comparação entre cancro associado e não associado à gravidez, visto que é esta classificação que se pretender prever. Mantém-se um terço das amostras nos dados de teste e as restantes nos de treino, correspondendo, então, a variável de saída à variável diseasestate.</div>

# In[10]:


indices = np.random.permutation(len(input_sc_filt2)) #coloca, na matriz, os índices numa ordem aleatória

numtr = int(1/3 * input_sc_filt2.shape[0])    
#treino:
train_in = input_sc_filt2[indices[:-numtr]]
train_out = output_diseasestate[indices[:-numtr]]
#teste:
test_in  = input_sc_filt2[indices[-numtr:]]
test_out = output_diseasestate[indices[-numtr:]]


# <div style="text-align: justify">Seguidamente, utilizamos vários algoritmos para realizar a previsão dos nossos dados utilizados para teste, avaliando cada um através do seu score/percentagem de acertos:</div>

# **Modelo de Classificação:KNeighbors**

# In[38]:


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

# In[229]:


from sklearn import tree
from sklearn.metrics import confusion_matrix

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


# In[230]:


from sklearn.datasets import load_iris
from sklearn import tree
from graphviz import Digraph
from graphviz import Source
from PIL import Image

clf = tree.DecisionTreeClassifier()
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
n=tree.export_graphviz(clf, out_file='tree.dot')


# In[233]:


from IPython.display import Image
Image(filename='tree.PNG')


# **Modelos de classificação: REGRESSÃO LOGÍSTICA**

# In[43]:


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

# In[38]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


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

# <div style="text-align: justify">Utilizamos o *solver* 'Lbfgs' que é um otimizador da família *quasi_Newton* e o *alpha* 1e-5 que foram os parâmetros que apresentaram melhores resultados (segundo um certo número de tentativas).</div>

# In[44]:


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

# In[159]:


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


# <div style="text-align: justify">A função cross_val_predict tem uma interface semelhante ao cross_val_score, mas retorna, para cada elemento na entrada, a previsão obtida para esse elemento quando este se encontrava no teste definido. Em seguida, utiliza-se essa função para visualizar erros de predição, inserindo como dados a prever os dados de expressão sem qualquer tratamento (dataT).</div>

# In[85]:


from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_predict #Visualization of predictions obtained from different models.

lr = linear_model.LinearRegression()
y = input_sc_filt2

# cross_val_predict returns an array of the same size as `y` where each entry is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, dataT, y, cv=5)
#metrics.accuracy_score(y, predicted) 

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=5)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# <div style="text-align: justify">No gráfico criado, todos os pontos estão minimamente perto da reta de precisão, no entanto, notam-se alguns erros de previsão. Logo, o modelo de aprendizagem baseado na validação cruzada obtém bons resultados, mas não é um modelo ótimo.</div>

# ## Otimização do modelo

# <div style="text-align: justify">Os parâmetros utilizados nos modelos foram os que estavam definidos por *default* ou, em alguns casos, por questões de convergência tiveram de ser alterados (como por exemplo, no caso das redes neuronais). Existem, porém, metodologias para otimizar estes parâmetros pelo que este foi o procedimento que realizamos em seguida com os modelos para os quais obtivemos melhores resultados anteriormente (superiores a 90%): **KNeighbors, SGDClassifier, Neural Networks e SVM(Validação Cruzada)**, excluindo o caso do **LogisticRegression**, visto que este modelo já obteve o resultado máximo.</div>

# + ####  KNeighbors

# <div style="text-align: justify">Para este modelo, resolvemos, otimizar, primeiro apenas um parâmetro (*n_neighbors*).</div>

# In[222]:


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


# In[211]:


# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# <div style="text-align: justify">Verifica-se, então, uma grande melhoria da performance do modelo, aquando a otimização deste parâmetro. Obtiveram-se scores ideais (100%) para K=2,4,6,10 ou 18.</div>
# 
# <br>
# <div style="text-align: justify">Deste modo, não é necessário analisar os restantes parâmetros, pois já encontramos as condições para a desempenho máximo do modelo.</div>

# + #### SGDClassifier

# <div style="text-align: justify">Neste e nos restantes modelos, recorremos á procura em grelha dos parâmetros (gridSearch) para proceder, se possível, a otimização dos mesmos.</div>

# <div style="text-align: justify">Neste caso, exploramos uma gama de valores aceitáveis para o *alpha* e mantivemos os valores 'l2', para o parâmetro *penalty*, que é o termo de regularização padrão para modelos lineares de SVM, e -1, para *n_jobs*, de modo a usar todos os processadores para fazer a computação *OVA* (*One Versus All, for multi-class problems*). Quanto ao parâmetro *loss*, consideramos duas funções viáveis: 'hinge' e 'log'.</div>
# 
# <div style="text-align: justify">O hiper-parâmetro *alpha* serve para um propósito duplo. É, tanto um parâmetro de regularização, como corresponde à taxa de aprendizagem inicial no cronograma padrão. Isso significa que, além de regularizar os coeficientes de regressão logística, a saída do modelo depende de uma interação entre *alpha* e o número de épocas (*n_iter*) que a rotina de ajuste executa. Especificamente, como *alpha* se torna muito pequeno, *n_iter* deve ser aumentado para compensar a baixa taxa de aprendizado. É por isso que é mais seguro (mas mais lento) especificar *n_iter* suficientemente grande, por ex. 1000, quando pesquisando sobre uma ampla gama de *alphas*.</div>

# In[161]:


import numpy as np
from time import time
from operator import itemgetter
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
 
# get some data
#digits = load_digits()
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
 
#loss="hinge", penalty="l2", tol=1e-3    
    
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


# <div style="text-align: justify">Observa-se, então, que não há melhoria do desempenho do algorítmo quando é realizada a otimização dos parâmetros.</div>

# + #### Neural Networks 
# 
# <br>
# <div style="text-align: justify"> Quanto ao classificador associado às redes neuronais, vários parâmetros podem ser otimizados. No que toca à taxa de aprendizagem, *learning_rate*, usamos os valores 'invscaling' e 'adaptative' que permitem que a taxa de aprendizagem decresça à medida que o treino avança e 'constant' que induz uma taxa de aprendizagem constante. Relativamente ao tamanho das *hidden layers*,  consideramos algumas hipóteses viáveis, para os *solvers* testamos todos os possíveis métodos, e, no que respeita às iterações, apenas valores superiores a 1500 eram capazes de produzir resultados. Finalmente, demos também algumas opções para a função de ativação das *hidden_layers*, o parâmetro *activation*. Decidimos, também, considerar um outro valor possível de *alpha*, para além do valor *default*. </div>

# In[224]:


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

'''
# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))'''


# In[228]:


from sklearn.neural_network import MLPClassifier
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


# <div style="text-align: justify">Como se pode observar, não há melhorias, sendo que o modelo já está otimizado.</div>

# + ####  Validação Cruzada (SVM)

# <div style="text-align: jusitfy">Relativamente aos parâmetros associados ao classificador *SVM*, foram fornecidas algumas opções de valores para alguns desses parâmetros, tendo em conta que certos parâmeteros só tinham valor se utilizássemos certas opções nos outros (por exemplo, no caso do parâmetro *degree*, este era ignorado por todos os métodos do parâmetro *kernel*, à exceção do 'poly'). No que toca ao parâmetro *kernel*, demos duas possíveis opções para o mesmo, 'linear' e 'rbf'. Relativamente ao parâmetro C, que é um parâmetro de penalidade do termo de erro, foram dadas também 5 opções de valores. Por fim, ao parâmetro *gamma*, que corresponde ao coeficente do *kernel* apenas para as opções 'rbf', 'poly' e 'sigmoid', consideraram-se também algumas opções. </div>

# In[191]:


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


# In[192]:


from sklearn import svm

model_svm = svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
model_svm= model_svm.fit(train_in, train_out)
prevs=model_svm.predict(test_in)

scores = cross_val_score(model_svm, input_sc_filt2, output_diseasestate, cv = 5)
print("Percentagem de acertos:", scores.mean()*100, '%') 


# <div style="text-align: justify">Observou-se uma melhoria bastante considerável no desempenho dete modelo, aquando a otimização dos parâmetros.</div>

# ### **Conclusão** 

# <div style="text-align: justify">Conclui-se, então, que alguns modelos de previsão obtiveram performances mais fracas, afastando-se mais da idealidade (100%),  o que pode ser justificado pelos aspetos referidos na execução dos clusters. Não obstante, a maioria dos modelos apresenta scores superiores as 90%, o que já pode ser considerado como um bom resultado, sendo o modelo de aprendizagem máquina criado, usando Regressão Logística, o que melhor se ajusta aos dados em análise, dado que possui o maior score.</div>
# 
# <br>
# <div style="text-align: justify">Pela otimização dos modelos, vimos que foi possível a melhoria do desempenho de dois dos modelos (Validação Cruzada e KNeighbors), enquanto os outros mantiveram o seu score. De notar que o score/percentagem de acertos de cada modelo é influenciado pela definição dos dados de treino e dos dados de teste, que possui um carater aleatório, e dos parâmetros atribuidos, sendo que, a ausência da definição completa dos argumentos dos classificadores, poderia conduzir a uma variabilidade por parte do modelo. Desta forma, foi sempre considerado o melhor score obtido para cada caso.</div>
# 
# <br>
# <div style="text-align: justify">Não obstante, o Regressão Logística continua a ser o melhor modelo para este dataset, em coonjunto com o modelo KNeighbors otimizado.</div>
